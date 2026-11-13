"""
Model logic for the perceived-emotion system.

  frozen AudioEncoder (Wav2Vec2 emotion)        -> 1024-d per clip ─┐
                                                                    ├─> FusionHead (trainable) -> valence, arousal in [-1, 1]
  frozen VideoEncoder (EmotionCLIP-V2, ViT-B/32) -> 512-d per clip ─┘

Loss: 1 - CCC (concordance correlation coefficient), averaged over valence and arousal.
"""
import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from common import AUDIO_MODEL_ID, EMOCLIP_REPO, EMOCLIP_WEIGHTS, EMOTIONS, TOKENIZER_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- frozen audio encoder -------------------------
class _RegressionHead(nn.Module):              # as on the audeering model card
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = torch.tanh(self.dense(self.dropout(x)))
        return self.out_proj(self.dropout(x))


class _EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        return hidden, self.classifier(hidden)  # 1024-d embedding, [arousal, dominance, valence]


class AudioEncoder:
    """Wav2Vec2 emotion model. Returns (1024-d embedding, its own [arousal, dominance, valence] in ~[0, 1])."""
    dim = 1024

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_ID, local_files_only=True)
        self.model = _EmotionModel.from_pretrained(AUDIO_MODEL_ID, local_files_only=True).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, wav_path):
        x, sr = sf.read(wav_path)
        if x.ndim > 1:
            x = x.mean(axis=1)                  # stereo -> mono
        assert sr == 16000, f"{wav_path}: audio must be 16 kHz (prep_clips.py does this)"
        inp = self.processor(x, sampling_rate=sr, return_tensors="pt").input_values.to(DEVICE)
        hidden, adv = self.model(inp)
        return hidden[0].cpu().numpy(), adv[0].cpu().numpy()


# ------------------------- frozen video encoder -------------------------
# EmotionCLIP-V2 (huggingface.co/jiangchengchengNLP/EmotionCLIP-V2): OpenAI CLIP ViT-B/32 + learned
# "prefix" key/value tokens in every attention layer (+ 20 prompt tokens on the text side), trained on
# EmoSet-118K. The official code needs a GPU + pickled helpers, so the same network is re-written here
# in plain float32 PyTorch (parameter names match the .pth file, so the weights load with strict=True).
class _Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, pk, pv):
        b, n, d = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        k, v = torch.cat([pk.expand(b, -1, -1), k], 1), torch.cat([pv.expand(b, -1, -1), v], 1)
        split = lambda t: t.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v))   # no mask, as in the original
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = _Attn(d, heads)
        self.mlp = nn.Module()
        self.mlp.c_fc, self.mlp.c_proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, pk, pv):
        x = x + self.attn(self.ln_1(x), pk, pv)
        h = self.mlp.c_fc(self.ln_2(x))
        return x + self.mlp.c_proj(h * torch.sigmoid(1.702 * h))                # QuickGELU


class _Transformer(nn.Module):
    def __init__(self, d, heads, layers, n_prefix=20):
        super().__init__()
        self.resblocks = nn.ModuleList(_Block(d, heads) for _ in range(layers))
        self.prefix = nn.Module()
        self.prefix.prefix_embedding = nn.Parameter(torch.empty(1, n_prefix, layers * 2 * d))
        self.register_buffer("prefix_tokens", torch.arange(n_prefix))
        self.shape = (n_prefix, layers, 2, d)

    def forward(self, x):
        pk, pv = self.prefix.prefix_embedding.view(self.shape).permute(2, 1, 0, 3)   # each: (layers, 20, d)
        for i, block in enumerate(self.resblocks):
            x = block(x, pk[i], pv[i])
        return x


class _EmotionCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        v = self.visual
        v.conv1 = nn.Conv2d(3, 768, 32, 32, bias=False)
        v.class_embedding = nn.Parameter(torch.empty(768))
        v.positional_embedding = nn.Parameter(torch.empty(50, 768))
        v.ln_pre, v.ln_post = nn.LayerNorm(768), nn.LayerNorm(768)
        v.transformer = _Transformer(768, 12, 12)
        v.proj = nn.Parameter(torch.empty(768, 512))
        self.token_embedding = nn.Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(77, 512))
        self.transformer = _Transformer(512, 8, 12)
        self.ln_final = nn.LayerNorm(512)
        self.text_projection = nn.Parameter(torch.empty(512, 512))
        self.prompt_block = nn.Module()
        self.prompt_block.prompt_embedding = nn.Embedding(20, 512)
        self.logit_scale = nn.Parameter(torch.empty([]))

    def encode_image(self, img):                                               # (b, 3, 224, 224) -> (b, 512)
        v = self.visual
        x = v.conv1(img).flatten(2).transpose(1, 2)
        x = torch.cat([v.class_embedding.expand(len(x), 1, -1), x], 1) + v.positional_embedding
        x = v.transformer(v.ln_pre(x))
        return v.ln_post(x[:, 0]) @ v.proj

    def encode_text(self, ids):                                                # (b, 57) token ids -> (b, 512)
        eot = ids.argmax(-1) + 20                                              # +20 prompt tokens inserted after <sot>
        x = self.token_embedding(ids)
        prompt = self.prompt_block.prompt_embedding.weight.expand(len(x), -1, -1)
        x = torch.cat([x[:, :1], prompt, x[:, 1:]], 1) + self.positional_embedding
        x = self.ln_final(self.transformer(x))
        return x[torch.arange(len(x)), eot] @ self.text_projection


class VideoEncoder:
    """EmotionCLIP-V2 on 1 frame per second (middle of each second), averaged over the clip.
    Returns (512-d emotion-tuned embedding, probabilities of the 9 EMOTIONS)."""
    dim = 512

    def __init__(self):
        if not EMOCLIP_WEIGHTS.exists():
            hf_hub_download(EMOCLIP_REPO, EMOCLIP_WEIGHTS.name, local_dir=EMOCLIP_WEIGHTS.parent)
        self.model = _EmotionCLIP()
        self.model.load_state_dict(torch.load(EMOCLIP_WEIGHTS, map_location="cpu", weights_only=True))
        self.model = self.model.float().to(DEVICE).eval()
        self.preprocess = transforms.Compose([                                  # standard CLIP preprocessing
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        tok = CLIPTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
        ids = torch.zeros(len(EMOTIONS), 57, dtype=torch.long)                 # zero padding, as in the original
        for i, e in enumerate(EMOTIONS):
            t = tok(f"This picture conveys a sense of {e}")["input_ids"]
            ids[i, :len(t)] = torch.tensor(t)
        with torch.no_grad():
            self.text = F.normalize(self.model.encode_text(ids.to(DEVICE)), dim=-1)

    @torch.no_grad()
    def __call__(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = round(cap.get(cv2.CAP_PROP_FPS)) or 25
        frames, i = [], 0
        ok, f = cap.read()
        while ok:
            if i % fps == fps // 2:
                frames.append(self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
            ok, f = cap.read()
            i += 1
        cap.release()
        img = F.normalize(self.model.encode_image(torch.stack(frames).to(DEVICE)), dim=-1)
        probs = (self.model.logit_scale.exp() * img @ self.text.T).softmax(-1)
        return img.mean(0).cpu().numpy(), probs.mean(0).cpu().numpy()


# ------------------------- trainable head + loss -------------------------
class FusionHead(nn.Module):
    """Each modality -> 64-d (Dropout-Linear-ReLU-Dropout), concatenated -> Linear -> tanh -> (valence, arousal).
    Heavy dropout because there are few clips (hundreds) and many input features (1024 + 512)."""

    def __init__(self, dims, hidden=64, dropout=0.5):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)) for d in dims)
        self.out = nn.Sequential(nn.Linear(hidden * len(dims), 2), nn.Tanh())

    def forward(self, xs):
        return self.out(torch.cat([b(x) for b, x in zip(self.branches, xs)], dim=1))


class PerceivedVA_Model(nn.Module):
    """
    Decoupled Cross-Modal Gated Attention Network (MGN)
    Multimodal fusion architecture for continuous valence and arousal prediction:
      - Video: 512-d (EmotionCLIP-V2) -> Linear(64) -> BiLSTM(128)
      - Audio: 1024-d (Wav2Vec2)      -> Linear(64) -> BiLSTM(128)
      - Bidirectional Cross-Modal Attention (V <-> A, 4 heads)
      - Time-varying Modality Gating (g_v, g_a)
      - Temporal Attention Pooling over clip sequence
      - Genre Conditioning (9-d -> 32-d embedding)
      - Output: Valence & Arousal in [-1, 1] (via Tanh)
    """
    def __init__(self, video_dim=512, audio_dim=1024, hidden_dim=64, genre_dim=9, genre_emb_dim=32, num_heads=4, dropout=0.3):
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.genre_dim = genre_dim
        self.genre_emb_dim = genre_emb_dim

        # 1. Feature Projections
        self.video_proj = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # 2. Temporal Context Encoders (BiLSTM)
        self.video_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        fusion_dim = hidden_dim * 2  # bidirectional

        # 3. Bidirectional Cross-Modal Attention
        self.cross_attn_va = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_av = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        # 4. Modality Gating
        self.gate_v = nn.Linear(fusion_dim, fusion_dim)
        self.gate_a = nn.Linear(fusion_dim, fusion_dim)

        # 5. Scene / Temporal Attention Pooling
        self.scene_attention = nn.Linear(fusion_dim, 1)

        # 6. Genre Projection
        if genre_dim > 0:
            self.genre_proj = nn.Sequential(
                nn.Linear(genre_dim, genre_emb_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            reg_in_dim = fusion_dim + genre_emb_dim
        else:
            self.genre_proj = None
            reg_in_dim = fusion_dim

        # 7. Regression Head (mapped to [-1, 1] for COGNIMUSE)
        self.fc1 = nn.Sequential(
            nn.Linear(reg_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, video, audio, mask=None, genre=None, return_gates=False):
        # Support 1D (D,), 2D (B, D) and 3D (B, T, D)
        if video.dim() == 1:
            video = video.unsqueeze(0).unsqueeze(0)
        elif video.dim() == 2:
            video = video.unsqueeze(1)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        b, t, _ = video.shape

        v = self.video_proj(video)
        a = self.audio_proj(audio)

        v, _ = self.video_lstm(v)
        a, _ = self.audio_lstm(a)

        key_padding_mask = (mask == 0) if mask is not None else None

        # Cross-modal attention (V attends to A, A attends to V)
        v_attn, _ = self.cross_attn_va(v, a, a, key_padding_mask=key_padding_mask)
        a_attn, _ = self.cross_attn_av(a, v, v, key_padding_mask=key_padding_mask)

        # Modality Gating
        g_v = torch.sigmoid(self.gate_v(v_attn))
        g_a = torch.sigmoid(self.gate_a(a_attn))
        fused = g_v * v_attn + g_a * a_attn

        # Temporal delta transition
        if t > 1:
            delta = fused[:, 1:, :] - fused[:, :-1, :]
        else:
            delta = torch.zeros(1, device=video.device)

        # Attention pooling over time
        scores = self.scene_attention(fused).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if t > 1:
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            clip_repr = torch.sum(weights * fused, dim=1)
        else:
            clip_repr = fused.squeeze(1)

        # Genre conditioning
        if self.genre_proj is not None:
            if genre is None:
                genre = torch.zeros((b, self.genre_dim), device=video.device)
            elif genre.dim() == 1:
                genre = genre.unsqueeze(0).expand(b, -1)
            g_emb = self.genre_proj(genre)
            clip_repr = torch.cat([clip_repr, g_emb], dim=-1)

        feat = self.fc1(clip_repr)
        va = self.fc2(feat)

        if return_gates:
            gv = g_v.mean().item()
            ga = g_a.mean().item()
            tot = gv + ga + 1e-8
            return va, delta, (gv / tot, ga / tot)

        return va, delta


def ccc(pred, gt):
    """Concordance correlation coefficient per column (valence, arousal)."""
    pm, gm = pred.mean(0), gt.mean(0)
    cov = ((pred - pm) * (gt - gm)).mean(0)
    return 2 * cov / (pred.var(0, unbiased=False) + gt.var(0, unbiased=False) + (pm - gm) ** 2 + 1e-8)


def ccc_loss(pred, gt):
    """Lin's Concordance Correlation Coefficient loss (1 - mean(CCC_v, CCC_a))."""
    return 1.0 - ccc(pred, gt).mean()
