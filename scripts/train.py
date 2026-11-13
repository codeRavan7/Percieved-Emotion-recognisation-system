"""
TRAINING: learn valence/arousal from cached clip features.
Supports both:
  1. --arch mgn  : Decoupled Cross-Modal Gated Attention Network (BiLSTM + Cross-Attn + Gating + Genre)
  2. --arch mlp  : Simple baseline FusionHead (concatenation + MLP).

Train / validation split:
  --val_movie CODE  -> leave that whole movie out for validation (e.g. --val_movie BMI)
  (default)         -> per movie, first 80 % of clips = train, last 20 % = validation

Run:
  python scripts/train.py --arch mgn --length 10 --label gauss
  python scripts/train.py --arch mlp --length 10 --label gauss
"""
import argparse
import sys

import numpy as np
import torch

from common import FEAT_DIR, MODEL_DIR, get_genre_vector
from models import FusionHead, PerceivedVA_Model, ccc, ccc_loss

p = argparse.ArgumentParser()
p.add_argument("--arch", choices=["mgn", "mlp"], default="mgn", help="Model architecture (mgn: Cross-Modal Attention, mlp: baseline)")
p.add_argument("--length", type=int, default=10, help="clip length in seconds (10 or 3)")
p.add_argument("--modality", choices=["audio", "video", "both"], default="both")
p.add_argument("--label", choices=["mean", "gauss"], default="gauss", help="Average or Gaussian ground truth")
p.add_argument("--val_movie", default=None, help="movie code held out for validation, e.g. BMI")
p.add_argument("--hidden_dim", type=int, default=64, help="latent dimension for BiLSTM / projections")
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--patience", type=int, default=100, help="stop if validation CCC does not improve")
args = p.parse_args()
torch.manual_seed(0)

if args.arch == "mgn" and args.modality != "both":
    sys.exit("MGN architecture requires --modality both (audio + video).")

MODS = ["audio", "video"] if args.modality == "both" else [args.modality]

# ---------------- data loading + split ----------------
files = sorted(FEAT_DIR.glob(f"*_{args.length}s.npz"))
if not files:
    sys.exit(f"No features in {FEAT_DIR} for {args.length} s clips. Run extract_features.py first.")

train = {m: [] for m in MODS + ["y", "genre"]}
val = {m: [] for m in MODS + ["y", "genre"]}

for f in files:
    d = np.load(f)
    movie = f.stem.rsplit("_", 1)[0]
    n = len(d["clip"])
    g_arr = np.tile(get_genre_vector(movie), (n, 1))

    if args.val_movie:
        parts = [(val, slice(0, n))] if movie == args.val_movie else [(train, slice(0, n))]
    else:
        cut = int(0.8 * n)
        parts = [(train, slice(0, cut)), (val, slice(cut, n))]

    for split, idx in parts:
        for m in MODS:
            split[m].append(d[m][idx])
        split["y"].append(d[f"labels_{args.label}"][idx])
        split["genre"].append(g_arr[idx])

if not val["y"] or not train["y"]:
    sys.exit(f"Train or validation set is empty (is --val_movie {args.val_movie} in {FEAT_DIR}, "
             f"and is there at least one other movie?)")

to_t = lambda split: {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in split.items()}
train, val = to_t(train), to_t(val)
print(f"Arch: {args.arch.upper()} | Train clips: {len(train['y'])}, Validation clips: {len(val['y'])}, Modalities: {MODS}")

# standardise features with training statistics
mu = {m: train[m].mean(0) for m in MODS}
sd = {m: train[m].std(0) + 1e-6 for m in MODS}
xs_train = {m: (train[m] - mu[m]) / sd[m] for m in MODS}
xs_val = {m: (val[m] - mu[m]) / sd[m] for m in MODS}

# ---------------- model setup ----------------
if args.arch == "mgn":
    model = PerceivedVA_Model(
        video_dim=512,
        audio_dim=1024,
        hidden_dim=args.hidden_dim,
        genre_dim=9,
        genre_emb_dim=32,
        dropout=0.3
    )
else:
    model = FusionHead([train[m].shape[1] for m in MODS], hidden=args.hidden_dim)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
best, best_state, wait = -1.0, None, 0

# ---------------- training loop ----------------
for epoch in range(args.epochs):
    model.train()
    if args.arch == "mgn":
        pred, _ = model(video=xs_train["video"], audio=xs_train["audio"], genre=train["genre"])
    else:
        pred = model([xs_train[m] for m in MODS])
    loss = ccc_loss(pred, train["y"])

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    model.eval()
    with torch.no_grad():
        if args.arch == "mgn":
            val_pred, _ = model(video=xs_val["video"], audio=xs_val["audio"], genre=val["genre"])
        else:
            val_pred = model([xs_val[m] for m in MODS])
        val_ccc = ccc(val_pred, val["y"]).mean().item()

    if val_ccc > best:
        best = val_ccc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= args.patience:
            break

    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  train loss {loss.item():.4f}  val CCC {val_ccc:.4f}")

# ---------------- evaluation + save ----------------
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    if args.arch == "mgn":
        pred, _, (gv, ga) = model(video=xs_val["video"], audio=xs_val["audio"], genre=val["genre"], return_gates=True)
        print(f"\nAverage Modality Attention Gates on Validation: Video={gv*100:.1f}%, Audio={ga*100:.1f}%")
    else:
        pred = model([xs_val[m] for m in MODS])

c = ccc(pred, val["y"])
rmse = ((pred - val["y"]) ** 2).mean(0).sqrt()
r = [np.corrcoef(pred[:, k].cpu().numpy(), val["y"][:, k].cpu().numpy())[0, 1] for k in range(2)]

print(f"\nBest Validation Results ({args.arch.upper()}, {args.modality}, {args.length}s, {args.label} labels):")
for k, dim in enumerate(["valence", "arousal"]):
    print(f"  {dim:8s}  CCC: {c[k]:.3f}   Pearson r: {r[k]:.3f}   RMSE: {rmse[k]:.3f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
out = MODEL_DIR / f"{args.arch}_{args.modality}_{args.length}s_{args.label}.pt"
save_dict = {
    "arch": args.arch,
    "state_dict": model.state_dict(),
    "modalities": MODS,
    "mu": mu,
    "sd": sd,
    "hidden_dim": args.hidden_dim,
}
torch.save(save_dict, out)
print(f"Saved checkpoint: {out}")
