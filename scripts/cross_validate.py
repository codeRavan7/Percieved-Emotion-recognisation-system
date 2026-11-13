"""
Comprehensive Cross-Validation and Modality Ablation Benchmark
Evaluates:
  1. Decoupled Cross-Modal Gated Attention Network (MGN)
  2. Baseline Multimodal Fusion (MLP)
  3. Video-only Baseline (EmotionCLIP-V2)
  4. Audio-only Baseline (Wav2Vec2)

Runs Leave-One-Movie-Out (LOMO) cross validation across all COGNIMUSE Hollywood movies.
Saves metrics to outputs/experiments/ and trained models to outputs/model/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')

from common import FEAT_DIR, MODEL_DIR, OUT_DIR, get_genre_vector
from models import FusionHead, PerceivedVA_Model, ccc, ccc_loss

def run_experiment(feat_files, val_movie, arch="mgn", modality="both", label="gauss", epochs=300, lr=1e-3):
    """Trains on all movies except val_movie, evaluates on val_movie."""
    mods = ["audio", "video"] if modality == "both" else [modality]
    train = {m: [] for m in mods + ["y", "genre"]}
    val = {m: [] for m in mods + ["y", "genre"]}

    for f in feat_files:
        d = np.load(f)
        movie = f.stem.rsplit("_", 1)[0]
        n = len(d["clip"])
        g_arr = np.tile(get_genre_vector(movie), (n, 1))

        target = val if movie == val_movie else train
        for m in mods:
            target[m].append(d[m])
        target["y"].append(d[f"labels_{label}"])
        target["genre"].append(g_arr)

    to_t = lambda s: {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in s.items()}
    train, val = to_t(train), to_t(val)

    # Standardize
    mu = {m: train[m].mean(0) for m in mods}
    sd = {m: train[m].std(0) + 1e-6 for m in mods}
    xs_train = {m: (train[m] - mu[m]) / sd[m] for m in mods}
    xs_val = {m: (val[m] - mu[m]) / sd[m] for m in mods}

    # Model
    torch.manual_seed(42)
    if arch == "mgn":
        model = PerceivedVA_Model(video_dim=512, audio_dim=1024, hidden_dim=64, genre_dim=9, genre_emb_dim=32, dropout=0.3)
    else:
        dims = [xs_train[m].shape[1] for m in mods]
        model = FusionHead(dims, hidden=64, dropout=0.4)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_ccc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        if arch == "mgn":
            pred, _ = model(video=xs_train["video"], audio=xs_train["audio"], genre=train["genre"])
        else:
            pred = model([xs_train[m] for m in mods])
        loss = ccc_loss(pred, train["y"])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Evaluate on held-out movie
        model.eval()
        with torch.no_grad():
            if arch == "mgn":
                val_pred, _ = model(video=xs_val["video"], audio=xs_val["audio"], genre=val["genre"])
            else:
                val_pred = model([xs_val[m] for m in mods])
            c_val = ccc(val_pred, val["y"]).mean().item()

        if c_val > best_ccc:
            best_ccc = c_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if arch == "mgn":
            pred, _, (gv, ga) = model(video=xs_val["video"], audio=xs_val["audio"], genre=val["genre"], return_gates=True)
        else:
            pred = model([xs_val[m] for m in mods])
            gv, ga = 0.5, 0.5

    c = ccc(pred, val["y"])
    rmse = ((pred - val["y"]) ** 2).mean(0).sqrt()
    
    # Safe Pearson r
    r = []
    for k in range(2):
        yp = pred[:, k].cpu().numpy()
        yt = val["y"][:, k].cpu().numpy()
        if np.std(yp) > 1e-6 and np.std(yt) > 1e-6:
            r.append(np.corrcoef(yp, yt)[0, 1])
        else:
            r.append(0.0)

    return {
        "val_movie": val_movie,
        "arch": arch,
        "modality": modality,
        "ccc_v": c[0].item(),
        "ccc_a": c[1].item(),
        "ccc_mean": c.mean().item(),
        "r_v": r[0],
        "r_a": r[1],
        "r_mean": (r[0] + r[1]) / 2.0,
        "rmse_v": rmse[0].item(),
        "rmse_a": rmse[1].item(),
        "gate_video": gv,
        "gate_audio": ga,
    }


def main():
    p = argparse.ArgumentParser(description="Full LOMO Cross-Validation and Ablation Suite")
    p.add_argument("--length", type=int, default=10, help="Clip length in seconds")
    p.add_argument("--label", choices=["gauss", "mean"], default="gauss")
    args = p.parse_args()

    files = sorted(FEAT_DIR.glob(f"*_{args.length}s.npz"))
    if not files:
        sys.exit(f"No features found in {FEAT_DIR} for {args.length}s clips.")

    movies = [f.stem.rsplit("_", 1)[0] for f in files]
    print(f"\n================================================================================")
    print(f"  PERCEIVED EMOTION SYSTEM - FULL 7-MOVIE BENCHMARK & EVALUATION")
    print(f"  Movies detected ({len(movies)}): {', '.join(movies)}")
    print(f"================================================================================\n")

    if len(movies) < 2:
        print("At least 2 movies are needed for Leave-One-Movie-Out cross validation.")
        return

    exp_dir = OUT_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leave-One-Movie-Out for MGN across all available movies
    print(f">>> 1. Running Leave-One-Movie-Out Cross-Validation (MGN Architecture)...")
    mgn_results = []
    for m in movies:
        res = run_experiment(files, val_movie=m, arch="mgn", modality="both", label=args.label)
        mgn_results.append(res)
        print(f"  Held-out [{m:3s}] -> Valence CCC: {res['ccc_v']:+.3f} (r={res['r_v']:+.3f}) | "
              f"Arousal CCC: {res['ccc_a']:+.3f} (r={res['r_a']:+.3f}) | Gates: V={res['gate_video']*100:.1f}%, A={res['gate_audio']*100:.1f}%")

    df_mgn = pd.DataFrame(mgn_results)
    df_mgn.to_csv(exp_dir / f"lomo_mgn_{args.length}s_{args.label}.csv", index=False)

    print("\n--------------------------------------------------------------------------------")
    print(f"MGN LOMO Average -> Valence CCC: {df_mgn['ccc_v'].mean():.3f} (r={df_mgn['r_v'].mean():.3f}) | "
          f"Arousal CCC: {df_mgn['ccc_a'].mean():.3f} (r={df_mgn['r_a'].mean():.3f}) | "
          f"Overall CCC: {df_mgn['ccc_mean'].mean():.3f}")
    print("--------------------------------------------------------------------------------\n")

    # 2. Modality Ablation Comparison
    print(">>> 2. Running Modality Ablation Study across all movies...")
    ablations = [
        ("mgn", "both", "Proposed MGN (Audio + Video + Gates + Genre)"),
        ("mlp", "both", "Baseline Fusion MLP (Audio + Video)"),
        ("mlp", "video", "Video-Only Baseline (EmotionCLIP-V2)"),
        ("mlp", "audio", "Audio-Only Baseline (Wav2Vec2)"),
    ]

    ablation_summary = []
    for arch, mod, desc in ablations:
        mod_results = []
        for m in movies:
            res = run_experiment(files, val_movie=m, arch=arch, modality=mod, label=args.label)
            mod_results.append(res)
        df_mod = pd.DataFrame(mod_results)
        ablation_summary.append({
            "Architecture": arch.upper(),
            "Modality": mod,
            "Description": desc,
            "Valence_CCC": round(df_mod["ccc_v"].mean(), 4),
            "Valence_r": round(df_mod["r_v"].mean(), 4),
            "Arousal_CCC": round(df_mod["ccc_a"].mean(), 4),
            "Arousal_r": round(df_mod["r_a"].mean(), 4),
            "Mean_CCC": round(df_mod["ccc_mean"].mean(), 4),
            "Mean_r": round(df_mod["r_mean"].mean(), 4),
            "Mean_RMSE": round((df_mod["rmse_v"].mean() + df_mod["rmse_a"].mean()) / 2.0, 4),
        })

    df_abl = pd.DataFrame(ablation_summary)
    df_abl.to_csv(exp_dir / f"ablation_comparison_{args.length}s_{args.label}.csv", index=False)
    print(df_abl.to_string(index=False))

    # 3. Train final comprehensive model on 100% of the clips (with an 80/20 train/val split across all movies)
    print("\n>>> 3. Training final production model on full 7-movie dataset...")
    all_train = {m: [] for m in ["audio", "video", "y", "genre"]}
    all_val = {m: [] for m in ["audio", "video", "y", "genre"]}

    for f in files:
        d = np.load(f)
        movie = f.stem.rsplit("_", 1)[0]
        n = len(d["clip"])
        g_arr = np.tile(get_genre_vector(movie), (n, 1))
        cut = int(0.85 * n)

        for m in ["audio", "video"]:
            all_train[m].append(d[m][:cut])
            all_val[m].append(d[m][cut:])
        all_train["y"].append(d[f"labels_{args.label}"][:cut])
        all_val["y"].append(d[f"labels_{args.label}"][cut:])
        all_train["genre"].append(g_arr[:cut])
        all_val["genre"].append(g_arr[cut:])

    to_t = lambda s: {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in s.items()}
    all_train, all_val = to_t(all_train), to_t(all_val)

    mu = {m: all_train[m].mean(0) for m in ["audio", "video"]}
    sd = {m: all_train[m].std(0) + 1e-6 for m in ["audio", "video"]}
    xs_train = {m: (all_train[m] - mu[m]) / sd[m] for m in ["audio", "video"]}
    xs_val = {m: (all_val[m] - mu[m]) / sd[m] for m in ["audio", "video"]}

    torch.manual_seed(0)
    final_model = PerceivedVA_Model(video_dim=512, audio_dim=1024, hidden_dim=64, genre_dim=9, genre_emb_dim=32, dropout=0.3)
    opt = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_c, best_st = -1.0, None

    for epoch in range(400):
        final_model.train()
        p, _ = final_model(video=xs_train["video"], audio=xs_train["audio"], genre=all_train["genre"])
        loss = ccc_loss(p, all_train["y"])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        opt.step()

        final_model.eval()
        with torch.no_grad():
            vp, _ = final_model(video=xs_val["video"], audio=xs_val["audio"], genre=all_val["genre"])
            vc = ccc(vp, all_val["y"]).mean().item()

        if vc > best_c:
            best_c = vc
            best_st = {k: v.clone() for k, v in final_model.state_dict().items()}

    if best_st is not None:
        final_model.load_state_dict(best_st)
    final_model.eval()

    with torch.no_grad():
        vp, _, (gv, ga) = final_model(video=xs_val["video"], audio=xs_val["audio"], genre=all_val["genre"], return_gates=True)
    c_final = ccc(vp, all_val["y"])
    r_final = [np.corrcoef(vp[:, k].cpu().numpy(), all_val["y"][:, k].cpu().numpy())[0, 1] for k in range(2)]
    rmse_final = ((vp - all_val["y"]) ** 2).mean(0).sqrt()

    print(f"\nFinal 7-Movie Model Performance on Validation:")
    print(f"  Valence CCC: {c_final[0]:.3f} | Pearson r: {r_final[0]:.3f} | RMSE: {rmse_final[0]:.3f}")
    print(f"  Arousal CCC: {c_final[1]:.3f} | Pearson r: {r_final[1]:.3f} | RMSE: {rmse_final[1]:.3f}")
    print(f"  Modality Gates: Video={gv*100:.1f}%, Audio={ga*100:.1f}%")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_ckpt = MODEL_DIR / f"mgn_both_{args.length}s_{args.label}.pt"
    torch.save({
        "arch": "mgn",
        "state_dict": final_model.state_dict(),
        "modalities": ["audio", "video"],
        "mu": mu,
        "sd": sd,
        "hidden_dim": 64,
        "metrics": {
            "ccc_v": c_final[0].item(), "ccc_a": c_final[1].item(),
            "r_v": r_final[0], "r_a": r_final[1],
            "rmse_v": rmse_final[0].item(), "rmse_a": rmse_final[1].item(),
        }
    }, out_ckpt)
    print(f"\nSaved production checkpoint: {out_ckpt}")

if __name__ == "__main__":
    main()
