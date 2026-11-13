"""
Dedicated 3-Second Model Training Script
Trains Decoupled Cross-Modal Gated Attention Network (MGN) or baseline FusionHead
on the 3s cached clip features.

Outputs: outputs/model/mgn_both_3s_gauss.pt
"""
import argparse
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')

from common import FEAT_DIR, MODEL_DIR, get_genre_vector
from models import FusionHead, PerceivedVA_Model, ccc, ccc_loss

def main():
    p = argparse.ArgumentParser(description="Train Perceived Emotion Model on 3s clips")
    p.add_argument("--arch", choices=["mgn", "mlp"], default="mgn", help="Model architecture")
    p.add_argument("--length", type=int, default=3, help="Clip length in seconds (default: 3)")
    p.add_argument("--modality", choices=["audio", "video", "both"], default="both")
    p.add_argument("--label", choices=["mean", "gauss"], default="gauss", help="Ground truth type")
    p.add_argument("--val_movie", default=None, help="Hold out a specific movie for validation, e.g. BMI")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    args = p.parse_args()

    torch.manual_seed(0)

    if args.arch == "mgn" and args.modality != "both":
        sys.exit("MGN architecture requires --modality both (audio + video).")

    MODS = ["audio", "video"] if args.modality == "both" else [args.modality]

    files = sorted(FEAT_DIR.glob(f"*_{args.length}s.npz"))
    if not files:
        sys.exit(f"No features found in {FEAT_DIR} for {args.length}s clips. Run extract_features_3s.py first.")

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
            cut = int(0.85 * n)
            parts = [(train, slice(0, cut)), (val, slice(cut, n))]

        for split, idx in parts:
            for m in MODS:
                split[m].append(d[m][idx])
            split["y"].append(d[f"labels_{args.label}"][idx])
            split["genre"].append(g_arr[idx])

    if not val["y"] or not train["y"]:
        sys.exit(f"Train or validation set is empty.")

    to_t = lambda split: {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in split.items()}
    train, val = to_t(train), to_t(val)
    print(f"Arch: {args.arch.upper()} | 3s Clips - Train: {len(train['y'])}, Val: {len(val['y'])}, Modalities: {MODS}", flush=True)

    mu = {m: train[m].mean(0) for m in MODS}
    sd = {m: train[m].std(0) + 1e-6 for m in MODS}
    xs_train = {m: (train[m] - mu[m]) / sd[m] for m in MODS}
    xs_val = {m: (val[m] - mu[m]) / sd[m] for m in MODS}

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
            print(f"epoch {epoch:4d}  train loss {loss.item():.4f}  val CCC {val_ccc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        if args.arch == "mgn":
            pred, _, (gv, ga) = model(video=xs_val["video"], audio=xs_val["audio"], genre=val["genre"], return_gates=True)
            print(f"\nAverage Modality Attention Gates on 3s Validation: Video={gv*100:.1f}%, Audio={ga*100:.1f}%", flush=True)
        else:
            pred = model([xs_val[m] for m in MODS])

    c = ccc(pred, val["y"])
    rmse = ((pred - val["y"]) ** 2).mean(0).sqrt()
    r = [np.corrcoef(pred[:, k].cpu().numpy(), val["y"][:, k].cpu().numpy())[0, 1] for k in range(2)]

    print(f"\nBest 3s Validation Results ({args.arch.upper()}, {args.modality}, {args.length}s, {args.label} labels):", flush=True)
    for k, dim in enumerate(["valence", "arousal"]):
        print(f"  {dim:8s}  CCC: {c[k]:.3f}   Pearson r: {r[k]:.3f}   RMSE: {rmse[k]:.3f}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"{args.arch}_{args.modality}_{args.length}s_{args.label}.pt"
    save_dict = {
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "modalities": MODS,
        "mu": mu,
        "sd": sd,
        "hidden_dim": args.hidden_dim,
        "metrics": {
            "ccc_v": c[0].item(), "ccc_a": c[1].item(),
            "r_v": r[0], "r_a": r[1],
            "rmse_v": rmse[0].item(), "rmse_a": rmse[1].item()
        }
    }
    torch.save(save_dict, out)
    print(f"\nSaved 3s production checkpoint: {out}", flush=True)

if __name__ == "__main__":
    main()
