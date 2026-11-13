"""
EXPERIMENT 2c - How well does each single modality predict perceived emotion?

Compares audio predictions and video predictions with the ground truth
(both labelling methods: Average and Temporal Gaussian Weighting).

Metrics (per dimension):
  Pearson r   - do prediction and ground truth go up/down together (scale-free)
  CCC         - concordance correlation coefficient (agreement incl. scale and offset)
  RMSE, MAE   - absolute error on the [-1, 1] scale
  Sign acc    - % of clips where the sign (positive/negative) is right

Run:  python scripts/exp2_compare.py
Out:  outputs/exp2/metrics_<L>s.csv, scatter_<L>s_<method>.png, timeline_<L>s_<method>.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLIP_LENGTHS, CLIPS_DIR, OUT_DIR

OUT = OUT_DIR / "exp2"
METHODS = {"mean": "Average", "gauss": "Gaussian"}
DIMS = ["valence", "arousal"]


def metrics(pred, gt):
    r = np.corrcoef(pred, gt)[0, 1]
    ccc = 2 * np.cov(pred, gt, bias=True)[0, 1] / (pred.var() + gt.var() + (pred.mean() - gt.mean()) ** 2)
    return {"pearson_r": r, "ccc": ccc,
            "rmse": np.sqrt(np.mean((pred - gt) ** 2)), "mae": np.mean(np.abs(pred - gt)),
            "sign_acc_%": 100 * np.mean(np.sign(pred) == np.sign(gt))}


for L in CLIP_LENGTHS:
    gt = pd.read_csv(CLIPS_DIR / "BMI" / f"clips_{L}s" / "labels.csv")
    preds = {}
    for modality in ["audio", "video"]:
        f = OUT / f"{modality}_pred_{L}s.csv"
        if f.exists():
            preds[modality] = gt.merge(pd.read_csv(f), on="clip")
        else:
            print(f"(skipping {modality}: {f.name} not found)")
    if not preds:
        continue

    rows = []
    for m, m_name in METHODS.items():
        fig_s, ax_s = plt.subplots(len(preds), 2, figsize=(10, 4.5 * len(preds)), squeeze=False)
        fig_t, ax_t = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        for i, (modality, df) in enumerate(preds.items()):
            for j, dim in enumerate(DIMS):
                p, g = df[dim].values, df[f"{dim}_{m}"].values
                res = metrics(p, g)
                rows.append({"modality": modality, "gt_method": m_name, "dimension": dim, **res})

                ax = ax_s[i, j]
                ax.scatter(g, p, s=10, alpha=0.6)
                ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
                ax.set_xlabel(f"ground truth {dim}"); ax.set_ylabel(f"{modality} prediction")
                ax.set_title(f"{modality} - {dim}: r={res['pearson_r']:.2f}, CCC={res['ccc']:.2f}")

                ax_t[j].plot(df["start_s"], p, label=f"{modality} prediction", alpha=0.8)
        for j, dim in enumerate(DIMS):
            ax_t[j].plot(gt["start_s"], gt[f"{dim}_{m}"], "k", lw=2, label="ground truth")
            ax_t[j].set_ylabel(dim); ax_t[j].set_ylim(-1, 1); ax_t[j].legend(loc="upper left")
        ax_t[1].set_xlabel("time in segment (s)")

        fig_s.suptitle(f"A Beautiful Mind, {L} s clips, ground truth = {m_name}")
        fig_t.suptitle(f"A Beautiful Mind, {L} s clips, ground truth = {m_name}")
        fig_s.tight_layout(); fig_t.tight_layout()
        fig_s.savefig(OUT / f"scatter_{L}s_{m}.png", dpi=130)
        fig_t.savefig(OUT / f"timeline_{L}s_{m}.png", dpi=130)
        plt.close("all")

    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT / f"metrics_{L}s.csv", index=False)
    print(f"\n=== {L} s clips ===")
    print(table.to_string(index=False))

print(f"\nAll results saved in {OUT}")
