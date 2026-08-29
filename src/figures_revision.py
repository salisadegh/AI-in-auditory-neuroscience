"""Regenerate the revision-stage figures (Figures 5-7) from the stored
out-of-fold predictions and the Mendeley analysis output."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGDIR = ROOT / "results", ROOT / "figures"
DET = ["wnsfmp", "LR-eng", "GBDT-eng", "MLP-raw"]
COL = {"wnsfmp": "#0072B2", "LR-eng": "#E69F00", "GBDT-eng": "#009E73", "MLP-raw": "#CC79A7"}


def auc(y, s, m=None):
    if m is not None:
        y, s = y[m], s[m]
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return (r[:len(pos)].sum() + len(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    D = np.load(RESULTS / "earndb_predictions.npz")
    yE, yS, fq = D["expert"].astype(float), D["surrogate"].astype(float), D["frequency"]
    lvl = D["level"].astype(float)
    men = json.load(open(RESULTS / "mendeley.json"))
    FIGDIR.mkdir(exist_ok=True)

    # Figure 5: label reordering, pooled vs within-frequency
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, mask) in zip(axes, [("Pooled", None), ("Within 4 kHz", fq == 4.0)]):
        for d in DET:
            s = auc(yS, D[f"surrogate__{d}"], mask)
            e = auc(yE, D[f"expert__{d}"], mask)
            ax.plot([0, 1], [s, e], "o-", color=COL[d], label=d)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["surrogate", "expert"])
        ax.set_title(title); ax.grid(alpha=.3)
    axes[0].set_ylabel("LOSO AUROC"); axes[0].legend(fontsize=8)
    fig.suptitle("Changing the label reorders the detectors; part of the reordering is pooling")
    fig.tight_layout(); fig.savefig(FIGDIR / "Figure5.png", dpi=300); plt.close(fig)

    # Figure 6: Mendeley per-annotator spread
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, d in enumerate(men["per_rater_auc"]):
        v = men["per_rater_auc"][d]
        ax.scatter([i] * len(v), v, color=COL.get(d, "k"), s=25)
        ax.plot([i - .2, i + .2], [np.median(v)] * 2, color="k")
    ax.set_xticks(range(len(men["per_rater_auc"])))
    ax.set_xticklabels(list(men["per_rater_auc"]), rotation=15)
    ax.set_ylabel("LOSO AUROC vs individual annotator")
    ax.set_title("Annotator identity moves performance as much as detector choice")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(FIGDIR / "Figure6.png", dpi=300); plt.close(fig)

    # Figure 7: label-free baselines vs detectors, pooled and stratified
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, (title, mask) in zip(axes, [("Pooled", None), ("1 kHz", fq == 1.0), ("4 kHz", fq == 4.0)]):
        names, vals = [], []
        base = lvl if mask is None else lvl
        names.append("level only"); vals.append(auc(yE, base, mask))
        if mask is None:
            names.append("frequency only"); vals.append(auc(yE, (fq == 4.0).astype(float)))
        for d in DET:
            names.append(d); vals.append(auc(yE, D[f"expert__{d}"], mask))
        ax.barh(range(len(vals)), vals,
                color=["#999"] * (len(vals) - 4) + [COL[d] for d in DET])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.axvline(.5, ls="--", c="k", lw=.8); ax.set_xlim(.4, 1); ax.set_title(title)
    axes[0].set_xlabel("AUROC vs expert label")
    fig.suptitle("Label-free baselines and detectors against the expert label")
    fig.tight_layout(); fig.savefig(FIGDIR / "Figure7.png", dpi=300); plt.close(fig)
    print("wrote Figures 5-7")


if __name__ == "__main__":
    main()
