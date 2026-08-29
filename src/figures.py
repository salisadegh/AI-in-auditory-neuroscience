"""Publication figures. Reads the saved analysis outputs and writes 300-dpi PNG
plus vector PDF for each figure. Palette is colour-vision-safe."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

from config import FIGURES, RESULTS

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9.5, "font.family": "DejaVu Sans",
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.18, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
    "figure.facecolor": "white", "savefig.bbox": "tight", "legend.frameon": False,
})
C = {"snr": "#0072B2", "lr": "#009E73", "gbdt": "#E69F00", "mlp": "#D55E00",
     "surrogate": "#CC79A7", "expert": "#111111", "baseline": "#8C8C8C"}
DETECTOR_COLOUR = {"wnsfmp": C["snr"], "RMS-SNR": C["snr"], "LR-eng": C["lr"],
                   "GBDT-eng": C["gbdt"], "MLP-raw": C["mlp"]}
LEVEL_BINS = [(20, 40, "20-39"), (40, 60, "40-59"), (60, 80, "60-79"), (80, 101, "80-100")]


def _tag(ax, letter, dx=-0.13, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)


def figure_label_validity(earndb, mendeley, arrays):
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    a = ax[0]
    for lo, hi, name in [(0, .2, "slight"), (.2, .4, "fair"), (.4, .6, "moderate"),
                         (.6, .8, "substantial"), (.8, 1, "almost perfect")]:
        a.add_patch(Rectangle((lo, -0.6), hi - lo, 3.2, color="#000000", alpha=0.035, lw=0))
        a.text((lo + hi) / 2, 2.62, name, ha="center", va="center",
               fontsize=7.2, color="#666666", style="italic")
    items = [("Surrogate vs expert\n(tone pip, 1 rater)", earndb["agreement"]["kappa"], None, C["surrogate"]),
             ("Surrogate vs consensus\n(click, 6 raters)", mendeley["agreement"]["kappa"], None, C["surrogate"]),
             ("Expert vs expert\n(click, 15 rater pairs)", mendeley["raters"]["inter_rater_kappa_median"],
              mendeley["raters"]["inter_rater_kappa_range"], C["expert"])]
    for i, (name, value, span, colour) in enumerate(items):
        a.barh(i, value, height=0.46, color=colour, alpha=0.92, zorder=3)
        if span:
            a.plot(span, [i, i], color=colour, lw=2.4, zorder=4, solid_capstyle="round")
        a.text(value + 0.022, i, f"{value:.2f}", va="center", fontsize=9.5,
               fontweight="bold", color=colour, zorder=5)
    a.set(yticks=range(3), xlim=(0, 1.0), ylim=(-0.6, 2.6),
          xlabel="Cohen's $\\kappa$ with the reference label")
    a.set_yticklabels([i[0] for i in items])
    a.invert_yaxis(); a.grid(axis="y", visible=False)
    a.set_title("The surrogate agrees far less than experts agree", loc="left")
    _tag(a, "A", dx=-0.52)

    b = ax[1]
    level, frequency = arrays["level"], arrays["frequency"]
    xs = np.arange(len(LEVEL_BINS))
    for freq, style, marker, alpha in [(1.0, "-", "o", 0.65), (4.0, "--", "s", 0.95)]:
        for values, colour in [(arrays["expert"], C["expert"]), (arrays["surrogate"], C["surrogate"])]:
            rate = [100 * values[(frequency == freq) & (level >= lo) & (level < hi)].mean()
                    if ((frequency == freq) & (level >= lo) & (level < hi)).any() else np.nan
                    for lo, hi, _ in LEVEL_BINS]
            b.plot(xs, rate, style, marker=marker, ms=5, color=colour, lw=1.9, alpha=alpha)
    b.set(xticks=xs, xlabel="stimulus level (dB)", ylabel="% labelled response-present", ylim=(-4, 104))
    b.set_xticklabels([n for _, _, n in LEVEL_BINS])
    b.set_title("Where they disagree: 1 kHz", loc="left")
    b.legend(handles=[Line2D([0], [0], color=C["expert"], lw=2, label="expert"),
                      Line2D([0], [0], color=C["surrogate"], lw=2, label="surrogate"),
                      Line2D([0], [0], color="#666", lw=1.6, ls="--", marker="s", ms=4, label="4 kHz"),
                      Line2D([0], [0], color="#666", lw=1.6, ls="-", marker="o", ms=4,
                             alpha=.65, label="1 kHz")], loc="upper left", ncol=2)
    _tag(b, "B")

    c = ax[2]
    rates = [100 * v for _, v in sorted(mendeley["raters"]["present_rate_by_rater"].items(),
                                        key=lambda kv: int(kv[0]))]
    c.scatter(range(1, len(rates) + 1), rates, s=58, color=C["expert"], zorder=4,
              label="individual annotators")
    c.axhline(np.mean(rates), color=C["expert"], lw=1.1, ls=":", zorder=2)
    c.axhline(100 * mendeley["raters"]["surrogate_present_rate"], color=C["surrogate"], lw=2.2,
              zorder=3, label="reproducibility surrogate")
    c.fill_between([0.4, len(rates) + 0.6], min(rates), max(rates), color=C["expert"], alpha=0.07, zorder=1)
    c.set(xlim=(0.4, len(rates) + 0.6), xticks=range(1, len(rates) + 1), xlabel="annotator",
          ylabel="% labelled response-present", ylim=(50, 90))
    c.set_title("Annotator-to-annotator variation", loc="left")
    c.annotate(f"{100 * (1 - mendeley['raters']['unanimous_fraction']):.0f}% of conditions\nnot unanimous",
               xy=(len(rates) / 2, max(rates) + 1), fontsize=8, color="#555555", ha="center")
    c.legend(loc="lower left")
    _tag(c, "C")
    _save(fig, "fig1_label_validity")


def figure_ranking(earndb, mendeley):
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))

    def slope(axis, values, title, ylim):
        for name, (left, right) in values.items():
            colour = DETECTOR_COLOUR[name]
            axis.plot([0, 1], [left, right], "-o", color=colour, lw=2.3, ms=7, zorder=3, clip_on=False)
            axis.text(-0.045, left, f"{name}  {left:.3f}", ha="right", va="center",
                      fontsize=8.8, color=colour, fontweight="bold")
            axis.text(1.045, right, f"{right:.3f}  {name}", ha="left", va="center",
                      fontsize=8.8, color=colour, fontweight="bold")
        axis.axhline(0.5, color="#999999", ls=":", lw=1)
        axis.set(xlim=(0, 1), xticks=[0, 1], ylabel="LOSO AUROC", ylim=ylim)
        axis.set_xticklabels(["surrogate\nlabel", "expert\nlabel"])
        axis.set_title(title, loc="left"); axis.grid(axis="x", visible=False)

    slope(ax[0], {k: (earndb["auc"]["surrogate"][k][0], earndb["auc"]["expert"][k][0])
                  for k in earndb["auc"]["expert"]},
          "Tone-pip corpus (1 annotator)", (0.55, 0.98))
    _tag(ax[0], "A", dx=-0.28)
    slope(ax[1], {k.replace(" (ref)", ""): (mendeley["auc"]["surrogate"][k][0],
                                            mendeley["auc"]["consensus"][k][0])
                  for k in mendeley["auc"]["consensus"]},
          "Click corpus (6-annotator consensus)", (0.45, 0.98))
    _tag(ax[1], "B", dx=-0.28)
    fig.suptitle("Changing only the label reorders the detectors", y=1.03,
                 fontsize=11, fontweight="bold")
    _save(fig, "fig2_ranking_inversion")


def figure_rater_uncertainty(mendeley, mde):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    a, names = ax[0], list(mendeley["per_rater_auc"])
    for i, name in enumerate(names):
        values = np.array(mendeley["per_rater_auc"][name])
        colour = DETECTOR_COLOUR[name.replace(" (ref)", "")]
        a.plot([values.min(), values.max()], [i, i], color=colour, lw=3, alpha=0.30,
               solid_capstyle="round", zorder=2)
        a.scatter(values, [i] * len(values), s=42, color=colour, zorder=4, alpha=0.95)
        a.scatter([np.median(values)], [i], s=120, marker="|", color=colour, zorder=5, linewidths=2.2)
        a.text(values.max() + 0.008, i, f"spread {values.max() - values.min():.3f}",
               va="center", fontsize=8.3, color=colour)
    a.set(yticks=range(len(names)), xlim=(0.53, 0.80),
          xlabel="LOSO AUROC, one point per annotator")
    a.set_yticklabels(names); a.invert_yaxis(); a.grid(axis="y", visible=False)
    a.set_title("Which annotator supplied the label moves AUROC", loc="left")
    _tag(a, "A", dx=-0.24)

    b = ax[1]
    consensus = [v[0] for v in mendeley["auc"]["consensus"].values()]
    bars = [("Between detectors\n(same labels)", max(consensus) - min(consensus), C["baseline"]),
            ("Between annotators\n(same detector)",
             float(np.mean(list(mendeley["per_rater_spread"].values()))), C["expert"]),
            ("Minimum detectable\ndifference", mde, "#B00020")]
    for i, (name, value, colour) in enumerate(bars):
        b.bar(i, value, width=0.55, color=colour, alpha=0.9, zorder=3)
        b.text(i, value + 0.004, f"{value:.3f}", ha="center", fontsize=9.5,
               fontweight="bold", color=colour)
    b.set(xticks=range(3), ylabel="$\\Delta$AUROC", ylim=(0, max(v for _, v, _ in bars) * 1.35))
    b.set_xticklabels([n for n, _, _ in bars], fontsize=8.6); b.grid(axis="x", visible=False)
    b.set_title("Label noise rivals the effect being measured", loc="left")
    _tag(b, "B", dx=-0.20)
    _save(fig, "fig3_rater_uncertainty")


def figure_baselines(earndb, se, n_listeners):
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.1))
    a = ax[0]
    order = ["stimulus level", "carrier frequency", "wnsfmp", "LR-eng", "GBDT-eng", "MLP-raw"]
    colours = [C["baseline"], C["baseline"], C["snr"], C["lr"], C["gbdt"], C["mlp"]]
    xs, width = np.arange(len(order)), 0.27
    for j, (key, alpha) in enumerate([("pooled", 1.0), ("within_1kHz", 0.62), ("within_4kHz", 0.34)]):
        values = [earndb["baselines"][n][key] for n in order]
        a.bar(xs + (j - 1) * width, values, width, color=colours, alpha=alpha, zorder=3,
              edgecolor="white", linewidth=0.6)
    a.axhline(0.5, color="#999999", ls=":", lw=1)
    reference = earndb["baselines"]["stimulus level"]["within_1kHz"]
    a.axhline(reference, color="#B00020", ls="--", lw=1.3, zorder=4)
    a.text(len(order) - 0.5, reference + 0.008, f"stimulus level alone, 1 kHz ({reference:.3f})",
           ha="right", fontsize=7.8, color="#B00020")
    a.set(xticks=xs, ylim=(0.42, 1.0), ylabel="AUROC vs expert label")
    a.set_xticklabels(order, rotation=22, ha="right", fontsize=8.5)
    a.set_title("No waveform detector beats stimulus level at 1 kHz", loc="left")
    a.legend(handles=[Rectangle((0, 0), 1, 1, color="#555", alpha=al, label=lb)
                      for al, lb in [(1.0, "pooled"), (0.62, "within 1 kHz"), (0.34, "within 4 kHz")]],
             loc="upper left", ncol=3)
    _tag(a, "A", dx=-0.15)

    b = ax[1]
    from stats import Z_80_POWER, listeners_required
    deltas = np.linspace(0.02, 0.20, 200)
    required = [listeners_required(se, d, n_listeners) for d in deltas]
    b.plot(deltas, required, color=C["snr"], lw=2.4)
    b.fill_between(deltas, 0, required, color=C["snr"], alpha=0.08)
    for delta in (0.10, 0.05, 0.03):
        n = listeners_required(se, delta, n_listeners)
        b.plot([delta, delta], [0, n], color="#999", ls=":", lw=1)
        b.plot(delta, n, "o", color="#B00020", ms=6, zorder=4)
        b.annotate(f"$\\Delta$={delta:.2f}\nN$\\approx${n:.0f}", xy=(delta, n), xytext=(6, 6),
                   textcoords="offset points", fontsize=8.2, color="#B00020")
    b.axhline(n_listeners, color=C["expert"], ls="--", lw=1.4)
    b.text(0.185, n_listeners + 2, f"this corpus (N = {n_listeners})", ha="right",
           fontsize=8.4, color=C["expert"])
    b.set(xlabel="true $\\Delta$AUROC to be detected", ylabel="listeners required (80% power)",
          xlim=(0.02, 0.20), ylim=(0, 240))
    b.set_title("What it would take to rank these detectors", loc="left")
    _tag(b, "B", dx=-0.16)
    _save(fig, "fig4_baselines_design")


def run():
    with open(RESULTS / "earndb.json") as fh:
        earndb = json.load(fh)
    with open(RESULTS / "mendeley.json") as fh:
        mendeley = json.load(fh)
    arrays = np.load(RESULTS / "earndb_predictions.npz")
    ses = [c["se"] for c in earndb["pairwise_expert"]]
    median_se = float(np.median(ses))
    from stats import Z_80_POWER
    figure_label_validity(earndb, mendeley, arrays)
    figure_ranking(earndb, mendeley)
    figure_rater_uncertainty(mendeley, Z_80_POWER * median_se)
    figure_baselines(earndb, median_se, earndb["n_listeners"])
    return sorted(p.name for p in FIGURES.glob("*.png"))


if __name__ == "__main__":
    for name in run():
        print("wrote", name)
