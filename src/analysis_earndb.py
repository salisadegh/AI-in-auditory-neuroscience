"""Primary analysis: label agreement, discrimination under each label, the
label x detector interaction, probability quality, and label-free baselines."""
import json

import numpy as np
from sklearn.metrics import brier_score_loss, cohen_kappa_score, roc_auc_score

import data_earndb as corpus
import stats
from config import RESULTS, SEED
from detectors import out_of_fold, platt_out_of_fold


def predictions(d, label):
    """Out-of-fold predictions with every detector trained under `label`."""
    y = d[label]
    common = dict(y=y, subject=d["subject"], X_eng=d["X_eng"], X_raw=d["X_raw"],
                  response_windows=d["response"])
    return {
        "wnsfmp": d["wnsfmp"],                       # untrained instrument statistic
        "LR-eng": out_of_fold("LR", **common),
        "GBDT-eng": out_of_fold("GBDT", **common),
        "MLP-raw": out_of_fold("MLP", **common),
    }


def run():
    d = corpus.load()
    subject, expert, surrogate = d["subject"], d["expert"], d["surrogate"]
    out = {"mapping": corpus.validate_label_mapping(),
           "n_samples": int(len(expert)), "n_conditions": int(len(expert) // 2),
           "n_listeners": int(len(np.unique(subject)))}

    # --- agreement between the surrogate and expert annotation ---
    out["agreement"] = {
        "overall": float((surrogate == expert).mean()),
        "kappa": float(cohen_kappa_score(expert, surrogate)),
        "expert_present_rate": float(expert.mean()),
        "surrogate_present_rate": float(surrogate.mean()),
        "surrogate_present_expert_absent": int(((surrogate == 1) & (expert == 0)).sum()),
        "surrogate_absent_expert_present": int(((surrogate == 0) & (expert == 1)).sum()),
        "by_frequency": {
            str(f): {"agreement": float((surrogate[m] == expert[m]).mean()),
                     "kappa": float(cohen_kappa_score(expert[m], surrogate[m])),
                     "expert_present": float(expert[m].mean()),
                     "surrogate_present": float(surrogate[m].mean())}
            for f in np.unique(d["frequency"]) for m in [d["frequency"] == f]},
    }

    # --- discrimination under each label ---
    pred = {"surrogate": predictions(d, "surrogate"), "expert": predictions(d, "expert")}
    out["auc"] = {
        label: {name: list(stats.auc_ci(score, d[label], subject))
                for name, score in pred[label].items()}
        for label in ("surrogate", "expert")}

    # --- pairwise comparisons under the expert annotation ---
    names = list(pred["expert"])
    comparisons = [(a, b, stats.paired_difference(pred["expert"][a], pred["expert"][b],
                                                  expert, subject)) for a, b in stats.pairs(names)]
    adjusted = stats.holm([c[2]["p"] for c in comparisons])
    out["pairwise_expert"] = [dict(a=a, b=b, holm_p=float(p), **r)
                              for (a, b, r), p in zip(comparisons, adjusted)]

    # --- primary analysis: label x detector interaction ---
    def bundle(name):
        return {"first": pred["surrogate"][name], "second": pred["expert"][name]}

    interactions = [(a, b, stats.interaction(bundle(a), bundle(b), surrogate, expert, subject))
                    for a, b in stats.pairs(names)]
    adjusted = stats.holm([c[2]["p"] for c in interactions])
    out["interaction"] = [dict(a=a, b=b, holm_p=float(p), **r)
                          for (a, b, r), p in zip(interactions, adjusted)]

    # --- probability quality against the expert annotation ---
    calibrated = dict(pred["expert"])
    calibrated["wnsfmp"] = platt_out_of_fold(d["wnsfmp"], expert, subject)
    out["calibration"] = {}
    for name, prob in calibrated.items():
        slope, intercept = stats.calibration_slope_intercept(prob, expert)
        out["calibration"][name] = dict(
            brier=float(brier_score_loss(expert, prob)), slope=slope, intercept=intercept,
            ece=[stats.expected_calibration_error(prob, expert, b, s)
                 for s in ("width", "mass") for b in (5, 10, 15, 20)])
    out["calibration"]["prevalence"] = dict(
        brier=float(brier_score_loss(expert, np.full(len(expert), expert.mean()))))

    # --- label-free baselines, pooled and stratified ---
    baselines = {"stimulus level": d["level"].astype(float),
                 "carrier frequency": (d["frequency"] == 4).astype(float)}
    out["baselines"] = {}
    for name, score in {**baselines, **pred["expert"]}.items():
        entry = {"pooled": float(roc_auc_score(expert, score))}
        for f in np.unique(d["frequency"]):
            m = d["frequency"] == f
            entry[f"within_{f:.0f}kHz"] = float(roc_auc_score(expert[m], score[m]))
        out["baselines"][name] = entry

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "earndb.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez(RESULTS / "earndb_predictions.npz", expert=expert, surrogate=surrogate,
             subject=subject, frequency=d["frequency"], level=d["level"], wnsfmp=d["wnsfmp"],
             **{f"{lab}__{name}": score for lab in pred for name, score in pred[lab].items()})
    return out


if __name__ == "__main__":
    result = run()
    print(f"listeners={result['n_listeners']} conditions={result['n_conditions']}")
    print(f"surrogate vs expert: agreement={result['agreement']['overall']:.3f} "
          f"kappa={result['agreement']['kappa']:.3f}")
    for label in ("surrogate", "expert"):
        row = "  ".join(f"{k} {v[0]:.3f}" for k, v in result["auc"][label].items())
        print(f"AUROC [{label}] {row}")
