"""Replication on the click-evoked corpus, which carries six independent
annotations and therefore supports a direct estimate of label variance."""
import itertools
import json

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import data_mendeley as corpus
import stats
from config import RESULTS
from detectors import out_of_fold, platt_out_of_fold


def predictions(d, y):
    common = dict(y=y, subject=d["subject"], X_eng=d["X_eng"], X_raw=d["X_raw"])
    return {
        "RMS-SNR": platt_out_of_fold(d["X_eng"][:, 0], y, d["subject"]),
        "LR-eng": out_of_fold("LR", **common),
        "GBDT-eng": out_of_fold("GBDT", **common),
        "MLP-raw": out_of_fold("MLP", **common),
    }


def run():
    d = corpus.load()
    subject, consensus, surrogate = d["subject"], d["consensus"], d["surrogate"]
    raters = d["rater"]
    out = {"n_samples": int(len(consensus)), "n_conditions": int(len(consensus) // 2),
           "n_listeners": int(len(np.unique(subject))), "n_raters": len(raters)}

    stacked = np.vstack([raters[r] for r in sorted(raters)])
    unanimous = (stacked.sum(0) == 0) | (stacked.sum(0) == len(raters))
    kappas = [cohen_kappa_score(stacked[i], stacked[j])
              for i, j in itertools.combinations(range(len(raters)), 2)]
    out["raters"] = {
        "consensus_present_rate": float(consensus.mean()),
        "surrogate_present_rate": float(surrogate.mean()),
        "unanimous_fraction": float(unanimous.mean()),
        "inter_rater_kappa_median": float(np.median(kappas)),
        "inter_rater_kappa_range": [float(min(kappas)), float(max(kappas))],
        "present_rate_by_rater": {str(r): float(raters[r].mean()) for r in sorted(raters)},
    }
    out["agreement"] = {"overall": float((surrogate == consensus).mean()),
                        "kappa": float(cohen_kappa_score(consensus, surrogate))}

    out["auc"] = {}
    for label, y in (("surrogate", surrogate), ("consensus", consensus)):
        out["auc"][label] = {name: list(stats.auc_ci(score, y, subject))
                             for name, score in predictions(d, y).items()}

    # label variance: the same detector scored against each annotator in turn
    out["per_rater_auc"] = {}
    for r in sorted(raters):
        for name, score in predictions(d, raters[r]).items():
            out["per_rater_auc"].setdefault(name, []).append(
                float(roc_auc_score(raters[r], score)))
    out["per_rater_spread"] = {k: float(max(v) - min(v)) for k, v in out["per_rater_auc"].items()}

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "mendeley.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    result = run()
    print(f"inter-rater kappa median={result['raters']['inter_rater_kappa_median']:.3f}")
    print(f"surrogate vs consensus kappa={result['agreement']['kappa']:.3f}")
    for name, spread in result["per_rater_spread"].items():
        print(f"  {name}: AUROC spread across raters {spread:.3f}")
