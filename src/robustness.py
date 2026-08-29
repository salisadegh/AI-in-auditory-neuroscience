"""Guards against small-sample inference.

Because both sub-averages of a condition share a label, and because a bootstrap
over few listener clusters can produce implausibly small p-values, the
interaction is re-tested at condition level, under an exact subject-level
sign-flip permutation, and with each listener omitted in turn. Seed stability
of the stochastic detectors is checked separately.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import data_earndb as corpus
import stats
from analysis_earndb import predictions
from config import RESULTS
from detectors import out_of_fold

GUARDED = [("wnsfmp", "MLP-raw"), ("GBDT-eng", "MLP-raw"), ("LR-eng", "MLP-raw")]


def run(seeds=range(10)):
    d = corpus.load()
    subject, expert, surrogate = d["subject"], d["expert"], d["surrogate"]
    pred = {"surrogate": predictions(d, "surrogate"), "expert": predictions(d, "expert")}
    bundle = lambda n: {"first": pred["surrogate"][n], "second": pred["expert"][n]}
    out = {}

    # seed stability of the stochastic detectors
    out["seed_stability"] = {}
    for kind, name in (("MLP", "MLP-raw"), ("GBDT", "GBDT-eng")):
        scores = [float(roc_auc_score(expert, out_of_fold(
            kind, y=expert, subject=subject, X_eng=d["X_eng"], X_raw=d["X_raw"],
            response_windows=d["response"], seed=s))) for s in seeds]
        out["seed_stability"][name] = dict(median=float(np.median(scores)),
                                           lo=float(min(scores)), hi=float(max(scores)),
                                           sd=float(np.std(scores)))

    # condition level: one sub-average per condition removes the paired duplicate
    keep = np.arange(0, len(expert), 2)
    out["condition_level"] = {}
    for a, b in GUARDED:
        sub = lambda n: {"first": pred["surrogate"][n][keep], "second": pred["expert"][n][keep]}
        out["condition_level"][f"{a} vs {b}"] = stats.interaction(
            sub(a), sub(b), surrogate[keep], expert[keep], subject[keep])

    # exact subject-level sign-flip permutation
    out["exact_permutation"] = {
        f"{a} vs {b}": stats.exact_sign_flip(bundle(a), bundle(b), surrogate, expert, subject)
        for a, b in GUARDED}

    # leave-one-listener-out influence
    out["influence"] = {}
    for a, b in GUARDED:
        def statistic(keep_subjects, a=a, b=b):
            m = np.isin(subject, keep_subjects)
            da = (roc_auc_score(expert[m], pred["expert"][a][m])
                  - roc_auc_score(surrogate[m], pred["surrogate"][a][m]))
            db = (roc_auc_score(expert[m], pred["expert"][b][m])
                  - roc_auc_score(surrogate[m], pred["surrogate"][b][m]))
            return da - db
        values = stats.leave_one_subject_influence(statistic, subject)
        out["influence"][f"{a} vs {b}"] = dict(
            lo=float(min(values.values())), hi=float(max(values.values())),
            sign_stable=bool(max(values.values()) < 0 or min(values.values()) > 0))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "robustness.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    result = run()
    for key, value in result["condition_level"].items():
        perm = result["exact_permutation"][key]
        print(f"{key}: condition-level p={value['p']:.4f}  "
              f"exact permutation p={perm['p']:.4f} (floor {perm['floor']:.4f})")
