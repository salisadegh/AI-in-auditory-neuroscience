"""Ancillary sensitivity analyses that require the raw tone-pip corpus:
the surrogate-threshold sweep and the matched-filter feature sensitivity.
Download the corpus as described in data/README.md, then run this script.
Both analyses are deterministic (seed 42)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_earndb as corpus
from detectors import out_of_fold


def threshold_sweep(cuts=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40)):
    """Surrogate present-rate and surrogate-vs-expert kappa as the
    reproducibility cut varies. Requires the raw records because the
    stored predictions carry only the binary surrogate at c = 0.25."""
    d = corpus.load()
    # recover the per-condition reproducibility correlation from the loader
    import glob, os, re, wfdb
    from config import EARNDB_DIR, EARNDB_FS, EARNDB_RESPONSE_MS
    from features import pearson, window
    expert = corpus.load_expert_labels()
    rows = []
    for header in sorted(glob.glob(str(EARNDB_DIR / "*_R1.hea"))):
        stem = os.path.basename(header)[:-4]
        m = corpus.RECORD_RE.match(stem)
        if not m:
            continue
        key = (int(m.group(1)), float(m.group(3)), int(m.group(2)))
        if key not in expert:
            continue
        r1 = wfdb.rdrecord(str(EARNDB_DIR / stem))
        r2 = wfdb.rdrecord(str(EARNDB_DIR / (stem[:-1] + "2")))
        x1 = r1.p_signal[:, r1.sig_name.index("ABR")].astype(float)
        x2 = r2.p_signal[:, r2.sig_name.index("ABR")].astype(float)
        rows.append((expert[key], pearson(window(x1, EARNDB_FS, EARNDB_RESPONSE_MS),
                                          window(x2, EARNDB_FS, EARNDB_RESPONSE_MS))))
    y = np.array([r[0] for r in rows])
    corr = np.array([r[1] for r in rows])
    print("cut  present-rate  kappa(vs expert)")
    for c in cuts:
        s = (corr > c).astype(int)
        print(f"{c:.2f}  {s.mean():12.3f}  {cohen_kappa_score(y, s):.3f}")


def matched_filter_sensitivity():
    """LR-eng with and without the per-fold matched-filter feature, against
    the expert annotation (reported in the Robustness section)."""
    d = corpus.load()
    y, subject = d["expert"], d["subject"]
    with_mf = out_of_fold("LR", y=y, subject=subject, X_eng=d["X_eng"],
                          X_raw=d["X_raw"], response_windows=d["response"])
    without = out_of_fold("LR", y=y, subject=subject, X_eng=d["X_eng"],
                          X_raw=d["X_raw"], response_windows=None)
    print(f"LR-eng with matched filter    AUROC {roc_auc_score(y, with_mf):.3f}")
    print(f"LR-eng without matched filter AUROC {roc_auc_score(y, without):.3f}")


if __name__ == "__main__":
    threshold_sweep()
    print()
    matched_filter_sensitivity()
