"""Loader for the averaged-waveform tone-pip corpus.

Records are WFDB pairs named N<subject>_evoked_ave<level>_F<frequency>_R<rep>.
Each stimulus condition is stored as two independent sub-averages (R1, R2).
Header comments carry the stimulus level, carrier frequency and the
instrument's weighted signal-to-noise statistic.
"""
import glob
import json
import os
import re

import numpy as np
import wfdb

from config import (EARNDB_ANALYSIS_MS, EARNDB_DIR, EARNDB_FS, EARNDB_NOISE_MS,
                    EARNDB_RESPONSE_MS, LABELS, REPRODUCIBILITY_CUT)
from features import bandpass, engineered_features, pearson, raw_vector, window, znorm

RECORD_RE = re.compile(r"N(\d+)_evoked_ave(\d+)_F(\d+)_R1$")


def _header_value(comment, pattern):
    m = re.search(pattern, comment)
    return float(m.group(1)) if m else float("nan")


def load_expert_labels():
    """Expert annotations keyed by (subject, carrier frequency in kHz, level in dB)."""
    with open(LABELS / "earndb_expert_labels.json") as fh:
        raw = json.load(fh)
    out = {}
    for key, value in raw.items():
        subject, frequency, level = key.split("|")
        out[(int(subject), float(frequency), int(level))] = int(value)
    return out


def load(expert_only=True):
    """Return a record dictionary of arrays.

    Both sub-averages of a condition enter as separate single-run samples
    carrying that condition's label, so every detector sees one averaged run.
    """
    expert = load_expert_labels()
    rows = []
    for header in sorted(glob.glob(str(EARNDB_DIR / "*_R1.hea"))):
        stem = os.path.basename(header)[:-4]
        m = RECORD_RE.match(stem)
        if not m:
            continue
        subject, level, frequency = int(m.group(1)), int(m.group(2)), float(m.group(3))
        first = str(EARNDB_DIR / stem)
        second = first[:-1] + "2"
        if not (os.path.exists(second + ".hea") and os.path.exists(second + ".dat")):
            continue
        key = (subject, frequency, level)
        if expert_only and key not in expert:
            continue
        try:
            r1, r2 = wfdb.rdrecord(first), wfdb.rdrecord(second)
        except Exception:
            continue
        x1 = r1.p_signal[:, r1.sig_name.index("ABR")].astype(float)
        x2 = r2.p_signal[:, r2.sig_name.index("ABR")].astype(float)

        reproducibility = pearson(window(x1, EARNDB_FS, EARNDB_RESPONSE_MS),
                                  window(x2, EARNDB_FS, EARNDB_RESPONSE_MS))
        surrogate = int(reproducibility > REPRODUCIBILITY_CUT)

        for waveform, record in ((x1, r1), (x2, r2)):
            filtered = bandpass(waveform, EARNDB_FS)
            rows.append(dict(
                subject=subject, frequency=frequency, level=level,
                expert=expert.get(key, -1), surrogate=surrogate,
                reproducibility=reproducibility,
                wnsfmp=_header_value(record.comments[0], r"<wnsfmp>:\s*([\-0-9.eE+]+)"),
                eng=engineered_features(filtered, EARNDB_FS, EARNDB_RESPONSE_MS, EARNDB_NOISE_MS),
                raw=raw_vector(waveform, EARNDB_FS, EARNDB_ANALYSIS_MS),
                response=znorm(window(filtered, EARNDB_FS, EARNDB_RESPONSE_MS)),
            ))
    if not rows:
        raise FileNotFoundError(
            f"No records found under {EARNDB_DIR}. See data/README.md for how to obtain the corpus.")
    return dict(
        subject=np.array([r["subject"] for r in rows]),
        frequency=np.array([r["frequency"] for r in rows]),
        level=np.array([r["level"] for r in rows]),
        expert=np.array([r["expert"] for r in rows]),
        surrogate=np.array([r["surrogate"] for r in rows]),
        wnsfmp=np.array([r["wnsfmp"] for r in rows]),
        X_eng=np.vstack([r["eng"] for r in rows]),
        X_raw=np.vstack([r["raw"] for r in rows]),
        response=np.vstack([r["response"] for r in rows]),
    )


def validate_label_mapping():
    """Check the annotation-to-record alignment.

    The published annotation table heads its level column with a sensation-level
    unit while the corpus records levels in peak-equivalent SPL. A systematic
    offset would still produce a superficially complete match, so we test
    alternative alignments and the populated-cell pattern directly.
    """
    expert = load_expert_labels()
    records = set()
    for header in sorted(glob.glob(str(EARNDB_DIR / "*_R1.hea"))):
        stem = os.path.basename(header)[:-4]
        m = RECORD_RE.match(stem)
        if m and os.path.exists(str(EARNDB_DIR / (stem[:-1] + "2.hea"))):
            records.add((int(m.group(1)), float(m.group(3)), int(m.group(2))))
    identity = sum(1 for k in expert if k in records)
    offsets = {d: sum(1 for k in expert if (k[0], k[1], k[2] + d) in records)
               for d in (-10, -5, 5, 10)}
    concordant = total = 0
    for subject in sorted({k[0] for k in records}):
        for frequency in sorted({k[1] for k in records}):
            for level in range(25, 101, 5):
                total += 1
                concordant += int(((subject, frequency, level) in records)
                                  == ((subject, frequency, level) in expert))
    return dict(annotations=len(expert), identity_matches=identity,
                offset_matches=offsets, unmatched_annotations=len(set(expert) - records),
                cell_concordance=concordant / total)
