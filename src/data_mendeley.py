"""Loader for the click-evoked corpus exported by a commercial evoked-potential
system as CSV.

Each file holds several traces; every trace stores an average together with two
independent buffers, which serve as the two sub-averages. Column blocks repeat
in groups of six, and a header block gives the stimulus onset sample.
Files are read as latin-1 because the exporter writes a non-UTF-8 micro sign.
"""
import csv
import json
import os
import re

import numpy as np
from scipy.signal import detrend

from config import (LABELS, MENDELEY_DIR, MENDELEY_FS, MENDELEY_NOISE_MS,
                    MENDELEY_RESPONSE_MS, REPRODUCIBILITY_CUT)
from features import bandpass, engineered_features, pearson, window, znorm

AVERAGE_COL, BUFFER1_COL, BUFFER2_COL, BLOCK = 2, 4, 6, 6


def _read_trace_block(path):
    with open(path, newline="", encoding="latin-1") as fh:
        rows = list(csv.reader(fh))
    header, start = {}, None
    for i, row in enumerate(rows):
        try:
            float(row[0])
            start = i
            break
        except (ValueError, IndexError):
            if row and row[0].endswith(":"):
                header[row[0]] = row[1] if len(row) > 1 else ""
    data = rows[start:]
    width = len(data[0])

    def column_mean(offset):
        cols = list(range(offset, width, BLOCK))
        matrix = np.array([[float(r[c]) if c < len(r) and r[c].strip() else np.nan
                            for c in cols] for r in data])
        return np.nan_to_num(np.nanmean(matrix, axis=1))

    onset = int(float(header.get("Zero Position:", "530")))
    return column_mean(AVERAGE_COL), column_mean(BUFFER1_COL), column_mean(BUFFER2_COL), onset


def _condition(path):
    """Grand-average the traces, trim the zero-padded margin, detrend and filter."""
    average, buf1, buf2, onset = _read_trace_block(path)
    valid = np.where(np.abs(average) > 1e-9)[0]
    lo, hi = valid.min(), valid.max() + 1
    prepare = lambda v: bandpass(detrend(v[lo:hi]), MENDELEY_FS)
    return prepare(buf1), prepare(buf2), onset - lo


def load_rater_labels():
    with open(LABELS / "mendeley_rater_labels.json") as fh:
        raw = json.load(fh)
    return {int(rater): {tuple(int(p) for p in k.split("|")): int(v) for k, v in d.items()}
            for rater, d in raw.items()}


def load_consensus_labels():
    with open(LABELS / "mendeley_consensus_labels.json") as fh:
        raw = json.load(fh)
    return {tuple(int(p) for p in k.split("|")): int(v) for k, v in raw.items()}


def load():
    consensus, raters = load_consensus_labels(), load_rater_labels()
    rows = []
    for participant in sorted(os.listdir(MENDELEY_DIR)):
        folder = MENDELEY_DIR / participant
        if not folder.is_dir():
            continue
        subject = int(re.sub(r"\D", "", participant))
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".txt") or "tmp" in name.lower():
                continue
            m = re.search(r"(\d+)db", name, re.I)
            if not m:
                continue
            level = int(m.group(1))
            if (subject, level) not in consensus:
                continue
            buf1, buf2, onset = _condition(str(folder / name))
            reproducibility = pearson(window(buf1, MENDELEY_FS, MENDELEY_RESPONSE_MS, onset),
                                      window(buf2, MENDELEY_FS, MENDELEY_RESPONSE_MS, onset))
            for waveform in (buf1, buf2):
                rows.append(dict(
                    subject=subject, level=level,
                    consensus=consensus[(subject, level)],
                    surrogate=int(reproducibility > REPRODUCIBILITY_CUT),
                    rater={r: raters[r][(subject, level)] for r in raters},
                    eng=engineered_features(waveform, MENDELEY_FS,
                                            MENDELEY_RESPONSE_MS, MENDELEY_NOISE_MS, onset),
                    raw=znorm(np.resize(window(waveform, MENDELEY_FS, (0.0, 20.0), onset), 220)),
                ))
    if not rows:
        raise FileNotFoundError(
            f"No recordings found under {MENDELEY_DIR}. See data/README.md.")
    return dict(
        subject=np.array([r["subject"] for r in rows]),
        level=np.array([r["level"] for r in rows]),
        consensus=np.array([r["consensus"] for r in rows]),
        surrogate=np.array([r["surrogate"] for r in rows]),
        rater={r: np.array([row["rater"][r] for row in rows]) for r in raters},
        X_eng=np.vstack([r["eng"] for r in rows]),
        X_raw=np.vstack([r["raw"] for r in rows]),
    )
