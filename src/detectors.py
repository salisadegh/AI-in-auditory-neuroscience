"""Detectors spanning the capacity range, evaluated leave-one-subject-out.

Hyperparameters are fixed a priori and are not tuned on these data.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import SEED
from features import znorm

DETECTORS = ["wnsfmp", "LR-eng", "GBDT-eng", "MLP-raw"]


def _estimator(kind, seed=SEED):
    if kind == "LR":
        return LogisticRegression(C=1.0, max_iter=2000)
    if kind == "GBDT":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06, random_state=seed)
    if kind == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                             max_iter=800, random_state=seed)
    raise ValueError(kind)


def _matched_filter(train_idx, y, response_windows):
    """Template built from training-fold response-present waveforms."""
    present = train_idx[y[train_idx] == 1]
    if len(present) == 0:
        return np.zeros(response_windows.shape[1])
    return znorm(response_windows[present].mean(axis=0))


def _matched_scores(idx, template, response_windows):
    norm = template / (np.linalg.norm(template) + 1e-12)
    return np.array([
        float(np.dot(response_windows[i], norm) / (np.linalg.norm(response_windows[i]) + 1e-12))
        for i in idx
    ])


def out_of_fold(kind, y, subject, X_eng=None, X_raw=None, response_windows=None, seed=SEED):
    """Subject-disjoint out-of-fold probabilities.

    Standardisation and the matched-filter template are fitted inside training
    folds only, so no test listener influences any fitted quantity.
    """
    n = len(y)
    out = np.full(n, np.nan)
    X = X_eng if kind in ("LR", "GBDT") else X_raw
    for train, test in LeaveOneGroupOut().split(X, y, groups=subject):
        if len(np.unique(y[train])) < 2:
            out[test] = float(y[train].mean())
            continue
        if kind == "LR" and response_windows is not None:
            template = _matched_filter(train, y, response_windows)
            tr = np.column_stack([X_eng[train], _matched_scores(train, template, response_windows)])
            te = np.column_stack([X_eng[test], _matched_scores(test, template, response_windows)])
        else:
            tr, te = X[train], X[test]
        scaler = StandardScaler().fit(tr)
        model = _estimator(kind, seed).fit(scaler.transform(tr), y[train])
        out[test] = model.predict_proba(scaler.transform(te))[:, 1]
    return out


def platt_out_of_fold(score, y, subject):
    """Subject-disjoint Platt map, used for probability quality only.

    Discrimination is always scored on the raw statistic: per-fold maps are not
    globally monotone and would alter the pooled area under the curve.
    """
    out = np.full(len(y), np.nan)
    for s in np.unique(subject):
        test = subject == s
        if len(np.unique(y[~test])) < 2:
            out[test] = float(y[~test].mean())
            continue
        model = LogisticRegression(max_iter=1000).fit(score[~test].reshape(-1, 1), y[~test])
        out[test] = model.predict_proba(score[test].reshape(-1, 1))[:, 1]
    return out
