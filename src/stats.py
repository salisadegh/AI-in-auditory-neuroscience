"""Subject-clustered inference: bootstrap intervals, paired and
difference-in-differences tests, exact permutation, multiplicity control,
calibration measures, and the design analysis.

All resampling is at the listener level: a listener's records move together,
because records from one listener are not independent.
"""
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import BOOTSTRAP_CI, BOOTSTRAP_TEST, SEED

Z_80_POWER = 1.959964 + 0.841621  # two-sided 5% + 80% power


def _resample(rng, subject, subjects):
    picked = rng.choice(subjects, len(subjects), replace=True)
    return np.concatenate([np.where(subject == s)[0] for s in picked])


def auc_ci(score, y, subject, n_boot=BOOTSTRAP_CI, seed=SEED):
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject)
    draws = []
    for _ in range(n_boot):
        idx = _resample(rng, subject, subjects)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], score[idx]))
    return (float(roc_auc_score(y, score)),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def paired_difference(score_a, score_b, y, subject, n_boot=BOOTSTRAP_TEST, seed=SEED):
    """Paired subject-clustered bootstrap of an AUROC difference."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject)
    draws = []
    for _ in range(n_boot):
        idx = _resample(rng, subject, subjects)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
    draws = np.array(draws)
    p = max(2 * min((draws <= 0).mean(), (draws >= 0).mean()), 1.0 / len(draws))
    return dict(estimate=float(draws.mean()), lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)), se=float(draws.std()),
                mde=float(Z_80_POWER * draws.std()), p=float(p))


def interaction(pred_a, pred_b, y_first, y_second, subject,
                n_boot=BOOTSTRAP_TEST, seed=SEED):
    """Difference-in-differences test of the label x detector interaction.

    pred_a and pred_b are dicts keyed by label name, each holding the
    out-of-fold predictions of that detector *trained under that label*.
    Any component of the label change common to both detectors cancels.
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject)
    draws = []
    for _ in range(n_boot):
        idx = _resample(rng, subject, subjects)
        if len(np.unique(y_first[idx])) < 2 or len(np.unique(y_second[idx])) < 2:
            continue
        da = (roc_auc_score(y_second[idx], pred_a["second"][idx])
              - roc_auc_score(y_first[idx], pred_a["first"][idx]))
        db = (roc_auc_score(y_second[idx], pred_b["second"][idx])
              - roc_auc_score(y_first[idx], pred_b["first"][idx]))
        draws.append(da - db)
    draws = np.array(draws)
    p = max(2 * min((draws <= 0).mean(), (draws >= 0).mean()), 1.0 / len(draws))
    return dict(estimate=float(draws.mean()), lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)), p=float(p))


def exact_sign_flip(pred_a, pred_b, y_first, y_second, subject):
    """Exact subject-level sign-flip permutation of the interaction.

    Computes each listener's own interaction contribution and enumerates all
    2^n sign assignments. With few listeners the attainable p-value has a hard
    floor of 1/2^n, which is reported alongside the result.
    """
    per_subject = []
    for s in np.unique(subject):
        m = subject == s
        if len(np.unique(y_first[m])) < 2 or len(np.unique(y_second[m])) < 2:
            continue
        da = (roc_auc_score(y_second[m], pred_a["second"][m])
              - roc_auc_score(y_first[m], pred_a["first"][m]))
        db = (roc_auc_score(y_second[m], pred_b["second"][m])
              - roc_auc_score(y_first[m], pred_b["first"][m]))
        per_subject.append(da - db)
    values = np.array(per_subject)
    n = len(values)
    if n == 0:
        return dict(n=0, estimate=float("nan"), p=float("nan"), floor=float("nan"))
    observed = values.mean()
    extreme = 0
    for bits in range(2 ** n):
        signs = np.array([1 if (bits >> i) & 1 else -1 for i in range(n)])
        if abs((signs * values).mean()) >= abs(observed) - 1e-12:
            extreme += 1
    return dict(n=int(n), estimate=float(observed),
                p=float(extreme / 2 ** n), floor=float(1.0 / 2 ** n))


def leave_one_subject_influence(fn, subject):
    """Recompute a statistic with each listener omitted in turn."""
    subjects = np.unique(subject)
    return {int(s): float(fn(np.setdiff1d(subjects, [s]))) for s in subjects}


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment; returns adjusted values in input order."""
    order = np.argsort(pvalues)
    k = len(pvalues)
    adjusted = np.empty(k)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (k - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def expected_calibration_error(prob, y, bins=10, scheme="width"):
    if scheme == "width":
        edges = np.linspace(0, 1, bins + 1)
    else:
        edges = np.percentile(prob, np.linspace(0, 100, bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    error = 0.0
    for i in range(bins):
        upper = prob <= edges[i + 1] if i == bins - 1 else prob < edges[i + 1]
        m = (prob >= edges[i]) & upper
        if m.sum():
            error += abs(y[m].mean() - prob[m].mean()) * m.sum() / len(prob)
    return float(error)


def calibration_slope_intercept(prob, y):
    logit = np.log(np.clip(prob, 1e-4, 1 - 1e-4) / (1 - np.clip(prob, 1e-4, 1 - 1e-4)))
    model = LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def listeners_required(se, target_delta, n_current):
    """Listeners needed for a target difference, assuming 1/sqrt(N) scaling."""
    return float(n_current * ((Z_80_POWER * se) / target_delta) ** 2)


def pairs(names):
    return list(itertools.combinations(names, 2))
