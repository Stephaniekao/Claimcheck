"""Scores to labels, and checks that the scores mean what they say (Experiment Card Fields 7, 8 and 10).

No torch here, so the unit tests run without a GPU.

* score  : the model's probability that the passage supports the claim (0-1).
* labels : two thresholds chosen on validation turn the score into three labels:
           score >= upper -> Supported, score < lower -> Unsupported, else Uncertain.
* ECE    : Expected Calibration Error, 10 equal-width bins on the support score.
* temperature scaling: one number T fitted on validation only; it never changes
           the ranking of claims, so AUROC is the same before and after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
MAX_UNCERTAIN = 1 / 3      # AC2
MIN_PRECISION = 0.60       # Field 6
TEMPERATURES = np.round(np.arange(0.25, 4.0001, 0.05), 2)
EPS = 1e-6


def to_labels(score: np.ndarray, lower: float, upper: float) -> pd.Series:
    score = np.asarray(score, dtype=float)
    out = np.where(score >= upper, "Supported", np.where(score < lower, "Unsupported", "Uncertain"))
    return pd.Series(out)


def ece(gold: pd.Series, score: np.ndarray, n_bins: int = N_BINS) -> float:
    """Weighted gap between the mean support score and the share of Supported claims, per bin."""
    y = np.asarray(gold.eq("Supported"), dtype=float)
    p = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    if not len(p):
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def apply_temperature(score: np.ndarray, t: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(score) / t))


def fit_temperature(gold: pd.Series, score: np.ndarray) -> float:
    """The T in TEMPERATURES with the lowest log loss on validation."""
    y = np.asarray(gold.eq("Supported"), dtype=float)
    best_t, best_nll = 1.0, float("inf")
    for t in TEMPERATURES:
        q = np.clip(apply_temperature(score, t), EPS, 1 - EPS)
        nll = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
        if nll < best_nll - 1e-12:
            best_t, best_nll = float(t), nll
    return best_t


def _counts(gold_bad: np.ndarray, score: np.ndarray, lower: float, upper: float) -> dict:
    pred_bad = score < lower
    unc = (score >= lower) & (score < upper)
    tp = int((gold_bad & pred_bad).sum())
    n_pred, n_gold = int(pred_bad.sum()), int(gold_bad.sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "strict_recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "uncertain_rate": float(unc.mean())}


def choose_thresholds(gold: pd.Series, score: np.ndarray, n_candidates: int = 101) -> dict:
    """Field 8 rule, on validation only.

    Keep Uncertain <= 1/3 (AC2) and precision >= 0.60, then the highest strict
    recall; ties go to the higher F1, then the narrower Uncertain band. If no
    pair meets both conditions, keep Uncertain <= 1/3, take the highest F1, and
    say so in the output (like the baseline's RAGBench config).
    """
    score = np.asarray(score, dtype=float)
    gold_bad = np.asarray(gold.eq("Unsupported"))
    cands = np.unique(np.quantile(score, np.linspace(0, 1, n_candidates)))
    cands = np.unique(np.append(cands, cands[-1] + EPS))  # lets every claim fall below a cut
    rows = []
    for i, lo in enumerate(cands):
        for hi in cands[i:]:
            c = _counts(gold_bad, score, lo, hi)
            rows.append({"lower": float(lo), "upper": float(hi), "band": float(hi - lo), **c})
    grid = pd.DataFrame(rows)
    capped = grid[grid.uncertain_rate <= MAX_UNCERTAIN]
    ok = capped[capped.precision >= MIN_PRECISION]
    if len(ok):
        best = ok.sort_values(["strict_recall", "f1", "band"], ascending=[False, False, True]).iloc[0]
    else:
        best = capped.sort_values(["f1", "strict_recall", "band"], ascending=[False, False, True]).iloc[0]
    return {"lower": float(best.lower), "upper": float(best.upper),
            "precision_target_met_on_validation": bool(len(ok)),
            "rule": "Uncertain <= 1/3 and precision >= 0.60, then highest strict recall"
                    if len(ok) else "no pair met precision >= 0.60: Uncertain <= 1/3, then highest F1",
            "validation": {k: float(best[k]) for k in ("strict_recall", "precision", "f1", "uncertain_rate")}}
