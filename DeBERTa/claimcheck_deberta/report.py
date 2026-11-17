"""Metrics for one seed, and mean +/- spread across seeds (Experiment Card Fields 5-8).

Uses the baseline's metrics module for every label-based number, so DeBERTa and
the baseline are scored by the same code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Baseline"))
from claimcheck_baseline import metrics  # noqa: E402

from . import calibration as cal

AGGREGATIONS = ["lowest", "mean"]  # claim scores -> answer score (Field 6), chosen on validation


def answer_scores(claims: pd.DataFrame, score: np.ndarray, how: str) -> pd.DataFrame:
    df = pd.DataFrame({"answer": claims.answer.to_numpy(), "score": score, "label": claims.answer_label.to_numpy()})
    agg = "min" if how == "lowest" else "mean"
    return df.groupby("answer").agg(score=("score", agg), label=("label", "first")).reset_index()


def label_metrics(gold: pd.Series, score: np.ndarray, setting: dict) -> dict:
    """Binary metrics at the chosen thresholds, plus score metrics and ECE."""
    s = cal.apply_temperature(score, setting["temperature"])
    pred = cal.to_labels(s, setting["lower"], setting["upper"])
    gold = gold.reset_index(drop=True)
    return {**metrics.binary_metrics(gold, pred),
            "score": metrics.score_metrics(gold, pd.Series(s)),
            "ece_raw": cal.ece(gold, score), "ece": cal.ece(gold, s)}


def choose_setting(gold: pd.Series, score: np.ndarray) -> dict:
    """Temperature, then the two thresholds on the calibrated score (validation only)."""
    t = cal.fit_temperature(gold.reset_index(drop=True), score)
    th = cal.choose_thresholds(gold.reset_index(drop=True), cal.apply_temperature(score, t))
    return {"temperature": t, **th}


def across_seeds(per_seed: dict, path: list[str]) -> dict:
    """Mean and standard deviation of one number across seeds, e.g. path=["overall", "f1"]."""
    vals = []
    for m in per_seed.values():
        for k in path:
            m = m[k]
        vals.append(float(m))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}
