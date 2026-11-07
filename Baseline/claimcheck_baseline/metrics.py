"""Metrics from Experiment Card Fields 6, 7 and 10.

Positive class = Unsupported (a wrong claim we want to catch).
* strict recall  : wrong claims labeled Unsupported / all wrong claims
* lenient recall : wrong claims labeled Unsupported or Uncertain / all wrong claims
* precision      : wrong claims among those labeled Unsupported
* F1             : strict precision and strict recall
* false-alarm rate: correct claims labeled Unsupported / all correct claims
* uncertain rate : share of all claims labeled Uncertain (AC2: at most 1/3)
"""
from __future__ import annotations

import pandas as pd

TARGETS = {"strict_recall": 0.80, "precision": 0.60, "storyboard_b_strict_recall": 0.80, "max_uncertain_rate": 1 / 3}


def _safe(a: float, b: float) -> float:
    return a / b if b else float("nan")


def binary_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    gold_bad = gold.eq("Unsupported")
    pred_bad = pred.eq("Unsupported")
    flagged = pred.isin(["Unsupported", "Uncertain"])
    tp = int((gold_bad & pred_bad).sum())
    p = _safe(tp, int(pred_bad.sum()))
    r = _safe(tp, int(gold_bad.sum()))
    f1 = _safe(2 * p * r, p + r) if p == p and r == r else float("nan")
    return {
        "n": int(len(gold)),
        "n_unsupported_gold": int(gold_bad.sum()),
        "strict_recall": r,
        "lenient_recall": _safe(int((gold_bad & flagged).sum()), int(gold_bad.sum())),
        "precision": p,
        "lenient_precision": _safe(int((gold_bad & flagged).sum()), int(flagged.sum())),
        "f1": f1,
        "false_alarm_rate": _safe(int((~gold_bad & pred_bad).sum()), int((~gold_bad).sum())),
        "uncertain_rate": float(pred.eq("Uncertain").mean()) if len(pred) else float("nan"),
    }


def per_error_type(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Share of each predicted label within each gold error type."""
    tab = pd.crosstab(df["error_type"], df[pred_col], normalize="index").reindex(
        columns=["Unsupported", "Uncertain", "Supported"], fill_value=0.0)
    tab.insert(0, "n", df.groupby("error_type").size())
    return tab


def confusion(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    return pd.crosstab(df["label"], df[pred_col]).reindex(columns=["Unsupported", "Uncertain", "Supported"], fill_value=0)


def meets_targets(overall: dict, storyboard: dict | None) -> dict:
    out = {
        "strict_recall>=0.80": overall["strict_recall"] >= TARGETS["strict_recall"],
        "precision>=0.60": overall["precision"] >= TARGETS["precision"],
        "uncertain_rate<=1/3": overall["uncertain_rate"] <= TARGETS["max_uncertain_rate"],
    }
    if storyboard is not None:
        out["storyboard_b_strict_recall>=0.80"] = storyboard["strict_recall"] >= TARGETS["storyboard_b_strict_recall"]
    return out


# --- Score-based metrics (Field 7), for the support score. -------------------
# The positive class is still Unsupported, so a claim's risk = 1 - support score.

def auroc(gold: pd.Series, risk: pd.Series) -> float:
    """Probability a wrong claim gets a higher risk than a correct one (ties count half)."""
    pos = gold.eq("Unsupported").to_numpy()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(risk.to_numpy()).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_points(gold: pd.Series, risk: pd.Series) -> pd.DataFrame:
    """Precision and recall when flagging every claim with risk >= t, for each distinct t."""
    df = pd.DataFrame({"pos": gold.eq("Unsupported").to_numpy(), "risk": risk.to_numpy()})
    g = df.groupby("risk", sort=True)["pos"].agg(["sum", "count"]).iloc[::-1]
    tp, flagged = g["sum"].cumsum(), g["count"].cumsum()
    return pd.DataFrame({"threshold": g.index, "precision": tp / flagged,
                         "recall": tp / max(int(df.pos.sum()), 1)}).reset_index(drop=True)


def average_precision(gold: pd.Series, risk: pd.Series) -> float:
    pts = _pr_points(gold, risk)
    prev_r = pts.recall.shift(fill_value=0.0)
    return float(((pts.recall - prev_r) * pts.precision).sum())


def precision_at_recall(gold: pd.Series, risk: pd.Series, target: float = 0.80) -> dict:
    """Best precision among thresholds whose recall reaches the target (Field 7)."""
    pts = _pr_points(gold, risk)
    ok = pts[pts.recall >= target]
    if ok.empty:
        return {"precision": float("nan"), "recall": float("nan"), "risk_threshold": float("nan")}
    best = ok.loc[ok.precision.idxmax()]
    return {"precision": float(best.precision), "recall": float(best.recall), "risk_threshold": float(best.threshold)}


def score_metrics(gold: pd.Series, score: pd.Series) -> dict:
    risk = 1 - score
    return {"n": int(len(gold)), "auroc": auroc(gold, risk), "average_precision": average_precision(gold, risk),
            "base_rate": float(gold.eq("Unsupported").mean()), "n_distinct_scores": int(score.nunique()),
            "precision_at_recall_0.80": precision_at_recall(gold, risk, 0.80)}
