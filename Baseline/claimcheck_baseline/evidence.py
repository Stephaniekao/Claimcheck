"""Cited passages for the Inserted Error Set.

The Inserted Error Set stores the cited evidence as structured XBRL facts,
not as text. This module renders them into a short plain-text passage that
every method (baseline, TF-IDF+LR, DeBERTa, LLM judge) can read.

Leakage control: the passage depends only on the company and the cited
filing, never on the claim or its label. Every claim about the same filing
gets the *same* passage, so the passage format cannot reveal the answer.

* General Claims: all FY facts that the cited 10-K reports (current and prior
  years) for the 17 line items used in the set, read from the committed SEC
  companyfacts snapshot.
* Storyboard B: every segment revenue line and the total revenue line for
  the cited filing, from the segment CSV (values were extracted from the
  filing's inline XBRL by the dataset notebook).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

EVIDENCE_VERSION = "1.0.0"

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "Data" / "Inserted Error Set"
SNAPSHOT_ZIP = DATA / "source_snapshots.zip"

GENERAL_TAGS = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "CostOfRevenue", "CostOfGoodsAndServicesSold",
    "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss", "Assets", "AssetsCurrent", "Liabilities",
    "LiabilitiesCurrent", "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
    "NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def _companyfacts(cik: str) -> dict:
    name = f"source_snapshots/companyfacts_{cik}.json"
    with zipfile.ZipFile(SNAPSHOT_ZIP) as z:
        with z.open(name) as f:
            return json.load(f)


def _fmt_millions(value: float) -> str:
    m = value / 1e6
    text = f"{m:,.3f}".rstrip("0").rstrip(".")
    return f"USD {text} million"


def _is_annual(fact: dict) -> bool:
    if "start" not in fact:
        return True  # instant (balance sheet) fact
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return 330 <= days <= 380


def general_passage(cik: str, accession: str, company: str) -> str:
    facts = _companyfacts(cik)["facts"]["us-gaap"]
    lines = []
    for tag in GENERAL_TAGS:
        if tag not in facts:
            continue
        label = facts[tag]["label"]
        seen: dict[str, float] = {}
        for x in facts[tag]["units"].get("USD", []):
            if x.get("accn") != accession or not _is_annual(x):
                continue
            seen.setdefault(x["end"], x["val"])
        for end in sorted(seen, reverse=True):
            fact_kind = "as of" if not any("start" in x for x in facts[tag]["units"]["USD"] if x.get("accn") == accession) else "for the fiscal year ended"
            lines.append(f"{label} {fact_kind} {end}: {_fmt_millions(seen[end])}.")
    header = f"{company} Form 10-K (accession {accession}), selected XBRL financial data:"
    return "\n".join([header, *lines])


def storyboard_passage(group: pd.DataFrame) -> str:
    r = group.iloc[0]
    lines = [f"{r.sec_name} Form 10-K (accession {r.accession}), revenue by segment for the fiscal year ended {r.report_end}:"]
    for seg, val in group.drop_duplicates("segment_name")[["segment_name", "segment_revenue_usd"]].itertuples(index=False):
        lines.append(f"{seg} segment revenue for the fiscal year ended {r.report_end}: USD {val:,.0f}.")
    lines.append(f"Total revenues for the fiscal year ended {r.report_end}: USD {r.total_revenue_usd:,.0f}.")
    return "\n".join(lines)


# Columns a model may read. Everything else (label, error_type, reference
# values, claimed values, actual_value_usd ...) is evaluation-only.
MODEL_INPUT_COLUMNS = ["claim_id", "subset", "split", "claim", "evidence"]
EVAL_COLUMNS = ["claim_id", "label", "error_type"]


def load_inserted_error_set() -> pd.DataFrame:
    """One row per claim with model inputs (claim, evidence) and evaluation columns."""
    g = pd.read_csv(DATA / "general_claims_300.csv", dtype={"cik": str})
    g["evidence"] = [general_passage(r.cik, r.accession, r.sec_name) for r in g.itertuples()]
    g["subset"] = "general"

    s = pd.read_csv(DATA / "storyboard_b_100.csv", dtype={"cik": str})
    passages = {k: storyboard_passage(grp) for k, grp in s.groupby(["accession", "report_end"])}
    s["evidence"] = [passages[(r.accession, r.report_end)] for r in s.itertuples()]
    s["subset"] = "storyboard_b"

    cols = list(dict.fromkeys(MODEL_INPUT_COLUMNS + EVAL_COLUMNS + ["ticker"]))
    return pd.concat([g[cols], s[cols]], ignore_index=True)


def input_hashes() -> dict[str, str]:
    files = ["general_claims_300.csv", "storyboard_b_100.csv", "source_snapshots.zip"]
    return {f: sha256_file(DATA / f) for f in files}
