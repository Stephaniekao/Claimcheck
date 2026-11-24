"""Claims and cited passages for DeBERTa, from the same files the baseline reads.

* RAGBench FinQA / TAT-QA: claims = `response_sentences` (the split every method
  uses), passage = the retrieved `documents_sentences`. Gold: a sentence is
  Unsupported iff its key is in `unsupported_response_sentence_keys`.
* Inserted Error Set v2: claim + passage from the baseline's evidence module,
  so all four methods read exactly the same passages (Field 5).

Model input is the claim and the passage only. `sentence_support_information`
is read for the TRAIN split only, to pick which window of a long passage a
Supported training claim is paired with. It is never read for validation or
test, and never given to the model as text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RAG = PROJECT / "Data" / "RAGBench"
SUBSETS = ["FinQA", "TATQA"]

sys.path.insert(0, str(PROJECT / "Baseline"))
from claimcheck_baseline import evidence  # noqa: E402


def _support_keys(info) -> dict[str, set[str]]:
    out = {}
    for item in info if info is not None else []:
        key = item.get("response_sentence_key") if isinstance(item, dict) else None
        keys = item.get("supporting_sentence_keys") if isinstance(item, dict) else None
        if key is not None:
            out[str(key)] = {str(k) for k in (keys if keys is not None else [])}
    return out


def load_ragbench(subset: str, split: str) -> tuple[pd.DataFrame, list[dict]]:
    """One row per claim, plus one passage per answer (claims point to it by `answer`)."""
    cols = ["id", "documents_sentences", "response_sentences", "unsupported_response_sentence_keys", "adherence_score"]
    if split == "train":
        cols.append("sentence_support_information")
    df = pd.read_parquet(RAG / subset / f"{split}.parquet", columns=cols)
    claims, passages = [], []
    for r in df.itertuples():
        sents = [(str(s[0]), str(s[1])) for doc in r.documents_sentences for s in doc]
        passages.append({"keys": [k for k, _ in sents], "sentences": [t for _, t in sents]})
        bad = set(r.unsupported_response_sentence_keys)
        support = _support_keys(r.sentence_support_information) if split == "train" else {}
        for key, sent in r.response_sentences:
            claims.append({"answer": len(passages) - 1, "id": r.id, "key": key, "claim": sent,
                           "label": "Unsupported" if key in bad else "Supported",
                           "answer_label": "Supported" if bool(r.adherence_score) else "Unsupported",
                           "support_keys": sorted(support.get(key, set()))})
    return pd.DataFrame(claims), passages


def load_inserted_error_set() -> tuple[pd.DataFrame, list[dict]]:
    """Same rows as the baseline. Each claim gets its own passage entry (passages are short)."""
    data = evidence.load_inserted_error_set()
    passages = [{"keys": [str(i) for i in range(len(p.split("\n")))], "sentences": p.split("\n")} for p in data.evidence]
    data = data.drop(columns=["evidence"]).copy()
    data["answer"] = range(len(data))
    return data, passages


def input_hashes() -> dict[str, str]:
    """SHA256 of every input file a DeBERTa result depends on."""
    out = {f"Inserted Error Set/{k}": v for k, v in evidence.input_hashes().items()}
    for s in SUBSETS:
        for split in ("train", "validation", "test"):
            path = RAG / s / f"{split}.parquet"
            if path.exists():
                out[f"RAGBench/{s}/{split}.parquet"] = evidence.sha256_file(path)
    return out
