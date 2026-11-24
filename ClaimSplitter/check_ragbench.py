"""How often our split matches RAGBench's own sentence split (validation only).

    python ClaimSplitter/check_ragbench.py

RAGBench gives each answer as `response` and as `response_sentences`. Since all
our labels are per sentence, the splitter should reproduce that split. Reports
the share of answers split exactly the same way, and sentence-level precision /
recall (a sentence counts if the same text, ignoring spaces, appears in both).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from claimcheck_splitter.split import SPLITTER_VERSION, split  # noqa: E402

RAG = HERE.parent / "Data" / "RAGBench"
SUBSETS = ["FinQA", "TATQA"]
OUT = HERE / "outputs"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = {"splitter_version": SPLITTER_VERSION, "split": "validation", "subsets": {}}
    for s in SUBSETS:
        df = pd.read_parquet(RAG / s / "validation.parquet", columns=["id", "response", "response_sentences"])
        exact = tp = n_ours = n_gold = 0
        diffs = []
        for r in df.itertuples():
            gold = [norm(x[1]) for x in r.response_sentences]
            ours = [norm(c.text) for c in split(r.response)]
            exact += ours == gold
            overlap = sum((Counter(ours) & Counter(gold)).values())
            tp, n_ours, n_gold = tp + overlap, n_ours + len(ours), n_gold + len(gold)
            if ours != gold and len(diffs) < 30:
                diffs.append({"id": r.id, "ours": " || ".join(ours), "ragbench": " || ".join(gold)})
        m = {"answers": len(df), "exact_match_rate": exact / len(df),
             "sentence_precision": tp / max(n_ours, 1), "sentence_recall": tp / max(n_gold, 1)}
        report["subsets"][s] = m
        pd.DataFrame(diffs).to_csv(OUT / f"ragbench_{s.lower()}_validation_differences.csv", index=False)
        print(f"{s:<6} answers {m['answers']}: exact {m['exact_match_rate']:.3f}  "
              f"sentence P {m['sentence_precision']:.3f}  R {m['sentence_recall']:.3f}")
    (OUT / "ragbench_check_validation.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
