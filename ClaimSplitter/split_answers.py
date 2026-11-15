"""Split every answer in a CSV into claims (one row per sentence).

    python ClaimSplitter/split_answers.py ANSWERS.csv --id-col ID --text-col TEXT --out CLAIMS.csv

Output columns: answer_id, claim_index, claim, start, end, splitter_version.
`start`/`end` are character offsets into the original answer, so each claim can
be traced back to its sentence. Used for FinanceBench before hand labelling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claimcheck_splitter.split import SPLITTER_VERSION, split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.answers, dtype=str, keep_default_na=False)
    for col in (args.id_col, args.text_col):
        if col not in df:
            ap.error(f"CSV column not found: {col}")
    rows = [{"answer_id": r[args.id_col], "claim_index": c.index, "claim": c.text, "start": c.start, "end": c.end,
             "splitter_version": SPLITTER_VERSION}
            for _, r in df.iterrows() for c in split(r[args.text_col])]
    out = pd.DataFrame(rows, columns=["answer_id", "claim_index", "claim", "start", "end", "splitter_version"])
    out.to_csv(args.out, index=False)
    print(f"{len(df)} answers -> {len(out)} claims ({len(out) / max(len(df), 1):.1f} per answer). Wrote {args.out}")


if __name__ == "__main__":
    main()
