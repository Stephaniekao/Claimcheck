"""Download RAGBench FinQA / TAT-QA without the `datasets` package.

Reproduces cells 1-2 of Data/data_download.ipynb (parquet + 12-column CSV in
Data/RAGBench/<subset>/) from Hugging Face's parquet export, then checks the
CSV SHA256 against Data/Inserted Error Set/reproducibility_manifest_400.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "Data" / "RAGBench"
MANIFEST = PROJECT / "Data" / "Inserted Error Set" / "reproducibility_manifest_400.json"
URL = "https://huggingface.co/api/datasets/galileo-ai/ragbench/parquet/{subset}/{split}/0.parquet"
SUBSETS = {"finqa": "FinQA", "tatqa": "TATQA"}
SPLITS = ["train", "validation", "test"]
KEEP_COLUMNS = ["id", "question", "documents", "response", "documents_sentences", "response_sentences",
                "sentence_support_information", "unsupported_response_sentence_keys",
                "overall_supported_explanation", "relevance_explanation", "dataset", "split"]


def to_python(v):
    if isinstance(v, np.ndarray):
        return to_python(v.tolist())
    if isinstance(v, (list, tuple)):
        return [to_python(x) for x in v]
    if isinstance(v, dict):
        return {str(k): to_python(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return v


def serialize(v):
    if isinstance(v, (np.ndarray, list, tuple, dict)):
        return json.dumps(to_python(v), ensure_ascii=False)
    return v


def main() -> None:
    expected = json.loads(MANIFEST.read_text())["input_sha256"]
    for subset, folder_name in SUBSETS.items():
        folder = BASE / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            pq = folder / f"{split}.parquet"
            if not pq.exists():
                # curl uses the system certificate store (python.org builds on macOS often have none).
                subprocess.run(["curl", "-sSfL", "-o", str(pq), URL.format(subset=subset, split=split)], check=True)
            df = pd.read_parquet(pq)
            df["dataset"], df["split"] = subset, split
            df.to_parquet(pq, index=False)
            out = folder / f"{split}.csv"
            clean = df[KEEP_COLUMNS].copy()
            for c in clean.columns:
                clean[c] = clean[c].apply(serialize)
            clean.to_csv(out, index=False, encoding="utf-8-sig")
            digest = hashlib.sha256(out.read_bytes()).hexdigest()
            key = f"Data/RAGBench/{folder_name}/{split}.csv"
            status = "matches manifest" if expected.get(key) == digest else "DIFFERS from manifest"
            print(f"{key}: {len(df)} rows, sha256 {digest[:12]}... {status}")


if __name__ == "__main__":
    main()
