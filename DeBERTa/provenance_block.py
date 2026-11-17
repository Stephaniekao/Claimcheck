"""The baseline's provenance block, plus the GPU and DeBERTa's own outputs folder.

Uses Baseline/claimcheck_baseline/provenance.py, so both methods log the same
fields. Two differences: the GPU name is filled in, and uncommitted changes
ignore DeBERTa/outputs as well as Baseline/outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Baseline"))
from claimcheck_baseline import provenance as base  # noqa: E402


def provenance(include_inserted: bool = True) -> dict:
    out = base.provenance(include_inserted=include_inserted)
    status = base._git("status", "--porcelain", "--untracked-files=no", "--", ".",
                       ":(exclude)Baseline/outputs", ":(exclude)DeBERTa/outputs")
    out["git"]["uncommitted_changes"] = bool(status) if status is not None else None
    try:
        from claimcheck_deberta.model import gpu_name
        out["hardware"]["gpu"] = gpu_name()
    except ImportError:
        pass
    return out
