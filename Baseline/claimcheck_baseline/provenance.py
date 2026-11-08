"""Where a result came from: code commit and dataset version (Week 8 reproducibility).

Every results JSON records this block, so each number can be traced to the
exact code and data that produced it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
IES_MANIFEST = PROJECT / "Data" / "Inserted Error Set" / "reproducibility_manifest_400.json"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit() -> dict:
    """HEAD commit, and whether tracked code had uncommitted changes (outputs excluded)."""
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)Baseline/outputs")
    return {"commit": _git("rev-parse", "HEAD"), "uncommitted_changes": bool(status) if status is not None else None}


def inserted_error_set_version() -> dict:
    manifest = json.loads(IES_MANIFEST.read_text())
    labels = manifest.get("total_label_counts")
    return {"format": manifest.get("format"),
            "version": "v2" if labels and labels.get("Supported") == labels.get("Unsupported") else "v1",
            "manifest_sha256": hashlib.sha256(IES_MANIFEST.read_bytes()).hexdigest()}


def _cpu_name() -> str | None:
    if platform.system() == "Darwin":
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return name or None
    return platform.processor() or None


def provenance(include_inserted: bool = True) -> dict:
    out = {"git": git_commit(), "python": platform.python_version(),
           "hardware": {"platform": platform.platform(), "machine": platform.machine(), "cpu": _cpu_name(), "gpu": None}}
    if include_inserted:
        out["inserted_error_set"] = inserted_error_set_version()
    return out
