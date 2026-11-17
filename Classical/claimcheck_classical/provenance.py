"""
ClaimCheck Classical Model - Provenance

Records what produced a saved model:
1. File checksums
2. Code and library versions
3. Run metadata saved next to the model

The metadata file also stores the selected thresholds,
so inference can use exactly the values chosen on validation data.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. FILE CHECKSUMS
# ============================================================

def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA256 checksum of a file."""

    digest = hashlib.sha256()

    # Read in chunks so large files do not fill memory.
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 2. CODE AND LIBRARY VERSIONS
# ============================================================

def git_info(directory) -> dict:
    """Return the current git commit and whether code has changed.

    Changes to result files (outputs/ and models/) do not count.
    """

    def run(arguments: list[str]) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(directory),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    try:
        commit = run(["rev-parse", "HEAD"])

        # Only code counts. Results that a run writes (outputs/, models/,
        # and the baseline's outputs/) are left out, as the baseline does,
        # so a run on a clean commit is recorded as clean.
        dirty = bool(run([
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ":/",
            ":(top,exclude)Classical/outputs",
            ":(top,exclude)Classical/models",
            ":(top,exclude)Baseline/outputs",
        ]))

    # Git may be missing, or the folder may not be a repository.
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "uncommitted_changes": None}

    return {"commit": commit, "uncommitted_changes": dirty}


def library_versions() -> dict:
    """Return the versions of the libraries that affect results."""

    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def hardware_info() -> dict:
    """Machine the run used (comparison table: cost and latency)."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "gpu": None,
    }


# ============================================================
# 3. RUN METADATA
# ============================================================

def build_metadata(
    model_path,
    input_paths: dict,
    project_dir,
    details: dict,
) -> dict:
    """Build the metadata saved next to a trained model.

    input_paths maps a short name to a data file. Each file is
    recorded with its SHA256 checksum.
    """

    model_path = Path(model_path)

    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "git": git_info(project_dir),
        "libraries": library_versions(),
        "input_sha256": {
            name: sha256_file(path)
            for name, path in input_paths.items()
        },
    }

    # Details such as thresholds and settings come from the caller.
    metadata.update(details)

    return metadata