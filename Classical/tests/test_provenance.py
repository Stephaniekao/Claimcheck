"""Unit tests for run provenance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from claimcheck_classical.provenance import (
    build_metadata,
    git_info,
    library_versions,
    sha256_file,
)


# ============================================================
# 1. PROVENANCE TESTS
# ============================================================

class TestProvenance(unittest.TestCase):

    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.txt"
            path.write_bytes(b"ClaimCheck")

            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(b"ClaimCheck").hexdigest(),
            )

    def test_git_info_outside_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            info = git_info(folder)

        self.assertEqual(info["commit"], "unknown")
        self.assertIsNone(info["uncommitted_changes"])

    def test_library_versions(self):
        versions = library_versions()

        for name in ("python", "numpy", "pandas", "scipy", "scikit-learn"):
            self.assertIn(name, versions)

    def test_build_metadata_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.joblib"
            data = Path(folder) / "train.csv"

            model.write_bytes(b"model")
            data.write_bytes(b"a,b\n1,2\n")

            metadata = build_metadata(
                model_path=model,
                input_paths={"train": data},
                project_dir=folder,
                details={"thresholds": {"feasible": False}},
            )

        # Must not raise.
        text = json.dumps(metadata, allow_nan=False)

        self.assertIn("model_sha256", metadata)
        self.assertEqual(metadata["model_file"], "model.joblib")
        self.assertIn("train", metadata["input_sha256"])
        self.assertEqual(metadata["thresholds"], {"feasible": False})
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
