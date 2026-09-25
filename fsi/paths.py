"""Where FSI reads and writes on disk.

REPO is this checkout. ROOT is where data, raw rollouts, results, logs and checkpoints live; it defaults to the
checkout and can be moved with the FSI_ROOT environment variable. The expected layout under ROOT is
raw/, data/, results/, logs/, checkpoints/, videos/ and scratch/; symlink any of them to a larger disk if needed.
See docs/CONFIGURATION.md.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FSI_ROOT", REPO))
