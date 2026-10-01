"""Locations shared by every script in the repository.

REPO_ROOT is the folder that holds this file. DATA_ROOT holds the data layers
(parquet_data/, inertial_data/, std_data/, regime_data/, subset_data/ and
canonical_snapshot.json). It defaults to a folder named "dataset" next to the
repository and is overridden by the environment variable THESIS_DATA_ROOT.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("THESIS_DATA_ROOT",
                                REPO_ROOT.parent / "dataset")).expanduser()
