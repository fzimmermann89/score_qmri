"""Keep optional preparation dependencies separate from the training suite."""

import importlib.util

# Raising importorskip while loading an initial conftest aborts pytest startup.
# Omit only this optional test directory when the preparation extras are absent.
collect_ignore_glob = ["test_*.py"] if any(importlib.util.find_spec(name) is None for name in ("nibabel", "nilearn", "pydicom")) else []
