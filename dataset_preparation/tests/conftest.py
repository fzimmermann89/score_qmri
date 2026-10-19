"""Keep optional preparation dependencies separate from the training suite."""

import pytest

pytest.importorskip("nibabel", reason="Install dataset_preparation/requirements.txt to run preparation tests")
pytest.importorskip("nilearn", reason="Install dataset_preparation/requirements.txt to run preparation tests")
pytest.importorskip("pydicom", reason="Install dataset_preparation/requirements.txt to run preparation tests")
