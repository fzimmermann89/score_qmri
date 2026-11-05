"""FastAPI server for data viewer."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from score_qmri.datasets.kind import DatasetKind

from .config_loader import load_config
from .mock_manager import MockSampleManager
from .sampling import DEFAULT_N, SampleManager, SampleResult

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Score-qMRI Data Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Shape", "X-Labels", "X-Kinds", "X-Has-Masks"],
)

# Global state
_manager: SampleManager | MockSampleManager | None = None
_is_mock_mode: bool = False
_config_path: str = "data.yaml"
_config_content: str | None = None  # Track raw config content for smart reload
_startup_complete: bool = False
_warmup_in_progress: bool = False


def _load_manager(config_path: str) -> tuple[SampleManager | MockSampleManager, bool, str]:
    """Load sample manager, falling back to mock if needed.

    Returns
    -------
    tuple
        (manager, is_mock_mode, config_content)
    """
    config_file = Path(config_path)
    config_content = ""

    try:
        config_content = config_file.read_text() if config_file.exists() else ""
        config = load_config(config_path)
        logger.info(f"Loaded {len(config['dataset_configs'])} datasets from {config_path}")
        if config.get("skipped_paths"):
            logger.warning(f"Skipped missing files: {config['skipped_paths'][:3]}...")
        return SampleManager(config, prewarm=False), False, config_content
    except FileNotFoundError as e:
        logger.warning(f"No valid datasets found, using mock mode: {e}")
        return MockSampleManager(), True, config_content
    except Exception as e:
        logger.exception(f"Config load failed, using mock mode: {e}")
        return MockSampleManager(), True, config_content


def _diff_raw_configs(old_configs: list[dict], new_configs: list[dict]) -> set[int]:
    """Find indices of configs that changed between old and new.

    Parameters
    ----------
    old_configs
        List of raw config dicts from previous load.
    new_configs
        List of raw config dicts from new load.

    Returns
    -------
    set[int]
        Set of indices that changed (modified, added, or removed).
    """
    changed = set()

    # Check for modified configs (compare by index)
    for i in range(min(len(old_configs), len(new_configs))):
        if old_configs[i] != new_configs[i]:
            changed.add(i)

    # Added configs
    if len(new_configs) > len(old_configs):
        changed.update(range(len(old_configs), len(new_configs)))

    # Removed configs (need to clear all caches for safety)
    if len(new_configs) < len(old_configs):
        changed.update(range(len(new_configs), len(old_configs)))

    return changed


@app.on_event("startup")
def startup() -> None:
    """Initialize manager on startup."""
    global _manager, _is_mock_mode, _config_content, _startup_complete
    _manager, _is_mock_mode, _config_content = _load_manager(_config_path)
    _startup_complete = True

    if isinstance(_manager, SampleManager):
        _start_prewarm(_manager)


def _start_prewarm(manager: SampleManager, indices: set[int] | None = None) -> None:
    """Prewarm datasets and dataloaders in a background thread."""
    global _warmup_in_progress

    if _warmup_in_progress:
        return

    _warmup_in_progress = True

    def _run() -> None:
        global _warmup_in_progress
        try:
            if indices:
                manager.prewarm_datasets(indices=sorted(indices), warm_dataloaders=True)
            else:
                manager.prewarm_datasets(warm_dataloaders=True)
        finally:
            _warmup_in_progress = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend."""
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd viewer && npm run build")
    return FileResponse(index_path)


@app.post("/api/reload")
def reload_config() -> dict[str, Any]:
    """Re-read data.yaml and reinitialize datasets.

    Uses smart diffing to only reload changed datasets.
    """
    global _manager, _is_mock_mode, _config_content

    config_file = Path(_config_path)
    new_content = config_file.read_text() if config_file.exists() else ""

    # Check if content actually changed
    if new_content == _config_content:
        return {
            "status": "unchanged",
            "mock_mode": _is_mock_mode,
            "num_datasets": len(_manager.list_datasets()) if _manager else 0,
            "changed_datasets": [],
            "message": "Config file unchanged",
        }

    # Content changed - load new config
    try:
        new_config = load_config(_config_path)
    except Exception as e:
        logger.exception("Failed to reload config")
        return {
            "status": "error",
            "mock_mode": _is_mock_mode,
            "num_datasets": len(_manager.list_datasets()) if _manager else 0,
            "changed_datasets": [],
            "message": f"Failed to parse config: {e}",
        }

    # Determine which datasets changed
    old_raw_configs = _manager.raw_configs if isinstance(_manager, SampleManager) else []
    new_raw_configs = new_config["raw_configs"]
    changed_indices = _diff_raw_configs(old_raw_configs, new_raw_configs)

    # Get names of changed datasets for message
    changed_names = []
    for idx in changed_indices:
        if idx < len(new_raw_configs):
            h5_path = new_raw_configs[idx].get("h5_path", "")
            changed_names.append(Path(h5_path).stem)

    if isinstance(_manager, SampleManager) and changed_indices:
        # Partial reload - only clear and reload changed datasets
        _manager.partial_reload(new_config, changed_indices)
        _start_prewarm(_manager, indices=changed_indices)
    elif changed_indices or not isinstance(_manager, SampleManager):
        # Full reload needed (e.g., switching from mock mode)
        if _manager is not None:
            _manager.clear_cache()
        _manager = SampleManager(new_config, prewarm=False)
        _is_mock_mode = False
        _start_prewarm(_manager)

    _config_content = new_content

    message = f"Reloaded {len(changed_indices)} dataset(s)"
    if changed_names:
        message += f": {', '.join(changed_names[:3])}"
        if len(changed_names) > 3:
            message += f"... (+{len(changed_names) - 3} more)"

    return {
        "status": "ok",
        "mock_mode": _is_mock_mode,
        "num_datasets": len(_manager.list_datasets()) if _manager else 0,
        "changed_datasets": list(changed_indices),
        "message": message,
    }


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Get current viewer status including startup state."""
    return {
        "ready": _startup_complete,
        "warming": _warmup_in_progress,
        "mock_mode": _is_mock_mode,
        "num_datasets": len(_manager.list_datasets()) if _manager else 0,
        "config_path": _config_path,
    }


@app.post("/api/warm/dataset/{dataset_idx}")
def warm_dataset(dataset_idx: int) -> dict[str, Any]:
    """Warm DataLoader workers for a specific dataset."""
    if not isinstance(_manager, SampleManager):
        return {"status": "noop", "message": "Mock mode: no warmup"}

    if dataset_idx < 0 or dataset_idx >= len(_manager.dataset_configs):
        raise HTTPException(status_code=400, detail=f"Invalid dataset_idx: {dataset_idx}")

    _start_prewarm(_manager, indices={dataset_idx})
    return {"status": "ok", "message": f"Warming dataset {dataset_idx}"}


@app.post("/api/warm/kind/{kind}")
def warm_kind(kind: str) -> dict[str, Any]:
    """Warm DataLoader workers for all datasets of a kind."""
    if not isinstance(_manager, SampleManager):
        return {"status": "noop", "message": "Mock mode: no warmup"}

    try:
        target_kind = DatasetKind[kind.upper()]
    except KeyError:
        valid = ", ".join(k.name for k in DatasetKind)
        raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}. Valid: {valid}") from None

    indices = {i for i, cfg in enumerate(_manager.dataset_configs) if cfg.kind == target_kind}
    if not indices:
        return {"status": "ok", "message": f"No datasets found for kind {target_kind.name}"}

    _start_prewarm(_manager, indices=indices)
    return {"status": "ok", "message": f"Warming {len(indices)} datasets for {target_kind.name}"}


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    """List all datasets grouped by kind."""
    if _manager is None:
        raise HTTPException(status_code=500, detail="Manager not initialized")

    datasets = _manager.list_datasets()

    # Group by kind for UI dropdown
    by_kind: dict[str, list[Any]] = {}
    for ds in datasets:
        kind = ds["kind"]
        by_kind.setdefault(kind, []).append(ds)

    return {
        "datasets": datasets,
        "by_kind": by_kind,
        "mock_mode": _is_mock_mode,
    }


@app.get("/api/kinds")
def list_kinds() -> dict[str, Any]:
    """List unique kinds with their dataset counts."""
    if _manager is None:
        raise HTTPException(status_code=500, detail="Manager not initialized")

    return {"kinds": _manager.get_unique_kinds()}


@app.get("/api/samples/dataset/{dataset_idx}")
def get_samples_dataset(dataset_idx: int, n: int = DEFAULT_N) -> Response:
    """Get n samples from a specific dataset.

    Returns binary float32 data with metadata in headers.
    """
    if _manager is None:
        raise HTTPException(status_code=500, detail="Manager not initialized")

    try:
        result = _manager.sample_dataset(dataset_idx, n)
        return _binary_response(result)
    except Exception as e:
        logger.exception("Error sampling dataset")
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/samples/kind/{kind}")
def get_samples_kind(kind: str, n: int = DEFAULT_N) -> Response:
    """Get n samples from datasets of given kind using training distribution.

    Returns binary float32 data with source dataset labels in headers.
    """
    if _manager is None:
        raise HTTPException(status_code=500, detail="Manager not initialized")

    try:
        target_kind = DatasetKind[kind.upper()]
    except KeyError:
        valid = ", ".join(k.name for k in DatasetKind)
        raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}. Valid: {valid}") from None

    try:
        result = _manager.sample_kind(target_kind, n)
        return _binary_response(result)
    except Exception as e:
        logger.exception("Error sampling kind")
        raise HTTPException(status_code=400, detail=str(e)) from e


def _binary_response(result: SampleResult) -> Response:
    """Create binary response with float32 images + masks, metadata in headers."""
    # Pack images and masks into single buffer
    # Format: images (n*2*H*W float32) + penumbra (n*H*W uint8) + background (n*H*W uint8)
    n = result.images.shape[0]
    h, w = result.images.shape[2], result.images.shape[3]

    images_bytes = result.images.astype(np.float32).tobytes()
    penumbra_bytes = result.penumbra_masks.astype(np.uint8).tobytes()
    background_bytes = result.background_masks.astype(np.uint8).tobytes()

    buffer = images_bytes + penumbra_bytes + background_bytes

    return Response(
        content=buffer,
        media_type="application/octet-stream",
        headers={
            "X-Shape": json.dumps([n, 2, h, w]),
            "X-Labels": json.dumps(result.labels),
            "X-Kinds": json.dumps(result.kinds),
            "X-Has-Masks": "true",
        },
    )


# Serve static frontend assets
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    # Mount assets subdirectory if it exists
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


def run_server(config: str = "data.yaml", host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the viewer server."""
    global _config_path
    _config_path = config

    import uvicorn

    uvicorn.run(app, host=host, port=port)
