"""FastAPI server for data viewer."""

from __future__ import annotations

import json
import logging
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
from .sampling import SampleManager, SampleResult

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
_config_path: str = "config.yaml"


def _load_manager(config_path: str) -> tuple[SampleManager | MockSampleManager, bool]:
    """Load sample manager, falling back to mock if needed."""
    try:
        config = load_config(config_path)
        logger.info(f"Loaded {len(config['dataset_configs'])} datasets from {config_path}")
        if config.get("skipped_paths"):
            logger.warning(f"Skipped missing files: {config['skipped_paths'][:3]}...")
        return SampleManager(config), False
    except FileNotFoundError as e:
        logger.warning(f"No valid datasets found, using mock mode: {e}")
        return MockSampleManager(), True
    except Exception as e:
        logger.exception(f"Config load failed, using mock mode: {e}")
        return MockSampleManager(), True


@app.on_event("startup")
def startup() -> None:
    """Initialize manager on startup."""
    global _manager, _is_mock_mode
    _manager, _is_mock_mode = _load_manager(_config_path)


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
    """Re-read config.yaml and reinitialize datasets."""
    global _manager, _is_mock_mode

    if _manager is not None:
        _manager.clear_cache()

    _manager, _is_mock_mode = _load_manager(_config_path)

    return {
        "status": "ok" if not _is_mock_mode else "fallback",
        "mock_mode": _is_mock_mode,
        "num_datasets": len(_manager.list_datasets()),
    }


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Get current viewer status."""
    return {
        "mock_mode": _is_mock_mode,
        "num_datasets": len(_manager.list_datasets()) if _manager else 0,
        "config_path": _config_path,
    }


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    """List all datasets grouped by kind."""
    if _manager is None:
        raise HTTPException(status_code=500, detail="Manager not initialized")

    datasets = _manager.list_datasets()

    # Group by kind for UI dropdown
    by_kind: dict[str, list] = {}
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
def get_samples_dataset(dataset_idx: int, n: int = 32) -> Response:
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
def get_samples_kind(kind: str, n: int = 32) -> Response:
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


def run_server(config: str = "config.yaml", host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the viewer server."""
    global _config_path
    _config_path = config

    import uvicorn

    uvicorn.run(app, host=host, port=port)
