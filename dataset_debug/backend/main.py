"""FastAPI backend for dataset debug viewer."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

fastapi_name = "".join(["fast", "api"])
fastapi = importlib.import_module(fastapi_name)
FastAPI = fastapi.FastAPI
HTTPException = fastapi.HTTPException
cors_name = ".".join([fastapi_name, "middleware", "cors"])
responses_name = ".".join([fastapi_name, "responses"])
static_name = ".".join([fastapi_name, "staticfiles"])
CORSMiddleware = importlib.import_module(cors_name).CORSMiddleware
responses = importlib.import_module(responses_name)
FileResponse = responses.FileResponse
JSONResponse = responses.JSONResponse
StaticFiles = importlib.import_module(static_name).StaticFiles

from .annotations import AnnotationField, AnnotationStore  # noqa: E402
from .config_loader import load_config  # noqa: E402
from .dataset_manager import DatasetManager  # noqa: E402
from .mock_manager import MockDatasetManager  # noqa: E402

_mock_mode = False
_mock_manager: MockDatasetManager | None = None
_real_manager: DatasetManager | None = None


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_state(config_path: str | None) -> tuple[DatasetManager | MockDatasetManager, AnnotationStore, bool]:
    global _mock_mode, _mock_manager, _real_manager
    try:
        config = load_config(config_path or "config.yaml")
        if len(config.dataset_configs) == 0:
            raise ValueError("No datasets configured")
        _real_manager = DatasetManager(config)
        annotations = AnnotationStore(Path(__file__).parent, config.config_path)
        return _real_manager, annotations, False
    except Exception as e:
        logger.warning(f"Failed to load real datasets, falling back to mock mode. Error: {e}")
        _mock_manager = MockDatasetManager()
        annotations = AnnotationStore(Path(__file__).parent, Path("mock_config.yaml"))
        _mock_mode = True
        return _mock_manager, annotations, True


config_path = os.environ.get("DATASET_DEBUG_CONFIG")
_manager, _annotations, _mock_mode = _load_state(config_path)

app = FastAPI(title="Dataset Debug Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
def index() -> FileResponse:
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(index_path)


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    return {
        "mock_mode": _mock_mode,
        "message": "Using mock phantom data - no real datasets available" if _mock_mode else "Real datasets loaded",
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    if _mock_mode:
        return {
            "config_path": "mock://phantom",
            "data_settings": {"mock": True, "num_datasets": 3},
            "mock_mode": True,
        }
    return {
        "config_path": "config.yaml",
        "data_settings": {},
        "mock_mode": False,
    }


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    datasets = _manager.list_datasets()
    return {"datasets": datasets, "mock_mode": _mock_mode}


@app.get("/api/datasets/{dataset_idx}/sample")
def get_sample(
    dataset_idx: int,
    index: int,
    augment: bool = True,
    seed: int | None = None,
    display_mode: str = "real_imag",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Any]:
    try:
        result = _manager.get_sample(dataset_idx, index, augment, seed, display_mode, vmin, vmax)
        result["mock_mode"] = _mock_mode
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in get_sample")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_idx}/sample-by-hierarchy")
def get_sample_by_hierarchy(
    dataset_idx: int,
    subject: int,
    slice: int,
    coil: int | None = None,
    augment: bool = True,
    seed: int | None = None,
    display_mode: str = "real_imag",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Any]:
    try:
        result = _manager.get_sample_by_hierarchy(dataset_idx, subject, slice, coil, augment, seed, display_mode, vmin, vmax)
        result["mock_mode"] = _mock_mode
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in get_sample_by_hierarchy")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_idx}/batch")
def get_batch_data(dataset_idx: int, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        indices = payload.get("indices", [])
        augment = payload.get("augment", True)
        seed = payload.get("seed", None)
        display_mode = payload.get("display_mode", "real_imag")
        return _manager.get_batch_data(dataset_idx, indices, augment, seed, display_mode)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in get_batch_data")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_idx}/random-index")
def get_random_index(dataset_idx: int) -> dict[str, int]:
    return {"index": _manager.random_index(dataset_idx)}


@app.get("/api/datasets/{dataset_idx}/histogram")
def get_histogram(
    dataset_idx: int,
    index: int,
    augment: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    try:
        return _manager.get_histogram(dataset_idx, index, augment, seed)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in get_histogram")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/annotation-fields")
def get_annotation_fields() -> dict[str, Any]:
    return {"fields": [field.__dict__ for field in _annotations.get_fields()]}


@app.post("/api/annotation-fields")
def add_annotation_field(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        field = AnnotationField(
            id=payload["id"],
            label=payload["label"],
            field_type=payload["field_type"],
        )
        _annotations.add_field(field)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"fields": [field.__dict__ for field in _annotations.get_fields()]}


@app.delete("/api/annotation-fields/{field_id}")
def delete_annotation_field(field_id: str) -> dict[str, Any]:
    try:
        _annotations.remove_field(field_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"fields": [field.__dict__ for field in _annotations.get_fields()]}


@app.get("/api/annotations/{dataset_idx}")
def get_dataset_annotations(dataset_idx: int) -> dict[str, Any]:
    return {"values": _annotations.get_dataset_annotations(dataset_idx)}


@app.put("/api/annotations/{dataset_idx}")
def set_dataset_annotations(dataset_idx: int, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        _annotations.set_dataset_annotations(dataset_idx, payload.get("values", {}))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"values": _annotations.get_dataset_annotations(dataset_idx)}


@app.get("/api/annotations/export")
def export_annotations() -> JSONResponse:
    data = _annotations.export()
    return JSONResponse(content=data)
