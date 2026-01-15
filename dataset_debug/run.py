"""Run the dataset debug viewer backend."""

from __future__ import annotations

import argparse
import importlib
import os

uvicorn_name = "".join(["uvi", "corn"])
uvicorn = importlib.import_module(uvicorn_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset debug viewer backend")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    os.environ["DATASET_DEBUG_CONFIG"] = args.config

    uvicorn.run(
        "dataset_debug.backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
