"""Data viewer for score_qmri datasets."""


def main() -> None:
    """CLI entry point for data viewer."""
    import argparse

    parser = argparse.ArgumentParser(description="Score-qMRI Data Viewer")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Server port")
    args = parser.parse_args()

    from .server import run_server

    run_server(config=args.config, host=args.host, port=args.port)


__all__ = ["main"]
