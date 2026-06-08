"""Logging utilities for establishing consistent console and file logging.

Sets up a root logger that writes to stdout and a timestamped file under a
specified log directory, with configurable level controls for third-party libraries.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_global_logging(app_name: str, log_dir: str = "log") -> logging.Logger:
    """Configure root logging with a console stream and timestamped log file.

    Ensures the log directory exists, configures formatting, registers console
    and file handlers on the root logger, and silences verbose third-party APIs.

    Args:
        app_name: Name of the application (used to name the log file).
        log_dir: Directory where the log file will be saved.

    Returns:
        Logger instance associated with app_name.
    """
    dir_path = Path(log_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = dir_path / f"{app_name}_{timestamp}.log"

    # Safeguard stdout against Unicode encoding crashes in restricted terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clean up any existing handlers to prevent duplicate lines
    root_logger.handlers = []

    # Stream to console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


    # Write to file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Prevent logs from double-bubbling
    root_logger.propagate = False

    # Quieten chatty external libraries
    for library in ("transformers", "datasets", "urllib3", "pynvml", "asyncio", "vllm", "modelopt", "llmcompressor"):
        logging.getLogger(library).setLevel(logging.WARNING)

    return logging.getLogger(app_name)
