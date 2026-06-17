from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("t2i_service")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_dir / "t2i_service.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    request_handler = RotatingFileHandler(
        log_dir / "requests.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    request_handler.setFormatter(formatter)
    request_logger = logging.getLogger("t2i_service.requests")
    request_logger.setLevel(getattr(logging, level, logging.INFO))
    request_logger.handlers.clear()
    request_logger.propagate = False
    request_logger.addHandler(request_handler)
    request_logger.addHandler(stream_handler)

    return logger
