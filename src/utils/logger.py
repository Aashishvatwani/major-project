"""
Aerospace-Grade Console and File Logger
Provides structured, timestamped logging with Rich formatting and severity color-coding.
"""

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def setup_aerospace_logger(
    name: str = "SatelliteHITL",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configures high-visibility logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if HAS_RICH:
        console = Console(color_system="auto")
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        rich_handler.setLevel(level)
        logger.addHandler(rich_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
