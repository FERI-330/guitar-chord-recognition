"""
Logging configuration for guitar chord recognition system.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "guitar_chord_recognition.log",
    level: str = "INFO"
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        log_file: Name of log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('guitar_chord_recognition')
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    log_path = Path(log_dir) / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
