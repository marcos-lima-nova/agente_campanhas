import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Configuration Constants
LOG_DIR = Path("logs")
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

def setup_logging(category: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger for a specific category.
    Logs are saved to logs/{category}.log and errors are also sent to logs/errors.log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use 'rag_system' as root segments for proper hierarchy if needed, 
    # but here we use the category as the logger name.
    logger = logging.getLogger(category)
    
    # Avoid duplicate handlers if setup_logging is called multiple times for the same category
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(DEFAULT_FORMAT)

    # 1. Category-specific File Handler
    log_file = LOG_DIR / f"{category}.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Global Error File Handler (Unified for all categories)
    error_file = LOG_DIR / "errors.log"
    error_handler = RotatingFileHandler(error_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # 3. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

