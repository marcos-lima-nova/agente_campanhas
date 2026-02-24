import logging
import sys
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Configuration Constants
LOG_DIR = Path("logs")
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
STRUCTURED_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | session=%(session_id)s | %(message)s"
TRACE_LOG_FILE = LOG_DIR / "file_analysis_trace.log"


def setup_trace_logging(level: int = logging.DEBUG) -> logging.Logger:
    """
    Sets up a dedicated logger for capturing detailed file analysis traces.
    Outputs to logs/file_analysis_trace.log with rotation.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trace")
    
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(STRUCTURED_FORMAT)

    # Dedicated Trace File Handler
    trace_handler = RotatingFileHandler(
        TRACE_LOG_FILE, 
        maxBytes=MAX_BYTES, 
        backupCount=BACKUP_COUNT, 
        encoding="utf-8"
    )
    trace_handler.setFormatter(formatter)
    logger.addHandler(trace_handler)
    
    # Add session context filter
    logger.addFilter(SessionContextFilter())

    return logger


class SessionContextFilter(logging.Filter):
    """
    Logging filter that injects ``session_id`` into every log record.

    Usage::

        logger = setup_logging("api")
        logger.addFilter(SessionContextFilter("my-session-id"))
    """

    def __init__(self, session_id: str = "-"):
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = self.session_id  # type: ignore[attr-defined]
        return True


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

    # 4. Default session context filter (session_id="-" until overridden)
    logger.addFilter(SessionContextFilter())

    return logger


def bind_session_id(logger: logging.Logger, session_id: str) -> None:
    """
    Replace the ``SessionContextFilter`` on *logger* so that all
    subsequent log records carry the given *session_id*.
    """
    # Remove existing SessionContextFilter(s)
    for f in list(logger.filters):
        if isinstance(f, SessionContextFilter):
            logger.removeFilter(f)
    logger.addFilter(SessionContextFilter(session_id))
