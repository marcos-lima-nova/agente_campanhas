"""
Configurable redaction hooks for sensitive data before session storage.

Environment variables
---------------------
ENABLE_REDACTION : "true" | "false" (default "false")
REDACTION_PATTERNS : JSON list of regex patterns to redact (optional)
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import List, Optional

from src.utils.logging_config import setup_logging

logger = setup_logging("redaction")

# Default patterns for common PII (CPF, email, phone)
_DEFAULT_PATTERNS: List[str] = [
    r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",          # CPF
    r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",    # CNPJ
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b",  # BR phone
]

_REDACTION_PLACEHOLDER = "[REDACTED]"


def _get_patterns() -> List[re.Pattern]:
    """Load redaction patterns from env or use defaults."""
    raw = os.getenv("REDACTION_PATTERNS")
    if raw:
        try:
            pattern_strings = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("REDACTION_PATTERNS env var is not valid JSON; using defaults.")
            pattern_strings = _DEFAULT_PATTERNS
    else:
        pattern_strings = _DEFAULT_PATTERNS

    return [re.compile(p) for p in pattern_strings]


def is_redaction_enabled() -> bool:
    return os.getenv("ENABLE_REDACTION", "false").lower() in ("true", "1", "yes")


def redact_text(text: str, patterns: Optional[List[re.Pattern]] = None) -> str:
    """
    Apply redaction patterns to *text* and return the sanitized version.

    If ``ENABLE_REDACTION`` is not set to a truthy value this function
    returns the original text unchanged.
    """
    if not is_redaction_enabled():
        return text

    if patterns is None:
        patterns = _get_patterns()

    for pattern in patterns:
        text = pattern.sub(_REDACTION_PLACEHOLDER, text)

    return text
