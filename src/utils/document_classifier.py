import re
import logging

logger = logging.getLogger(__name__)

BRIEFING_VARIANTS = ["briefing", "breafing", "brefing", "briefng"]
EDITAL_VARIANTS = ["edital", "editl", "edtial"]

def classify_filename(filename: str) -> str:
    """
    Classifies a filename into 'briefing' or 'edital' based on strict orthographic variants.
    Returns 'invalid' if no mandatory keywords are detected.
    """
    fn_lower = filename.lower()
    
    # Check for edital variants
    if any(v in fn_lower for v in EDITAL_VARIANTS):
        return "edital"
    
    # Check for briefing variants
    if any(v in fn_lower for v in BRIEFING_VARIANTS):
        return "briefing"
    
    logger.warning(f"File naming non-compliant: '{filename}'.")
    return "invalid"
