import re
import logging

logger = logging.getLogger(__name__)

BRIEFING_KEYWORDS = ["briefing", "brief", "brf", "proposta", "campanha", "apresentação", "deck", "summary"]
EDITAL_KEYWORDS = ["edital", "tender", "rfp", "bid", "procurement", "licitação", "termo de referência", "tr"]

def classify_filename(filename: str) -> str:
    """
    Classifies a filename into 'briefing' or 'edital' based on keywords.
    Returns 'briefing' as a default if no clear match is found.
    """
    fn_lower = filename.lower()
    
    # Check for edital keywords first
    if any(k in fn_lower for k in EDITAL_KEYWORDS):
        return "edital"
    
    # Check for briefing keywords
    if any(k in fn_lower for k in BRIEFING_KEYWORDS):
        return "briefing"
    
    # Default to briefing if unknown, but log it
    logger.warning(f"Could not find clear keywords in '{filename}'. Defaulting to 'briefing'.")
    return "briefing"
