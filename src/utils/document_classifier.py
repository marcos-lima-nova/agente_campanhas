import re

BRIEFING_KEYWORDS = ["briefing", "brief", "brf"]
EDITAL_KEYWORDS = ["edital", "tender", "rfp", "bid", "procurement"]

def classify_filename(filename: str) -> str:
    """
    Classifies a filename into 'briefing' or 'edital' based on keywords.
    Returns the classification string or raises ValueError if unknown.
    """
    fn_lower = filename.lower()
    
    # Check for edital keywords first (RFPs are usually more specific)
    if any(k in fn_lower for k in EDITAL_KEYWORDS):
        return "edital"
        
    # Check for briefing keywords
    if any(k in fn_lower for k in BRIEFING_KEYWORDS):
        return "briefing"
        
    raise ValueError(
        f"Could not infer document type from filename: '{filename}'. "
        f"Expected keywords for Briefing: {BRIEFING_KEYWORDS}. "
        f"Expected keywords for Edital: {EDITAL_KEYWORDS}."
    )
