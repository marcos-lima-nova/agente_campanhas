import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.agents.briefing_analyzer import BriefingAnalyzer
from src.agents.edital_analyzer import EditalAnalyzer
from src.utils.document_classifier import classify_filename
from src.utils.logging_config import setup_logging

logger = setup_logging()

class DocumentOrchestrator:
    def __init__(self):
        self.briefing_analyzer = BriefingAnalyzer()
        self.edital_analyzer = EditalAnalyzer()

    def analyze_document(self, file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies and analyzes a document.
        Returns a dictionary with 'markdown', 'filename', and 'doc_type'.
        """
        try:
            # 1. Mandatory Classification
            doc_type = classify_filename(filename)
            logger.info(f"Automatically classified '{filename}' as {doc_type}")
            
            # 2. Mandatory Analysis Execution
            if doc_type == "briefing":
                markdown = self.briefing_analyzer.analyze_briefing(file_bytes, filename, content_type)
            else: # edital
                markdown = self.edital_analyzer.analyze_edital(file_bytes, filename, content_type)
            
            if not markdown:
                raise RuntimeError(f"Analysis failed for {doc_type}: {filename}")
                
            return {
                "markdown": markdown,
                "filename": f"{Path(filename).stem}_{doc_type}_summary.md",
                "doc_type": doc_type
            }
            
        except ValueError as ve:
            logger.error(f"Classification error: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            raise e
