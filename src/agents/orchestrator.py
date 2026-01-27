import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.agents.briefing_analyzer import BriefingAnalyzer
from src.agents.edital_analyzer import EditalAnalyzer
from src.agents.unified_analyzer import UnifiedAnalyzer
from src.utils.document_classifier import classify_filename
from src.utils.logging_config import setup_logging
from src.ingestion.processors import DocumentProcessor

logger = setup_logging()

class DocumentOrchestrator:
    def __init__(self):
        self.briefing_analyzer = BriefingAnalyzer()
        self.edital_analyzer = EditalAnalyzer()
        self.unified_analyzer = UnifiedAnalyzer()
        self.processor = DocumentProcessor()

    def analyze_document(self, file_bytes: Optional[bytes] = None, filename: str = "document", content_type: Optional[str] = None, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies and analyzes a document.
        Returns a dictionary with 'markdown', 'filename', and 'doc_type'.
        Supports either raw bytes (needs extraction) or pre-extracted content.
        """
        try:
            # 1. Mandatory Classification
            doc_type = classify_filename(filename)
            
            if doc_type == "invalid":
                logger.warning(f"Aborting analysis for non-compliant file: {filename}")
                return {
                    "markdown": "Warning: The file is not valid because its name does not comply.",
                    "filename": filename,
                    "doc_type": "invalid"
                }
                
            logger.info(f"Automatically classified '{filename}' as {doc_type}")
            
            # 2. Mandatory Analysis Execution
            if not content:
                if file_bytes is None:
                    raise ValueError("Either file_bytes or content must be provided.")
                content = self.processor.extract_text(file_bytes, filename=filename)

            if doc_type == "briefing":
                markdown = self.briefing_analyzer.analyze_briefing(b"", filename, content_type, content=content)
            else: # edital
                markdown = self.edital_analyzer.analyze_edital(b"", filename, content_type, content=content)
            
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


    def analyze_unified(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts content from multiple files and runs the unified analysis.
        Each file in 'files' should have 'bytes' and 'name'.
        """
        logger.info(f"Orchestrating unified analysis for {len(files)} files.")
        
        extracted_files = []
        for f in files:
            filename = f.get("name", "document")
            content = f.get("content")
            file_bytes = f.get("bytes")
            
            try:
                # 1. Extract text if not provided
                if not content:
                    if not file_bytes:
                        continue
                    content = self.processor.extract_text(file_bytes, filename=filename)
                
                # 2. Classify
                doc_type = classify_filename(filename)

                if doc_type == "invalid":
                    logger.warning(f"Aborting unified analysis due to non-compliant file: {filename}")
                    return {
                        "markdown": "Warning: The file is not valid because its name does not comply.",
                        "filename": filename,
                        "doc_type": "invalid"
                    }
                
                extracted_files.append({
                    "filename": filename,
                    "content": content,
                    "type": doc_type
                })
            except Exception as e:
                logger.error(f"Failed to process {filename} during batch extraction: {e}")
                continue

        if not extracted_files:
            raise RuntimeError("No files were successfully processed for unified analysis.")

        # 3. Run Unified Analysis
        markdown = self.unified_analyzer.analyze_unified(extracted_files)
        
        if not markdown:
            raise RuntimeError("Unified analysis returned no result.")

        return {
            "markdown": markdown,
            "filename": "unified_document_analysis.md",
            "doc_type": "unified"
        }
