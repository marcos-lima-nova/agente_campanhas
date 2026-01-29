import sys
import argparse
from pathlib import Path
from src.agents.orchestrator import DocumentOrchestrator
from src.utils.logging_config import setup_logging

logger = setup_logging("agents")

def main():
    parser = argparse.ArgumentParser(description="Automatically analyze a briefing or edital based on filename.")
    parser.add_argument("file_path", help="Path to the PDF or DOCX file.")
    
    args = parser.parse_args()
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
        
    try:
        orchestrator = DocumentOrchestrator()
        
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        logger.info(f"Running automatic analysis for: {file_path.name}")
        result = orchestrator.analyze_document(file_bytes, file_path.name)
        
        print("\n" + "="*50)
        print(f"DOCUMENT TYPE: {result['doc_type'].upper()}")
        print(f"OUTPUT FILE: {result['filename']}")
        print("="*50 + "\n")
        print(result['markdown'])
        print("\n" + "="*50)
        
    except ValueError as ve:
        logger.error(str(ve))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
