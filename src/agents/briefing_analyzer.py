import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import logging
import io

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from src.utils.logging_config import setup_logging
from src.ingestion.processors import DocumentProcessor

# Load config
load_dotenv()
SUMMARIES_FOLDER = Path("data/summaries")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = setup_logging()

class BriefingAnalyzer:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.summaries_folder = SUMMARIES_FOLDER
        self.summaries_folder.mkdir(parents=True, exist_ok=True)
        
        # Define the analysis prompt template
        template = """

You are a senior expert in marketing, strategic planning, and analysis of advertising campaign briefings, with extensive experience in RFPs, tenders, and the development of Solution Reports.

Develop the following analysis entirely in Portuguese-BR.
Your task is to thoroughly analyze the following marketing campaign briefing and extract, structure, and synthesize the most relevant information in a clear, objective, and standardized manner.

The purpose of this summary is to serve as input for a LLM responsible for searching and ranking similar campaigns in a historical database, in order to select cases that demonstrate the agency’s ability to meet the client’s requirements (Solution Report).

When analyzing the briefing, explicitly identify and organize, whenever available, the following elements:

### Client context
- Industry/market segment
- Type of institution (public, private, mixed, non-profit, etc.)
- Core business challenge or central problem

### Campaign objectives
- Primary objective
- Secondary objectives (e.g., awareness, engagement, conversion, institutional positioning, public education, etc.)

### Campaign scope
- Type of campaign (institutional, promotional, educational, digital, 360°, etc.)
- Main expected deliverables (e.g., digital assets, offline media, content, strategy, technology, events, etc.)

### Target audience
- Demographic and/or behavioral profile
- Relevant stakeholders (when applicable)

### Channels and media
- Requested or expected communication channels
- Predominant environment (digital, offline, or hybrid)

### Technical and strategic requirements
- Constraints, legal, or institutional requirements
- Technical criteria relevant for experience validation

### Similarity indicators for campaign retrieval
- Strategic keywords
- Expected solution types
- Elements that characterize comparable campaigns

At the end, generate:
A structured executive summary and a list of keywords and tags to facilitate the retrieval of similar campaigns.

Be precise, avoid assumptions not grounded in the text, prioritize information that helps demonstrate the agency’s technical, strategic, and operational capabilities.


---
Briefing Content:
{{content}}
---
        """
        
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["content"]))
        
        # Initialize LLM. Using a longer timeout (120s) for large briefings.
        api_key = Secret.from_token(OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.pipeline.add_component("llm", OpenAIGenerator(
            api_key=api_key, 
            model="gpt-4o-mini",
            timeout=120.0
        ))
        self.pipeline.connect("prompt_builder", "llm")

    def analyze_briefing(self, file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> Optional[str]:
        """
        Analyzes a briefing provided as an attachment (bytes).
        Returns the generated summary markdown content.
        """
        logger.info(f"Analyzing briefing from attachment: {filename}")
        
        # Validate file extension
        suffix = Path(filename).suffix.lower()
        if suffix not in [".pdf", ".docx"]:
            logger.error(f"Unsupported file type: {suffix}. Only .pdf and .docx are supported.")
            return None

        try:
            # Extract text using the updated DocumentProcessor
            content = self.processor.extract_text(file_bytes, filename=filename)
            
            if not content.strip():
                logger.warning(f"No text extracted from {filename}")
                return None

            logger.info(f"Extracted {len(content)} characters. Sending to LLM...")

            result = self.pipeline.run({
                "prompt_builder": {"content": content}
            })
            
            summary_content = result["llm"]["replies"][0]
            
            # Generate output filename
            stem = Path(filename).stem
            output_file = self.summaries_folder / f"{stem}_summary.md"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary_content)
            
            logger.info(f"Summary saved to {output_file}")
            return summary_content
            
        except Exception as e:
            logger.error(f"Failed to analyze briefing {filename}: {e}")
            return None

    def analyze_file(self, file_path: Union[Path, str]) -> Optional[str]:
        """
        Legacy/Helper method to analyze a file from a local path.
        Now reads the file into bytes and calls analyze_briefing.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        with open(path, "rb") as f:
            file_bytes = f.read()
            
        return self.analyze_briefing(file_bytes, path.name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a marketing briefing file.")
    parser.add_argument("file_path", help="Path to the PDF or DOCX briefing file.")
    
    args = parser.parse_args()
    
    analyzer = BriefingAnalyzer()
    summary = analyzer.analyze_file(args.file_path)
    
    if summary:
        print("\nAnalysis complete!\n")
        print(summary)
    else:
        print("\nAnalysis failed. Check logs for details.")
