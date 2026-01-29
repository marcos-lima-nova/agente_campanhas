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
from src.utils.llm_factory import get_llm_generator
from src.utils.logging_config import setup_logging
from src.ingestion.processors import DocumentProcessor

# Load config
load_dotenv()
SUMMARIES_FOLDER = Path("data/summaries")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = setup_logging("agents")

class EditalAnalyzer:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.summaries_folder = SUMMARIES_FOLDER
        self.summaries_folder.mkdir(parents=True, exist_ok=True)
        
        # Define the analysis prompt template for Bidding Documents (Editais)
        template = """
You are a senior expert in bidding processes, public RFPs, and strategic planning for marketing campaigns. You have extensive experience in analyzing government and private "Editais" (Bidding Documents).

Develop the following analysis entirely in Portuguese-BR.
Your task is to thoroughly analyze the following Bidding Document (Edital) and extract, structure, and synthesize the most critical requirements, constraints, and validation criteria.

The purpose of this summary is to provide a technical and legal roadmap for the agency to evaluate its technical capacity and prepare a competitive proposal.

When analyzing the Edital, explicitly identify and organize, whenever available, the following elements:

### Institutional Context
- Issuing entity (Public, Private, Mixed)
- Core service or product requested
- Total estimated budget (if available)

### Technical Requirements
- Minimum technical experience required (Atestados de Capacidade Técnica)
- Specific tools, technologies, or methodologies mandated
- Key personnel requirements (Certifications, and experience)

### Strategic Scope
- Main communication challenge described
- Requested deliverables (Digital, Offline, Events, PR, etc.)
- Evaluation criteria (Technical score vs. Price)

### Execution Constraints
- Legal and compliance requirements
- Deadlines and milestones
- Territorial scope (Local, National, International)

### Risk and Critical Success Factors
- Main pitfalls or common elimination reasons
- Key differentiators requested

At the end, generate:
A structured executive summary and a list of Technical Keywords to facilitate the search for similar internal cases.

Be precise, do not assume information not present in the text, and prioritize data that impacts the "Go/No-Go" decision and technical validation.

---
Edital Content:
{{content}}
---
        """
        
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["content"]))
        
        # Initialize LLM via Factory
        self.pipeline.add_component("llm", get_llm_generator(
            model_name="gpt-4o-mini",
            timeout=120.0
        ))
        self.pipeline.connect("prompt_builder", "llm")

    def analyze_edital(self, file_bytes: bytes, filename: str, content_type: Optional[str] = None, content: Optional[str] = None) -> Optional[str]:
        """
        Analyzes an edital provided as an attachment (bytes) or pre-extracted content (str).
        Returns the generated summary markdown content.
        """
        logger.info(f"Analyzing edital: {filename}")
        
        # Validate file extension only if we need to extract from bytes
        if not content:
            suffix = Path(filename).suffix.lower()
            if suffix not in [".pdf", ".docx"]:
                logger.error(f"Unsupported file type for extraction: {suffix}. Only .pdf and .docx are supported.")
                return None

        try:
            # Extract text if not already provided
            if not content:
                content = self.processor.extract_text(file_bytes, filename=filename)
            
            if not content or not content.strip():
                logger.warning(f"No content available for analysis of {filename}")
                return None

            logger.info(f"Analyzing {len(content)} characters. Sending to LLM...")

            result = self.pipeline.run({
                "prompt_builder": {"content": content}
            })
            
            summary_content = result["llm"]["replies"][0]
            
            # Save to file as well for record
            stem = Path(filename).stem
            output_file = self.summaries_folder / f"{stem}_edital_summary.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary_content)
            
            logger.info(f"Edital summary saved to {output_file}")
            return summary_content
            
        except Exception as e:
            logger.error(f"Failed to analyze edital {filename}: {e}")
            return None

    def analyze_file(self, file_path: Union[Path, str]) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        with open(path, "rb") as f:
            file_bytes = f.read()
            
        return self.analyze_edital(file_bytes, path.name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a marketing edital file.")
    parser.add_argument("file_path", help="Path to the PDF or DOCX edital file.")
    args = parser.parse_args()
    
    analyzer = EditalAnalyzer()
    summary = analyzer.analyze_file(args.file_path)
    if summary:
        print("\nAnalysis complete!\n")
        print(summary)
    else:
        print("\nAnalysis failed. Check logs for details.")
