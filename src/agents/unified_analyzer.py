import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

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

class UnifiedAnalyzer:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.summaries_folder = SUMMARIES_FOLDER
        self.summaries_folder.mkdir(parents=True, exist_ok=True)
        
        template = """
You are a senior expert in marketing, strategic planning, and analysis of advertising campaign briefings, with extensive experience in RFPs, tenders, and the development of Solution Reports.
Develop the following analysis entirely in Portuguese-BR.Your task is to thoroughly analyze a marketing campaign briefing and extract, structure, and synthesize the most relevant information in a clear, objective, and standardized manner.
The purpose of this summary is to serve as input for a LLM responsible for searching and ranking similar campaigns in a historical database, in order to select cases that demonstrate the agency’s ability to meet the client’s requirements (Solution Report).
When analyzing the briefing, explicitly identify and organize, whenever available, the following elements:

1- Client context
Industry/market segment
Type of institution (public, private, mixed, non-profit, etc.)
Core business challenge or central problem

2- Campaign objectives
Primary objective
Secondary objectives (e.g., awareness, engagement, conversion, institutional positioning, public education, etc.)

3- Campaign scope
Type of campaign (institutional, promotional, educational, digital, 360°, etc.)
Main expected deliverables (e.g., digital assets, offline media, content, strategy, technology, events, etc.)

4- Target audience
Demographic and/or behavioral profile
Relevant stakeholders (when applicable)

5- Channels and media
Requested or expected communication channels
Predominant environment (digital, offline, or hybrid)
Technical and strategic requirements

6- Constraints, legal, or institutional requirements
Technical criteria relevant for experience validation

7- Similarity indicators for campaign retrieval
Strategic keywords:
Expected solution types
Elements that characterize comparable campaigns

At the end, generate:

- A structured executive summary
- A list of keywords and tags to facilitate the retrieval of similar campaigns

Be precise, avoid assumptions not grounded in the text, and prioritize information that helps demonstrate the agency’s technical, strategic, and operational capabilities.
---
CONTEÚDO DOS DOCUMENTOS:
{{content}}
---
        """
        
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["content"]))
        
        # Initialize LLM via Factory
        self.pipeline.add_component("llm", get_llm_generator(
            model_name="gpt-4o-mini",
            timeout=180.0 # Longer timeout for multiple files
        ))
        self.pipeline.connect("prompt_builder", "llm")

    def analyze_unified(self, files_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        Analyzes a batch of documents and returns a unified markdown summary.
        files_data: List of dicts with {'filename': str, 'content': str, 'type': str}
        """
        if not files_data:
            logger.warning("No files provided for unified analysis.")
            return None

        logger.info(f"Starting unified analysis for {len(files_data)} files.")
        
        # Combine all contents with labels
        combined_text = ""
        filenames = []
        for i, file in enumerate(files_data):
            filename = file.get('filename', f'Arquivo_{i+1}')
            filenames.append(filename)
            doc_type = file.get('type', 'unknown').upper()
            content = file.get('content', '')
            
            combined_text += f"\n--- DOCUMENTO {i+1}: {filename} (Tipo: {doc_type}) ---\n"
            combined_text += content
            combined_text += "\n\n"

        try:
            logger.info("Sending batch content to LLM for unified analysis...")
            result = self.pipeline.run({
                "prompt_builder": {"content": combined_text}
            })
            
            summary_content = result["llm"]["replies"][0]
            
            # Save the unified summary
            batch_id = "-".join([Path(fn).stem for fn in filenames[:3]]) # use first 3 filenames
            output_file = self.summaries_folder / f"unified_analysis_{batch_id[:50]}.md"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary_content)
            
            logger.info(f"Unified summary saved to {output_file}")
            return summary_content
            
        except Exception as e:
            logger.error(f"Failed unified analysis: {e}")
            return None
