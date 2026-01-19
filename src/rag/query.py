import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
from src.rag.pipeline import RAGPipeline

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.query \"Your question here\"")
        return

    question = sys.argv[1]
    
    try:
        # RAGPipeline handles internal initialization
        rag = RAGPipeline(model_name=EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error initializing RAG Pipeline: {e}")
        return

    print(f"\nQuestion: {question}")
    print("Thinking...")
    
    result = rag.query(question)
    
    print(f"\nAnswer: {result['answer']}")
    
    if result.get('sources'):
        print("\nSources:")
        for source in result['sources']:
            filename = source.get('filename') or "Unknown Source"
            campaign = source.get('campaign') or "Unknown Campaign"
            print(f"- {filename} (Campaign: {campaign})")
    else:
        print("\nNo sources cited.")

if __name__ == "__main__":
    main()
