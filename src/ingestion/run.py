import os
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
from src.utils.logging_config import setup_logging
from src.utils.hashing import calculate_file_hash
from src.utils.manifest import ManifestManager
from src.ingestion.processors import DocumentProcessor
from src.indexing.pipeline import IndexingPipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
import json

# Load config
load_dotenv()
INBOX_FOLDER = Path(os.getenv("INBOX_FOLDER", "data/inbox"))
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "data/manifest.json")
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "vectorstore/")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

logger = setup_logging()

def run_ingestion():
    logger.info("Starting ingestion process...")
    
    manifest = ManifestManager(MANIFEST_PATH)
    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # Use ChromaDocumentStore for persistence
    try:
        document_store = ChromaDocumentStore(persist_path=str(VECTOR_STORE_PATH), collection_name="documents")
        logger.info(f"Initialized ChromaDocumentStore at {VECTOR_STORE_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDocumentStore: {e}")
        return

    indexer = IndexingPipeline(model_name=EMBEDDING_MODEL, document_store=document_store)
    
    files_to_process = list(INBOX_FOLDER.glob("*.pdf")) + list(INBOX_FOLDER.glob("*.docx"))
    
    if not files_to_process:
        logger.info("No files found in inbox.")
        return

    all_chunks = []
    
    for file_path in files_to_process:
        filename = file_path.name
        file_hash = calculate_file_hash(file_path)
        
        if manifest.is_already_ingested(file_hash, filename):
            logger.info(f"Skipping {filename} (already ingested).")
            continue
        
        logger.info(f"Processing {filename}...")
        try:
            text = processor.extract_text(file_path)
            campaign = processor.infer_campaign(filename)
            metadata = {
                "filename": filename,
                "source_path": str(file_path),
                "file_type": file_path.suffix,
                "hash": file_hash,
                "campaign": campaign
            }
            
            chunks = processor.create_chunks(text, metadata)
            # Ensure each chunk has a unique ID by appending index to hash
            for i, chunk in enumerate(chunks):
                chunk.id = f"{file_hash}_{i}"
            
            all_chunks.extend(chunks)
            
            # Update manifest
            manifest.update(filename, file_hash, file_path.suffix, str(file_path), campaign)
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")

    if all_chunks:
        logger.info(f"Indexing {len(all_chunks)} chunks into Chroma...")
        try:
            indexer.run(all_chunks)
            logger.info(f"Successfully indexed and persisted {len(all_chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to run indexing pipeline: {e}")
    else:
        logger.info("No new content to index.")

if __name__ == "__main__":
    run_ingestion()
