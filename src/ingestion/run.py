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
REPERTORIO_FOLDER = Path(os.getenv("REPERTORIO_FOLDER", "data/fichas_de_repertorio"))
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
    
    # Recursive file discovery
    files_to_process = []
    extensions = ["*.pdf", "*.docx"]
    for ext in extensions:
        try:
            # rglob handles recursive search
            files_to_process.extend(list(REPERTORIO_FOLDER.rglob(ext)))
        except PermissionError as pe:
            logger.error(f"Permission denied while traversing {REPERTORIO_FOLDER}: {pe}")
        except Exception as e:
            logger.error(f"Unexpected error during directory traversal: {e}")
    
    if not files_to_process:
        logger.info(f"No files found in {REPERTORIO_FOLDER} or its subdirectories.")
        return

    all_chunks = []
    
    for file_path in files_to_process:
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        
        # Determine client based on directory structure
        # data/fichas_de_repertorio/ClientName/Project/file.pdf -> ClientName
        try:
            relative_path = file_path.relative_to(REPERTORIO_FOLDER)
            path_parts = relative_path.parts
            client = path_parts[0] if len(path_parts) > 1 else "General"
        except Exception:
            client = "General"

        try:
            file_hash = calculate_file_hash(file_path)
        except (PermissionError, OSError) as e:
            logger.error(f"Could not access file {file_path}: {e}")
            continue

        if manifest.is_already_ingested(file_hash, filename):
            logger.info(f"Skipping {filename} (already ingested).")
            continue
        
        logger.info(f"Processing {filename} (Client: {client})...")
        try:
            text = processor.extract_text(file_path)
            campaign = processor.infer_campaign(filename)
            metadata = {
                "filename": filename,
                "source_path": str(file_path),
                "relative_path": str(relative_path),
                "file_type": file_path.suffix,
                "hash": file_hash,
                "campaign": campaign,
                "client": client
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
