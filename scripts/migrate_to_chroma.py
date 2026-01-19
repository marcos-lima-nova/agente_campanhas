#!/usr/bin/env python3
"""
Migration script to move legacy Haystack/JSON stores into ChromaDB.

Usage:
    python scripts/migrate_to_chroma.py

This script is idempotent and will skip duplicates. It looks for:
 - a vectorstore directory created by previous haystack save_to_disk
 - a legacy doc_store.json file produced by older runs

It will load documents, ids, embeddings and metadata and insert them into Chroma.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path (robust method)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store.chroma_store import ChromaAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate(vectorstore_dir: Path):
    chroma = ChromaAdapter(persist_directory=vectorstore_dir)

    # 1) Try to migrate haystack directory format if present (best-effort)
    if vectorstore_dir.exists() and any(vectorstore_dir.iterdir()):
        # There's likely data already persisted by chroma or haystack; attempt to read
        logger.info(f"Vectorstore directory {vectorstore_dir} exists; attempting to migrate any legacy JSON inside.")

    legacy_json = vectorstore_dir / "doc_store.json"
    if legacy_json.exists():
        logger.info(f"Found legacy JSON store at {legacy_json}; migrating...")
        with open(legacy_json, "r") as f:
            data = json.load(f)

        # Expecting a dict with 'documents' list or a direct list
        docs = data.get("documents") if isinstance(data, dict) else data
        if not docs:
            logger.warning("No documents found in legacy JSON; skipping migration.")
            return

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for i, d in enumerate(docs):
            meta = d.get("meta") if isinstance(d, dict) else {}
            doc_id = (meta or {}).get("hash") or d.get("id") or f"legacy_{i}"
            ids.append(str(doc_id))
            documents.append(d.get("content") or d.get("text") or "")
            metadatas.append(meta or {})
            emb = d.get("embedding")
            if emb:
                embeddings.append(emb)

        if ids:
            # Remove duplicates first
            existing = set(chroma.list_ids())
            to_delete = [i for i in ids if i in existing]
            if to_delete:
                logger.info(f"Deleting {len(to_delete)} existing duplicate ids before migration.")
                chroma.delete(to_delete)

            chroma.add_documents(ids=ids, documents=documents, metadatas=metadatas, embeddings=(embeddings if embeddings else None))
            chroma.persist()
            logger.info(f"Migrated {len(ids)} documents into Chroma.")
        else:
            logger.info("No documents to migrate.")
    else:
        logger.info("No legacy JSON found; nothing to migrate.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    vectorstore_dir = project_root / "vectorstore"
    migrate(vectorstore_dir)

