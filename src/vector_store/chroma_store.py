import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb

logger = logging.getLogger(__name__)


class ChromaAdapter:
    """Adapter that wraps a ChromaDB collection and exposes a minimal vector-store API.

    Methods provided:
    - add_documents(ids, documents, metadatas, embeddings)
    - query_by_embedding(embedding, top_k)
    - persist()
    - load()
    - delete(ids)
    - list_ids()
    - migrate_from_json(path)  # basic legacy migration support
    """

    def __init__(self, persist_directory: Path, collection_name: str = "documents"):
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name

        # Use the new Chroma client construction (no deprecated Settings)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        """Add documents to the Chroma collection. Embeddings are optional (if not provided, Chroma will try to compute them if configured).
        ids: list of unique ids
        documents: list of text contents
        metadatas: list of metadata dicts
        embeddings: list of vector lists (optional)
        """
        kwargs = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self.collection.add(**kwargs)

    def query_by_embedding(self, embedding: List[float], top_k: int = 5):
        """Query the collection by an embedding vector. Returns a dict with ids, documents, metadatas, distances."""
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k, include=["ids", "documents", "metadatas", "distances"]) 
        if not result or len(result.get("ids", [])) == 0:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        return {
            "ids": result["ids"][0] if "ids" in result else [],
            "documents": result["documents"][0] if "documents" in result else [],
            "metadatas": result["metadatas"][0] if "metadatas" in result else [],
            "distances": result["distances"][0] if "distances" in result else [],
        }

    def persist(self):
        try:
            self.client.persist()
        except Exception as e:
            logger.warning(f"Chroma persist() failed: {e}")

    def load(self):
        # Re-get the collection (useful after restart)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def delete(self, ids: Optional[List[str]] = None):
        if ids is None:
            self.collection.delete()  # deletes everything
        else:
            self.collection.delete(ids=ids)

    def list_ids(self) -> List[str]:
        try:
            data = self.collection.get(include=["ids"])
            return data.get("ids", [])
        except Exception:
            return []

    def migrate_from_json(self, json_path: Path, id_prefix: str = "legacy"):
        """Attempt to migrate a legacy JSON-based store into Chroma.

        This function performs a best-effort mapping of common legacy dump shapes into Chroma inputs.
        It's idempotent if called with the same ids (Chroma will error on duplicates unless overwritten by delete first).
        """
        import json
        from uuid import uuid4

        if not Path(json_path).exists():
            logger.info("No legacy json store found for migration.")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        # Common haystack-like shape: {"documents": [{"content": ..., "meta": {...}, "embedding": [...]}, ...]}
        documents = []
        metadatas = []
        ids = []
        embeddings = []

        docs = data.get("documents") if isinstance(data, dict) else None
        if not docs:
            logger.warning("Legacy JSON format not recognized for migration; skipping.")
            return

        for i, d in enumerate(docs):
            doc_id = d.get("meta", {}).get("hash") or d.get("id") or f"{id_prefix}_{i}_{uuid4().hex}"
            ids.append(str(doc_id))
            documents.append(d.get("content") or d.get("text") or "")
            metadatas.append(d.get("meta") or {})
            emb = d.get("embedding")
            if emb:
                embeddings.append(emb)

        if documents:
            # If duplicates exist, drop existing ids first to avoid errors (idempotent behavior)
            try:
                existing = set(self.list_ids())
                to_delete = [i for i in ids if i in existing]
                if to_delete:
                    self.delete(to_delete)
            except Exception:
                pass

            self.add_documents(ids=ids, documents=documents, metadatas=metadatas, embeddings=(embeddings if embeddings else None))
            self.persist()
            logger.info(f"Migrated {len(documents)} documents into Chroma collection '{self.collection_name}'.")
