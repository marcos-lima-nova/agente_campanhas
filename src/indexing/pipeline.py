import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Standard Haystack-based indexing pipeline (chunk-then-embed)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", document_store: Optional[ChromaDocumentStore] = None):
        if document_store is None:
            # Default to ChromaDocumentStore if not provided
            persist_path = os.getenv("VECTOR_STORE_PATH", "vectorstore")
            self.document_store = ChromaDocumentStore(persist_path=persist_path, collection_name="documents")
        else:
            self.document_store = document_store

        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=model_name))
        self.pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
        self.pipeline.connect("embedder", "writer")

    def run(self, documents: List[Document]) -> Dict:
        return self.pipeline.run({"embedder": {"documents": documents}})

    def get_document_store(self) -> ChromaDocumentStore:
        return self.document_store


class LateChunkingPipeline:
    """Indexing pipeline for documents that already carry pre-computed embeddings.

    When using :class:`~src.ingestion.processors.LateChunkerProcessor`, each
    ``Document`` already has its ``embedding`` set.  This pipeline skips the
    embedding step and writes directly to the document store.

    It is API-compatible with :class:`IndexingPipeline` (same ``run`` and
    ``get_document_store`` interface).

    Args:
        document_store: An existing ChromaDocumentStore.  If ``None``, a
            default persistent store is created.
    """

    def __init__(self, document_store: Optional[ChromaDocumentStore] = None) -> None:
        if document_store is None:
            persist_path = os.getenv("VECTOR_STORE_PATH", "vectorstore")
            self.document_store = ChromaDocumentStore(
                persist_path=persist_path, collection_name="documents"
            )
        else:
            self.document_store = document_store

        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "writer", DocumentWriter(document_store=self.document_store)
        )

    def run(self, documents: List[Document]) -> Dict:
        """Write documents (with pre-computed embeddings) to the store.

        Args:
            documents: Haystack ``Document`` objects whose ``embedding``
                fields are already populated.

        Returns:
            Dict with writer output.

        Raises:
            ValueError: If any document is missing its embedding.
        """
        missing = [
            i for i, d in enumerate(documents) if d.embedding is None
        ]
        if missing:
            raise ValueError(
                f"LateChunkingPipeline requires pre-computed embeddings. "
                f"Documents at indices {missing[:5]}… are missing embeddings."
            )

        logger.info(
            "LateChunkingPipeline: writing %d documents with pre-computed embeddings",
            len(documents),
        )
        return self.pipeline.run({"writer": {"documents": documents}})

    def get_document_store(self) -> ChromaDocumentStore:
        return self.document_store
