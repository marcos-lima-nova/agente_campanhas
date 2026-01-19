import os
from pathlib import Path
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from typing import List, Optional
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

class IndexingPipeline:
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

    def run(self, documents):
        return self.pipeline.run({"embedder": {"documents": documents}})

    def get_document_store(self):
        return self.document_store
