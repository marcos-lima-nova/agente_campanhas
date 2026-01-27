import os
from pathlib import Path
from typing import List, Dict, Any
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, model_name: str = "BAAI/bge-m3", document_store: ChromaDocumentStore = None):
        # 1. Setup Document Store
        if document_store is None:
            vector_store_path = Path(os.getenv("VECTOR_STORE_PATH", "vectorstore/")).resolve()
            self.document_store = ChromaDocumentStore(persist_path=str(vector_store_path), collection_name="documents")
            logger.info(f"Initialized ChromaDocumentStore for RAG from {vector_store_path}")
        else:
            self.document_store = document_store
        
        # 2. Setup Prompt Template
        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

        # 3. Build Pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=model_name))
        self.pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=self.document_store, top_k=5))
        self.pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["documents", "question"]))
        
        # Using OpenAIGenerator for compatibility
        self.pipeline.add_component("llm", OpenAIGenerator(model="gpt-4.1-nano")) 

        # 4. Connect Components
        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "llm")

    def _get_unique_sources(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Helper to de-duplicate sources by filename."""
        unique_sources = []
        seen_filenames = set()
        
        for doc in documents:
            filename = doc.meta.get("filename")
            if filename and filename not in seen_filenames:
                unique_sources.append(doc.meta)
                seen_filenames.add(filename)
            elif not filename:
                # Fallback if filename is missing (should not happen with current ingestion)
                unique_sources.append(doc.meta)
        
        return unique_sources

    def query(self, question: str):
        try:
            logger.info(f"Executing RAG pipeline for question: {question}")
            result = self.pipeline.run({
                "embedder": {"text": question},
                "prompt_builder": {"question": question}
            }, include_outputs_from={"retriever"})
            
            # Formatting response
            answer = result["llm"]["replies"][0]
            
            # Extract unique sources only from the retrieved documents
            retrieved_docs = result.get("retriever", {}).get("documents", [])
            sources = self._get_unique_sources(retrieved_docs)
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"RAG pipeline query failed: {str(e)}", exc_info=True)
            return {"answer": f"Error fulfilling request: {str(e)}", "sources": []}
