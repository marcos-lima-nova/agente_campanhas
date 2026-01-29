import os
from pathlib import Path
from typing import List, Dict, Any
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from src.utils.llm_factory import get_llm_generator
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

        ### Instructions:
        - MANDATORY: For every piece of information you extract from the context, you MUST cite the source using the format `[SOURCE_N]` where N is the index provided in the context header.
        - Only cite a source if it truly contributed to your answer.

        ### Context:
        {% for document in documents %}
        --- [SOURCE_{{ loop.index0 }}] ---
        Document: {{ document.meta.filename }}
        Content: {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

        # 3. Build Pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=model_name))
        self.pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=self.document_store, top_k=5))
        self.pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["documents", "question"]))
        
        # Initialize LLM via Factory
        self.pipeline.add_component("llm", get_llm_generator(model_name="gpt-4o-mini")) 

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
            
            # 1. Extract used tags from answer
            import re
            used_indices = set(re.findall(r"\[SOURCE_(\d+)\]", answer))
            used_indices = {int(idx) for idx in used_indices}
            
            # 2. Map indices back to retrieved documents
            retrieved_docs = result.get("retriever", {}).get("documents", [])
            actively_used_docs = []
            for i, doc in enumerate(retrieved_docs):
                if i in used_indices:
                    actively_used_docs.append(doc)
            
            # 3. Extract unique sources only from the actively used documents
            sources = self._get_unique_sources(actively_used_docs)
            
            # Log results for audit
            logger.info(f"Retrieved {len(retrieved_docs)} docs. Actively used {len(used_indices)} docs based on LLM citations.")
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"RAG pipeline query failed: {str(e)}", exc_info=True)
            return {"answer": f"Error fulfilling request: {str(e)}", "sources": []}
