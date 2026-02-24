import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from src.utils.llm_factory import get_llm_generator
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from src.utils.logging_config import setup_logging
from src.utils.diagnostics import AnalysisDiagnostics

logger = setup_logging("rag")

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

        # 5. Context-aware pipeline (used by query_with_context)
        context_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        ### Instructions:
        - MANDATORY: For every piece of information you extract from the context, you MUST cite the source using the format `[SOURCE_N]` where N is the index provided in the context header.
        - Only cite a source if it truly contributed to your answer.
        - When prior analysis context is provided, use it as background reference to inform your answer. Do NOT repeat, restate, or reproduce the prior analysis text. Answer the question directly and concisely.
        - Focus exclusively on answering the user's current question.

        {% if conversation_history %}
        ### Conversation History:
        {% for msg in conversation_history %}
        {{ msg.role }}: {{ msg.content }}
        {% endfor %}
        {% endif %}

        {% if extracted_context %}
        ### Prior Analysis Context (reference only — do NOT reproduce this content):
        {% for ctx in extracted_context %}
        --- Analysis: {{ ctx.filename }} ({{ ctx.doc_type }}) ---
        {{ ctx.markdown }}
        {% endfor %}
        {% endif %}

        ### Retrieved Documents:
        {% for document in documents %}
        --- [SOURCE_{{ loop.index0 }}] ---
        Document: {{ document.meta.filename }}
        Content: {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

        self.context_pipeline = Pipeline()
        self.context_pipeline.add_component(
            "embedder",
            SentenceTransformersTextEmbedder(model=model_name),
        )
        self.context_pipeline.add_component(
            "retriever",
            ChromaEmbeddingRetriever(document_store=self.document_store, top_k=5),
        )
        self.context_pipeline.add_component(
            "prompt_builder",
            PromptBuilder(
                template=context_template,
                required_variables=["documents", "question"],
            ),
        )
        self.context_pipeline.add_component("llm", get_llm_generator(model_name="gpt-4o-mini"))
        self.context_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.context_pipeline.connect("retriever", "prompt_builder.documents")
        self.context_pipeline.connect("prompt_builder", "llm")

    # -- class-level access to the context template for testing -----------
    CONTEXT_TEMPLATE = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        ### Instructions:
        - MANDATORY: For every piece of information you extract from the context, you MUST cite the source using the format `[SOURCE_N]` where N is the index provided in the context header.
        - Only cite a source if it truly contributed to your answer.
        - When prior analysis context is provided, use it as background reference to inform your answer. Do NOT repeat, restate, or reproduce the prior analysis text. Answer the question directly and concisely.
        - Focus exclusively on answering the user's current question.

        {% if conversation_history %}
        ### Conversation History:
        {% for msg in conversation_history %}
        {{ msg.role }}: {{ msg.content }}
        {% endfor %}
        {% endif %}

        {% if extracted_context %}
        ### Prior Analysis Context (reference only — do NOT reproduce this content):
        {% for ctx in extracted_context %}
        --- Analysis: {{ ctx.filename }} ({{ ctx.doc_type }}) ---
        {{ ctx.markdown }}
        {% endfor %}
        {% endif %}

        ### Retrieved Documents:
        {% for document in documents %}
        --- [SOURCE_{{ loop.index0 }}] ---
        Document: {{ document.meta.filename }}
        Content: {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

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

    def query_with_context(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        extracted_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Context-aware RAG query that includes conversation history and
        prior analyzer outputs in the prompt.

        Falls back to the standard ``query()`` when no extra context is
        provided.
        """
        if not conversation_history and not extracted_context:
            logger.info("No extra context provided, falling back to standard query.")
            return self.query(question)

        AnalysisDiagnostics.log_event(
            "DECISION_POINT",
            f"RAG query with context (num_context_items={len(extracted_context or [])})",
            extra_state={
                "num_history_msgs": len(conversation_history or []),
                "num_context_items": len(extracted_context or []),
                "context_filenames": [c.get("filename") for c in (extracted_context or [])]
            }
        )

        # 1. Build Analysis Context (Briefings/Editais)
        try:
            logger.info(
                f"Executing context-aware RAG query: {question} "
                f"(history={len(conversation_history or [])}, "
                f"context_entries={len(extracted_context or [])})"
            )

            prompt_vars: Dict[str, Any] = {"question": question}
            if conversation_history:
                prompt_vars["conversation_history"] = conversation_history
            if extracted_context:
                prompt_vars["extracted_context"] = extracted_context

            result = self.context_pipeline.run(
                {
                    "embedder": {"text": question},
                    "prompt_builder": prompt_vars,
                },
                include_outputs_from={"retriever"},
            )

            answer = result["llm"]["replies"][0]

            import re
            used_indices = {int(idx) for idx in re.findall(r"\[SOURCE_(\d+)\]", answer)}
            retrieved_docs = result.get("retriever", {}).get("documents", [])
            actively_used_docs = [
                doc for i, doc in enumerate(retrieved_docs) if i in used_indices
            ]
            sources = self._get_unique_sources(actively_used_docs)

            logger.info(
                f"Context-aware RAG: retrieved {len(retrieved_docs)} docs, "
                f"actively used {len(used_indices)} based on citations."
            )

            return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.error(f"Context-aware RAG query failed: {str(e)}", exc_info=True)
            return {"answer": f"Error fulfilling request: {str(e)}", "sources": []}
