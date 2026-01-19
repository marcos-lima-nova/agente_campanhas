import os
import sys
from pathlib import Path
import logging

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack import Pipeline
from src.vector_store.chroma_store import ChromaAdapter
import json

load_dotenv()
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "vectorstore/"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_document_store():
    logger.info("Starting debug test...")

    # Load document store from directory (expected behavior) — prefer Chroma
    chroma = None
    if VECTOR_STORE_PATH.exists() and any(VECTOR_STORE_PATH.iterdir()):
        try:
            chroma = ChromaAdapter(persist_directory=VECTOR_STORE_PATH)
            logger.info("Initialized Chroma adapter for debug tests.")
        except Exception as e:
            logger.warning(f"Chroma adapter init failed: {e}")

        try:
            document_store = InMemoryDocumentStore.load_from_disk(str(VECTOR_STORE_PATH))
            logger.info("Loaded document store from directory.")
        except Exception as e:
            logger.error(f"Failed to load document store from directory: {e}")
            # If both chroma and document store fail, abort
            if chroma is None:
                return
    else:
        doc_store_file = VECTOR_STORE_PATH / "doc_store.json"
        if not doc_store_file.exists():
            logger.error("Document store not found. Run the ingestion to create vectorstore/ directory.")
            return
        try:
            with open(doc_store_file, "r") as f:
                data = json.load(f)
            document_store = InMemoryDocumentStore.from_dict(data)
            logger.info("Loaded document store from json file.")
        except Exception as e:
            logger.error(f"Failed to load document store from json: {e}")
            return

    # Check documents (if using haystack docstore)
    documents = []
    if 'document_store' in locals() and hasattr(document_store, 'storage'):
        documents = list(document_store.storage.values())
    elif chroma:
        # Try to list docs from chroma collection
        try:
            data = chroma.collection.get(include=["documents", "metadatas", "ids"]) if chroma else {}
            docs_from_chroma = data.get("documents", [])
            metas_from_chroma = data.get("metadatas", [])
            documents = []
            for d, m in zip(docs_from_chroma, metas_from_chroma):
                doc = Document(content=d, meta=m)
                documents.append(doc)
        except Exception:
            documents = []
    logger.info(f"Total documents in store: {len(documents)}")
    for i, doc in enumerate(documents):
        logger.info(f"Document {i}: content length {len(doc.content)}, has embedding: {hasattr(doc, 'embedding') and doc.embedding is not None}")
        if i > 4:  # Limit log
            break

    # Initialize embedder
    try:
        embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
        logger.info("Embedder initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        return

    # Test embedding generation
    if documents:
        sample_doc = list(documents)[0]
        logger.info("Testing embedding generation on sample document...")
        try:
            embedded_docs = embedder.run(documents=[sample_doc])
            logger.info(f"Embedding generated: {len(embedded_docs['documents'][0].embedding)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return

    # Initialize retriever
    try:
        retriever = InMemoryEmbeddingRetriever(document_store=document_store)
        logger.info("Retriever initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        return

    # Test retrieval
    if documents and hasattr(list(documents)[0], 'embedding') and list(documents)[0].embedding:
        query_embedding = list(documents)[0].embedding  # Use existing embedding for test
        logger.info("Testing retrieval with existing embedding...")
        try:
            results = retriever.run(query_embedding=query_embedding)
            logger.info(f"Retrieval returned {len(results['documents'])} documents.")
        except Exception as e:
            logger.error(f"Failed retrieval: {e}")
    else:
        logger.warning("No documents with embeddings for retrieval test.")

    # Simulate full query pipeline
    logger.info("Simulating full query pipeline...")
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

    try:
        pipeline = Pipeline()
        pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL))
        pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
        pipeline.add_component("prompt_builder", PromptBuilder(template=template))
        pipeline.add_component("llm", HuggingFaceLocalGenerator(model="google/flan-t5-small", task="text2text-generation", generation_kwargs={"max_new_tokens": 200}))

        pipeline.connect("embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")

        result = pipeline.run({
            "embedder": {"text": "What is the main topic?"},
            "prompt_builder": {"question": "What is the main topic?"}
        })

        logger.info(f"Pipeline result: answer - {result['llm']['replies'][0]}, sources - {len(result['retriever']['documents'])}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    debug_document_store()
