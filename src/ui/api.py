import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag.pipeline import RAGPipeline
from src.utils.logging_config import setup_logging
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

load_dotenv()
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "vectorstore/")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

logger = setup_logging()
app = FastAPI(title="Haystack RAG API", description="API for Open WebUI integration")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# Global RAG instance
rag_instance = None

def get_rag():
    global rag_instance
    if rag_instance is None:
        try:
            # Initialize ChromaDocumentStore
            document_store = ChromaDocumentStore(persist_path=str(VECTOR_STORE_PATH), collection_name="documents")
            rag_instance = RAGPipeline(model_name=EMBEDDING_MODEL, document_store=document_store)
            logger.info("RAG instance initialized successfully with ChromaDocumentStore.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG instance: {e}")
            return None
    return rag_instance

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    rag = get_rag()
    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized. Check server logs.")
    
    try:
        logger.info(f"Received query: {request.question}")
        result = rag.query(request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "rag_loaded": get_rag() is not None}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001")) # Changed from 8000 to avoid conflict
    uvicorn.run(app, host=host, port=port)
