import os
import sys
import time
import uuid
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.rag.pipeline import RAGPipeline
from src.agents.orchestrator import DocumentOrchestrator
from src.utils.logging_config import setup_logging
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

# Load config
load_dotenv()
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "vectorstore/")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Open WebUI Integration Config
OPEN_WEBUI_URL = os.getenv("OPEN_WEBUI_URL", "http://localhost:3000")
OPEN_WEBUI_API_KEY = os.getenv("OPEN_WEBUI_API_KEY")

logger = setup_logging()

app = FastAPI(
    title="Marketing Agent API", 
    description="Backend for Open WebUI integration with Automatic Document Analysis and Attachment Support"
)

# --- Models ---

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# OpenAI-compatible models
class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "marketing-agent"

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatCompletionRequest(BaseModel, extra="allow"):
    model: str
    messages: List[OpenAIMessage]
    stream: bool = False
    files: Optional[List[Dict[str, Any]]] = None # Open WebUI sends this

# --- Global Instances (Lazy Loaded) ---

class AppState:
    rag: Optional[RAGPipeline] = None
    orchestrator: Optional[DocumentOrchestrator] = None
    document_store: Optional[ChromaDocumentStore] = None
    combined_pipeline: Optional[Pipeline] = None

state = AppState()

def get_rag():
    if state.rag is None:
        try:
            if state.document_store is None:
                state.document_store = ChromaDocumentStore(persist_path=str(VECTOR_STORE_PATH), collection_name="documents")
            state.rag = RAGPipeline(model_name=EMBEDDING_MODEL, document_store=state.document_store)
            logger.info("RAG Pipeline initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise HTTPException(status_code=500, detail="RAG system initialization failed.")
    return state.rag

def get_orchestrator():
    if state.orchestrator is None:
        state.orchestrator = DocumentOrchestrator()
    return state.orchestrator

def get_combined_pipeline():
    if state.combined_pipeline is None:
        template = """
You are a senior marketing strategist. You have been provided with a Briefing Summary and an Edital (Bidding Document) Summary.
Your task is to merge these two analyses into a single, cohesive "Strategic Alignment Report".

Develop the document entirely in Portuguese-BR.

Identify:
1. Points of alignment (where the agency proposal meets the edital requirements).
2. Potential gaps or risks (requirements in the edital not clearly addressed in the briefing).
3. Recommended strategy to win the bid.

---
Briefing Summary:
{{briefing}}

---
Edital Summary:
{{edital}}
---
        """
        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", PromptBuilder(template=template))
        api_key = Secret.from_token(OPENAI_API_KEY) if OPENAI_API_KEY else None
        pipeline.add_component("llm", OpenAIGenerator(api_key=api_key, model="gpt-4o-mini"))
        pipeline.connect("prompt_builder", "llm")
        state.combined_pipeline = pipeline
    return state.combined_pipeline

# --- Utilities ---

def fetch_file_from_webui(file_id: str) -> Optional[bytes]:
    """Fetches file bytes from Open WebUI API using a file ID."""
    if not OPEN_WEBUI_API_KEY:
        logger.warning("OPEN_WEBUI_API_KEY not set. Cannot fetch attachments by ID.")
        return None
        
    url = f"{OPEN_WEBUI_URL.rstrip('/')}/api/v1/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {OPEN_WEBUI_API_KEY}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to fetch file {file_id} from Open WebUI: {e}")
        return None

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "Marketing Agent API is running. Attachment support enabled."}

@app.get("/v1")
async def v1_root():
    return {"message": "OpenAI-compatible API base. Use /v1/chat/completions for queries."}

@app.get("/health")
async def health():
    doc_count = 0
    try:
        if state.document_store is None:
            state.document_store = ChromaDocumentStore(persist_path=str(VECTOR_STORE_PATH), collection_name="documents")
        doc_count = state.document_store.count_documents()
    except Exception as e:
        logger.error(f"Health check store error: {e}")
    
    return {
        "status": "ok",
        "document_count": doc_count,
        "vector_store_path": str(VECTOR_STORE_PATH)
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: RAGPipeline = Depends(get_rag)):
    try:
        logger.info(f"Querying RAG: {request.message}")
        result = rag.query(request.message)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), 
    orchestrator: DocumentOrchestrator = Depends(get_orchestrator)
):
    try:
        logger.info(f"Received file for automatic analysis: {file.filename}")
        file_bytes = await file.read()
        result = orchestrator.analyze_document(file_bytes, file.filename, file.content_type)
        return JSONResponse(content=result)
    except ValueError as ve:
        logger.warning(f"Classification failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- OpenAI Compatibility Adapter with Attachment Logic ---

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            OpenAIModel(id="marketing-rag-agent"),
            OpenAIModel(id="gpt-4o-mini-analyzer")
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest, 
    rag: RAGPipeline = Depends(get_rag),
    orchestrator: DocumentOrchestrator = Depends(get_orchestrator)
):
    # 1. Check for Attachments (files field from Open WebUI)
    files = request.files or []
    
    # Also check if messages contain file references (sometimes Open WebUI sends them differently)
    # For now, we trust the 'files' field which is standard in Open WebUI's custom provider calls.
    
    if files:
        logger.info(f"Detected {len(files)} attachments in chat request.")
        combined_markdown = ""
        
        for file_info in files:
            filename = file_info.get("name") or file_info.get("filename")
            file_id = file_info.get("id")
            
            if not file_id or not filename:
                continue
                
            logger.info(f"Processing attachment: {filename} (ID: {file_id})")
            
            # Fetch bytes from Open WebUI
            file_bytes = fetch_file_from_webui(file_id)
            if not file_bytes:
                combined_markdown += f"\n\n**Erro:** Não foi possível acessar o arquivo '{filename}'. Verifique se o backend tem acesso à API do Open WebUI.\n"
                continue
                
            try:
                result = orchestrator.analyze_document(file_bytes, filename)
                combined_markdown += f"\n\n---\n### Análise de {filename}\n\n" + result["markdown"]
            except Exception as e:
                logger.error(f"Failed to analyze {filename} from chat: {e}")
                combined_markdown += f"\n\n**Erro ao analisar {filename}:** {str(e)}\n"

        if combined_markdown:
            response_id = f"chatcmpl-{uuid.uuid4()}"
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": combined_markdown.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    # 2. Regular RAG Query if no attachments or analysis failed
    user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found.")

    try:
        logger.info(f"OpenAI Adapter received query: {user_msg}")
        result = rag.query(user_msg)
        
        answer = result["answer"]
        sources = result["sources"]
        
        if sources:
            source_list = "\n\n**Fontes:**\n" + "\n".join([f"- {s.get('filename', 'Unknown')}" for s in sources])
            answer += source_list

        response_id = f"chatcmpl-{uuid.uuid4()}"
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        logger.error(f"OpenAI Adapter error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
