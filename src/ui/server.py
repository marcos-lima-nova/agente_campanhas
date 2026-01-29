import os
import sys
import time
import uuid
import requests
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

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

logger = setup_logging("api")

app = FastAPI(
    title="Marketing Agent API", 
    description="Backend for Open WebUI integration with Automatic Document Analysis and Attachment Support"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Analyzer focus models (these don't use RAG)
ANALYZER_MODELS = ["gpt-4o-mini-analyzer", "unified-analyzer-agent"]

# --- OpenAI-compatible models (Minimal for Model Listing) ---
class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "marketing-agent"

# --- Global Instances (Lazy Loaded) ---

class AppState:
    rag: Optional[RAGPipeline] = None
    orchestrator: Optional[DocumentOrchestrator] = None
    document_store: Optional[ChromaDocumentStore] = None

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
        if response.status_code != 200:
            logger.error(f"Failed to fetch file {file_id} from Open WebUI. Status: {response.status_code}, Response: {response.text[:200]}")
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Request error fetching file {file_id} from Open WebUI: {e}")
        return None

def get_content_text(content: Any) -> str:
    """Robust helper to extract text from any OpenAI message content format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle multi-modal content list [ {"type": "text", "text": "..."}, ... ]
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif "text" in item: # Fallback
                    texts.append(item.get("text", ""))
        return " ".join(texts)
    if isinstance(content, dict):
        return content.get("text", "")
    return ""

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

def extract_from_source_tags(text: str) -> List[Dict[str, Any]]:
    """
    Parses <source name="..." id="..."> content from a string.
    Returns a list of virtual file objects with 'content', 'name', and 'id'.
    """
    import re
    # Pattern to match <source id="1" name="Briefing.pdf">content</source>
    # We use non-greedy matching for content and allow for quotes and spacing
    pattern = r'<source\s+id=["\']?(\d+)["\']?\s+name=["\']?([^"\'>]+)["\']?>(.*?)</source>'
    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
    
    found = []
    for m in matches:
        found.append({
            "id": m.group(1),
            "name": m.group(2),
            "content": m.group(3).strip(),
            "type": "virtual" # Indicator for pre-extracted content
        })
    return found

def find_files_recursively(data: Any) -> List[Dict[str, Any]]:
    """Recursively searches for file-like objects in the request body."""
    found_files = []
    if isinstance(data, dict):
        # Check standard keys
        for key in ["files", "attachments", "attached_files"]:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict) and ("id" in item or "file_id" in item):
                        found_files.append(item)
        
        # Check for single file object
        if "id" in data and "name" in data and ("meta" in data or "size" in data):
             # This looks like a file object itself
             found_files.append(data)
             
        # Recurse
        for value in data.values():
            found_files.extend(find_files_recursively(value))
            
    elif isinstance(data, list):
        for item in data:
            found_files.extend(find_files_recursively(item))
            
    return found_files

@app.post("/v1/chat/completions")
async def chat_completions(
    raw_request: Request, 
    rag: RAGPipeline = Depends(get_rag),
    orchestrator: DocumentOrchestrator = Depends(get_orchestrator)
):
    """
    OpenAI-compatible chat completions endpoint.
    Handles both regular RAG queries and automatic file analysis for attachments.
    """
    # 1. Flexible Request Parsing
    try:
        body = await raw_request.json()
        model_id = body.get("model", "marketing-rag-agent")
        is_analyzer = model_id in ANALYZER_MODELS
        logger.info(f"--- NEW REQUEST ---")
        logger.info(f"Model: {model_id}")
        
        # LOG ENTIRE BODY FOR INSPECTION (Debug only)
        logger.debug(f"BODY KEYS: {list(body.keys())}")
        logger.debug(f"RAW BODY: {json.dumps(body, indent=2)}")
        
        messages = body.get("messages", [])
        
        # Search for files everywhere
        files = find_files_recursively(body)
        
        if files:
            # Deduplicate by ID
            unique_files = {}
            for f in files:
                fid = f.get("id") or f.get("file_id")
                if fid:
                    unique_files[fid] = f
            files = list(unique_files.values())
            logger.info(f"Resolved {len(files)} unique files from request body.")
            for f in files:
                logger.info(f"File found: {f.get('name') or f.get('filename')} (ID: {f.get('id') or f.get('file_id')})")
        
        # Fallback: Search for <source> tags in system/user messages
        if is_analyzer and not files:
            logger.info("No attached files found. Searching for <source> tags in messages...")
            for msg in messages:
                content_text = get_content_text(msg.get("content", ""))
                if "<source" in content_text.lower():
                    source_count = content_text.lower().count("<source")
                    logger.info(f"Found {source_count} possible <source> tags in message {messages.index(msg)}. Running regex...")
                    extracted = extract_from_source_tags(content_text)
                    if extracted:
                        logger.info(f"Extracted {len(extracted)} virtual files from <source> tags.")
                        files.extend(extracted)
            
            # Deduplicate virtual files too
            if files:
                unique_files = {}
                for f in files:
                    fid = f.get("id") or f.get("file_id")
                    if fid:
                        unique_files[fid] = f
                files = list(unique_files.values())

        if not files:
            logger.info("No files found in body or message context.")
        
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 2. Handle Attachments (File Analysis Flow)
    if files:
        logger.info(f"Detected {len(files)} attachments for model {model_id}.")
        logger.debug(f"Files metadata: {files}") # Log full metadata for debugging
        
        # Branch based on model type
        if model_id == "unified-analyzer-agent":
            # Unified batch analysis
            try:
                # Prepare file list for orchestrator
                batch_files = []
                for file_info in files:
                    filename = file_info.get("name") or file_info.get("filename")
                    file_id = file_info.get("id")
                    if not file_id or not filename:
                        continue
                    
                    extracted_content = file_info.get("content")
                    if extracted_content:
                         batch_files.append({"content": extracted_content, "name": filename, "id": file_id})
                    else:
                        file_bytes = fetch_file_from_webui(file_id)
                        if file_bytes:
                            batch_files.append({"bytes": file_bytes, "name": filename, "id": file_id})
                
                if not batch_files:
                    logger.warning(f"Unified analyzer: No files could be fetched for model {model_id}.")
                    raise HTTPException(status_code=400, detail="Certifique-se de que os arquivos foram enviados corretamente e o backend tem acesso à API do Open WebUI.")
                
                logger.info(f"Triggering unified analysis for {len(batch_files)} files.")
                result = orchestrator.analyze_unified(batch_files)
                combined_markdown = result["markdown"]
                
                # If unified also returns the warning, ensure it's the ONLY thing returned
                if result.get("doc_type") == "invalid":
                    logger.warning("Unified analysis aborted due to non-compliant file in batch.")
                    combined_markdown = result["markdown"]
                
            except Exception as e:
                logger.error(f"Unified analysis failed: {e}")
                combined_markdown = f"**Erro na análise unificada:** {str(e)}"
        
        else:
            # Individual sequential analysis (Default or gpt-4o-mini-analyzer)
            combined_markdown = ""
            for file_info in files:
                filename = file_info.get("name") or file_info.get("filename")
                file_id = file_info.get("id")
                
                if not file_id or not filename:
                    logger.warning(f"Skipping incomplete file metadata: {file_info}")
                    continue
                    
                logger.info(f"Processing (model: {model_id}): {filename} (ID: {file_id})")
                
                # Check if content is already available (virtual file)
                extracted_content = file_info.get("content")
                file_bytes = None
                
                if not extracted_content:
                    file_bytes = fetch_file_from_webui(file_id)
                
                if not extracted_content and not file_bytes:
                    combined_markdown += f"\n\n**Erro:** Não foi possível acessar o arquivo '{filename}'.\n"
                    continue
                    
                try:
                    result = orchestrator.analyze_document(file_bytes=file_bytes, filename=filename, content=extracted_content)
                    
                    if result.get("doc_type") == "invalid":
                        logger.warning(f"Returning ONLY warning for non-compliant file: {filename}")
                        combined_markdown = result["markdown"]
                        break # Abort all other files
                        
                    combined_markdown += f"\n\n---\n### Análise de {filename}\n\n" + result["markdown"]
                except Exception as e:
                    logger.error(f"Failed to analyze {filename}: {e}")
                    combined_markdown += f"\n\n**Erro ao analisar {filename}:** {str(e)}\n"

        if combined_markdown:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
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

    # 3. Handle Analyzer Fallback (if no files or no markdown produced)
    if is_analyzer:
        error_msg = "Este agente requer um arquivo anexado (Briefing ou Edital) para realizar a análise."
        if not files:
            logger.warning(f"Analyzer model {model_id} called without files.")
        else:
            logger.warning(f"Analyzer model {model_id} produced no output.")
            error_msg = "Não foi possível extrair dados dos arquivos enviados ou a análise falhou."
            
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": error_msg}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 4. Regular RAG Query Flow
    # Extract the last user message
    user_msg_obj = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    user_msg = get_content_text(user_msg_obj.get("content")) if user_msg_obj else None
    
    if not user_msg:
        # Check if files were present but no text was found (only happens if combined_markdown was empty)
        if files:
            raise HTTPException(status_code=500, detail="Document analysis produced no results.")
        raise HTTPException(status_code=400, detail="No user message found in request.")

    try:
        logger.info(f"Executing RAG query: {user_msg}")
        result = rag.query(user_msg)
        
        answer = result["answer"]
        sources = result["sources"]
        
        if sources:
            source_list = "\n\n**Fontes:**\n" + "\n".join([f"- {s.get('filename', 'Unknown')}" for s in sources])
            answer += source_list

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
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
        logger.error(f"RAG execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """Returns a list of available models for Open WebUI."""
    return {
        "object": "list",
        "data": [
            OpenAIModel(id="marketing-rag-agent"),
            OpenAIModel(id="gpt-4o-mini-analyzer"),
            OpenAIModel(id="unified-analyzer-agent")
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
