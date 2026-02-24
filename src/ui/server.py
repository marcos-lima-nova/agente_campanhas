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
from src.agents.conversation_orchestrator import ConversationOrchestrator
from src.utils.logging_config import setup_logging, bind_session_id
from src.utils.session_manager import SessionManager, AnalysisEntry
from src.utils.diagnostics import AnalysisDiagnostics
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

# The new default orchestrated model — routes through ConversationOrchestrator
ORCHESTRATED_MODELS = ["marketing-campaign-agent"]

# Legacy analyzer models — kept for backward compatibility (these don't use RAG)
ANALYZER_MODELS = ["gpt-4o-mini-analyzer", "unified-analyzer-agent"]

# --- OpenAI-compatible models (Minimal for Model Listing) ---
class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "marketing-agent"

# --- Global Instances (Lazy Loaded) ---

class AppState:
    """Lazy-loaded singleton container for heavyweight components."""

    rag: Optional[RAGPipeline] = None
    orchestrator: Optional[DocumentOrchestrator] = None
    conversation_orchestrator: Optional[ConversationOrchestrator] = None
    document_store: Optional[ChromaDocumentStore] = None
    session_manager: Optional[SessionManager] = None

state = AppState()

def get_rag():
    """Lazy-load and return the RAG pipeline singleton."""
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
    """Lazy-load and return the legacy DocumentOrchestrator singleton."""
    if state.orchestrator is None:
        state.orchestrator = DocumentOrchestrator()
    return state.orchestrator

def get_session_manager():
    """Lazy-load and return the SessionManager singleton."""
    if state.session_manager is None:
        state.session_manager = SessionManager()
        logger.info("SessionManager initialized.")
    return state.session_manager

def get_conversation_orchestrator():
    """Lazy-load and return the ConversationOrchestrator singleton.

    This is the single user-facing entrypoint for the ``marketing-campaign-agent``
    model.  It is built on top of the existing DocumentOrchestrator, RAGPipeline,
    and SessionManager, all of which are initialised lazily.
    """
    if state.conversation_orchestrator is None:
        state.conversation_orchestrator = ConversationOrchestrator(
            document_orchestrator=get_orchestrator(),
            rag_pipeline=get_rag(),
            session_manager=get_session_manager(),
        )
        logger.info("ConversationOrchestrator initialized.")
    return state.conversation_orchestrator

# --- Utilities ---

def extract_session_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Fallback: extract a session_id from message-level metadata.
    Looks for a system message containing a JSON block with ``session_id``.
    """
    import re as _re
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content_text = get_content_text(msg.get("content", ""))
        # Try JSON in the system message
        try:
            parsed = json.loads(content_text)
            if isinstance(parsed, dict) and "session_id" in parsed:
                return parsed["session_id"]
        except (json.JSONDecodeError, TypeError):
            pass
        # Try inline key=value pattern
        match = _re.search(r'session_id["\s:=]+([a-zA-Z0-9_-]+)', content_text)
        if match:
            return match.group(1)

    # 3. Ultimate Fallback: Hash the first user message content
    # This provides a stable ID for conversational threads in UIs like Open WebUI
    # that preserve message history but don't send a top-level chat_id.
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if user_msgs:
        first_content = get_content_text(user_msgs[0].get("content", ""))
        if first_content:
            import hashlib as _hashlib
            # Combine content with a prefix to avoid collisions with doc IDs
            h = _hashlib.sha256(f"session:{first_content}".encode("utf-8", errors="ignore")).hexdigest()
            return f"msg-{h[:16]}"

    return None

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
    orchestrator: DocumentOrchestrator = Depends(get_orchestrator),
    sm: SessionManager = Depends(get_session_manager),
    conv_orchestrator: ConversationOrchestrator = Depends(get_conversation_orchestrator),
):
    """
    OpenAI-compatible chat completions endpoint.

    Routing
    -------
    - ``marketing-campaign-agent`` (default/new): routes through
      ``ConversationOrchestrator``, which auto-detects intent and calls the
      appropriate sub-agent(s).  Users never need to choose an agent manually.
    - ``marketing-rag-agent``: legacy path — direct RAG query (backward compat).
    - ``gpt-4o-mini-analyzer`` / ``unified-analyzer-agent``: legacy paths —
      explicit document analysis (backward compat).

    All paths support ``session_id`` in the request body (or in system-message
    metadata) to persist context across turns.
    """
    # 1. Flexible Request Parsing
    try:
        body = await raw_request.json()
        # Default to the new orchestrated model so users need not specify one
        model_id = body.get("model", "marketing-campaign-agent")
        is_orchestrated = model_id in ORCHESTRATED_MODELS
        is_analyzer = model_id in ANALYZER_MODELS
        logger.info(f"--- NEW REQUEST ---")
        logger.info(f"Model: {model_id} (orchestrated={is_orchestrated})")

        # LOG ENTIRE BODY FOR INSPECTION (Debug only)
        logger.debug(f"BODY KEYS: {list(body.keys())}")
        logger.debug(f"RAW BODY: {json.dumps(body, indent=2)}")

        messages = body.get("messages", [])

        # Search for files everywhere in the request body
        files = find_files_recursively(body)

        # Extract the last user message text (needed for logging)
        user_msg_obj = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        user_msg = get_content_text(user_msg_obj.get("content")) if user_msg_obj else ""

        # Log entry for diagnostic tracing
        AnalysisDiagnostics.log_event(
            "REQUEST_ENTRY",
            f"Incoming chat completion request for model '{model_id}'",
            session_id=str(body.get("session_id") or body.get("chat_id") or "-"),
            extra_state={
                "model": model_id,
                "num_files": len(files),
                "all_body_keys": list(body.keys()),
                "headers": dict(raw_request.headers),
                "num_messages": len(messages),
                "first_msg_role": messages[0].get("role") if messages else None,
                "file_metadata": [
                    {"id": f.get("id"), "name": f.get("name") or f.get("filename")} 
                    for f in files
                ],
                "user_message_snippet": user_msg[:100]
            }
        )
        # --- Session ID resolution ---
        # Priority: chat_id (Open WebUI native) > session_id > system-message metadata > new session
        # Open WebUI sends 'chat_id' for each conversation which is consistent across all messages
        chat_id = body.get("chat_id")
        incoming_session_id = (
            chat_id or
            body.get("session_id") or
            extract_session_from_messages(messages)
        )
        session = sm.get_or_create_session(incoming_session_id)
        bind_session_id(logger, session.session_id)
        
        # Log the resolution result back to diagnostics
        AnalysisDiagnostics.log_event(
            "DECISION_POINT",
            f"Resolved session ID: {session.session_id}",
            session_id=session.session_id,
            extra_state={
                "incoming_id": incoming_session_id,
                "is_new": not incoming_session_id,
                "history_len": len(session.conversation_history)
            }
        )
        logger.info(f"Session: {session.session_id} (chat_id={chat_id}, session_id={body.get('session_id')}, incoming={incoming_session_id})")
        if incoming_session_id:
            logger.info(f"[SESSION_PRESERVED] Reusing existing session: {incoming_session_id}")
        else:
            logger.warning(f"[SESSION_NEW] No session_id in request - created new session: {session.session_id}")
        logger.debug(
            f"Session state: documents_keys={list(session.documents.keys())}, "
            f"active_document_id={session.active_document_id}, "
            f"last_analysis={'present' if session.last_analysis else 'None'}, "
            f"history_len={len(session.conversation_history)}"
        )

        # Search for files everywhere in the request body
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
        # (applies to legacy analyzer models AND the orchestrated model)
        if (is_analyzer or is_orchestrated) and not files:
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

    # ── 2a. NEW: Orchestrated path — marketing-campaign-agent ───────────────
    # The ConversationOrchestrator handles intent detection, routing, and
    # shared-state management internally.  We only need to prepare the
    # file descriptors (fetching bytes from Open WebUI when necessary) and
    # pass everything to handle_message().
    if is_orchestrated:
        # Extract the last user message text
        user_msg_obj = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        user_msg = get_content_text(user_msg_obj.get("content")) if user_msg_obj else ""

        if not user_msg and not files:
            raise HTTPException(status_code=400, detail="No user message found in request.")

        # Resolve file bytes/content for the orchestrator
        prepared_files: List[Dict[str, Any]] = []
        for file_info in files:
            filename = file_info.get("name") or file_info.get("filename", "document")
            file_id = file_info.get("id")

            # Virtual file (content already embedded via <source> tags)
            if file_info.get("content"):
                prepared_files.append({
                    "name": filename,
                    "id": file_id,
                    "content": file_info["content"],
                })
                logger.debug(f"Prepared virtual file '{filename}' with content len={len(file_info['content'])}")
                continue

            # Real attachment — fetch bytes from Open WebUI
            if file_id:
                file_bytes = fetch_file_from_webui(file_id)
                if file_bytes:
                    prepared_files.append({
                        "name": filename,
                        "id": file_id,
                        "bytes": file_bytes,
                    })
                    logger.debug(f"Prepared file '{filename}' with bytes len={len(file_bytes)}")
                else:
                    logger.warning(f"Could not fetch file '{filename}' (id={file_id}) from Open WebUI.")

        try:
            response_obj = conv_orchestrator.handle_message(
                message=user_msg,
                session_id=session.session_id,
                files=prepared_files,
            )
        except Exception as exc:
            logger.error(f"ConversationOrchestrator.handle_message failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

        # Build OpenAI-compatible response
        content = response_obj.content or "Não foi possível processar a solicitação."
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "session_id": response_obj.session_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ── 2b. LEGACY: Handle Attachments (File Analysis Flow) ─────────────────
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
                else:
                    # Persist structured output into session
                    sm.append_extracted_context(
                        session.session_id,
                        AnalysisEntry(
                            filename=result.get("filename", "unified_document_analysis.md"),
                            doc_type=result.get("doc_type", "unified"),
                            markdown=result["markdown"],
                            timestamp=result.get("timestamp", time.time()),
                            analyzer_id=result.get("analyzer_id", "unified-analyzer"),
                        ),
                    )
                
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
                        
                        # Persist structured output into session
                        sm.append_extracted_context(
                            session.session_id,
                            AnalysisEntry(
                                filename=result.get("filename", filename),
                                doc_type=result.get("doc_type", "unknown"),
                                markdown=result["markdown"],
                                timestamp=result.get("timestamp", time.time()),
                                analyzer_id=result.get("analyzer_id", model_id),
                            ),
                        )
                            
                        combined_markdown += f"\n\n---\n### Análise de {filename}\n\n" + result["markdown"]
                    except Exception as e:
                        logger.error(f"Failed to analyze {filename}: {e}")
                        combined_markdown += f"\n\n**Erro ao analisar {filename}:** {str(e)}\n"

        if combined_markdown:
            # Store assistant response in session history
            sm.append_message(session.session_id, "assistant", combined_markdown.strip())
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "session_id": session.session_id,
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
            "session_id": session.session_id,
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

    # Store user message in session
    sm.append_message(session.session_id, "user", user_msg)

    try:
        # Use context-aware RAG when session has prior analysis data
        current_session = sm.get_session(session.session_id)
        conversation_history = current_session.conversation_history if current_session else []
        extracted_context = current_session.extracted_context if current_session else []

        if extracted_context or len(conversation_history) > 1:
            logger.info(f"Using context-aware RAG (history={len(conversation_history)}, context={len(extracted_context)})")
            result = rag.query_with_context(
                question=user_msg,
                conversation_history=conversation_history,
                extracted_context=extracted_context,
            )
        else:
            logger.info(f"Executing standard RAG query: {user_msg}")
            result = rag.query(user_msg)
        
        answer = result["answer"]
        sources = result["sources"]
        
        if sources:
            source_list = "\n\n**Fontes:**\n" + "\n".join([f"- {s.get('filename', 'Unknown')}" for s in sources])
            answer += source_list

        # Store assistant response in session
        sm.append_message(session.session_id, "assistant", answer)

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "session_id": session.session_id,
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
    """Returns a list of available models for Open WebUI.

    ``marketing-campaign-agent`` is the recommended default.  It auto-detects
    intent and routes to the correct sub-agent without user intervention.
    The other three models are kept for backward compatibility.
    """
    return {
        "object": "list",
        "data": [
            OpenAIModel(id="marketing-campaign-agent"),   # new: orchestrated default
            OpenAIModel(id="marketing-rag-agent"),         # legacy: RAG-only
            OpenAIModel(id="gpt-4o-mini-analyzer"),        # legacy: analysis-only
            OpenAIModel(id="unified-analyzer-agent"),      # legacy: batch analysis
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
