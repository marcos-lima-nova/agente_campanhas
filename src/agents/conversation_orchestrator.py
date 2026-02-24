"""
ConversationOrchestrator
========================
Single user-facing entrypoint for the marketing agent system.

Responsibilities
----------------
1. **Intent detection** — decide whether to run the Analysis Agent, the Q&A
   Agent, or both (MIXED).
2. **Routing** — delegate to the appropriate sub-agent(s):
   - Analysis path → ``DocumentOrchestrator``
   - Q&A path      → ``RAGPipeline.query_with_context()``
3. **Shared state management** — read/write the ``Session`` (conversation
   history, analysis outputs, document metadata, routing log) via
   ``SessionManager``.
4. **Mixed-intent chaining** — when both intents are detected, the analysis
   agent runs first, its output is stored in the shared state, and *then* the
   Q&A agent runs against the updated context.
5. **Structured response** — always returns an ``OrchestratorResponse`` that
   the server layer can turn into an OpenAI-compatible JSON body.

Usage::

    orchestrator = ConversationOrchestrator(
        document_orchestrator=DocumentOrchestrator(),
        rag_pipeline=RAGPipeline(...),
        session_manager=SessionManager(),
    )
    response = orchestrator.handle_message(
        message="analyze this briefing and suggest similar campaigns",
        session_id="abc-123",
        files=[{"name": "briefing.pdf", "id": "file-1", "bytes": b"..."}],
    )
    print(response.content)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.intent_detector import Intent, IntentDetector, IntentResult
from src.agents.orchestrator import DocumentOrchestrator
from src.rag.pipeline import RAGPipeline
from src.utils.logging_config import setup_logging
from src.utils.session_manager import AnalysisEntry, Session, SessionManager
from src.utils.diagnostics import AnalysisDiagnostics

logger = setup_logging("conversation_orchestrator")


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResponse:
    """The unified response returned by ``ConversationOrchestrator.handle_message()``.

    Attributes
    ----------
    content         : Final text to present to the user.  For MIXED intents this
                      combines the analysis markdown and the QA answer.
    intent          : The intent that was detected for this turn.
    routing_log     : Ordered list of agent names that were invoked.
    session_id      : Session identifier — the client should echo this on the
                      next request to preserve conversation context.
    analysis_output : Populated when the Analysis Agent ran; contains at minimum
                      ``filename``, ``doc_type``, ``markdown``, ``timestamp``,
                      ``analyzer_id``, and ``keywords``.
    qa_answer       : Populated when the Q&A Agent ran.
    sources         : Source documents cited by the Q&A Agent (if any).
    error           : Non-empty when an agent failed (partial results may still
                      be present in ``content``).
    """

    content: str
    intent: Intent
    routing_log: List[str] = field(default_factory=list)
    session_id: str = ""
    analysis_output: Optional[Dict[str, Any]] = None
    qa_answer: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ConversationOrchestrator
# ---------------------------------------------------------------------------

class ConversationOrchestrator:
    """Single user-facing orchestrator that routes to Analysis and/or QA agents.

    Parameters
    ----------
    document_orchestrator : Existing ``DocumentOrchestrator`` for document
                            classification and analysis.
    rag_pipeline          : Existing ``RAGPipeline`` for Q&A queries.
    session_manager       : ``SessionManager`` for persistent conversation state.
    """

    def __init__(
        self,
        document_orchestrator: DocumentOrchestrator,
        rag_pipeline: RAGPipeline,
        session_manager: SessionManager,
    ) -> None:
        self._doc_orchestrator = document_orchestrator
        self._rag = rag_pipeline
        self._session_manager = session_manager
        self._intent_detector = IntentDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> OrchestratorResponse:
        """Process one user turn end-to-end.

        Parameters
        ----------
        message    : The user's latest message text.
        session_id : Optional existing session identifier.  When omitted a new
                     session is created.
        files      : List of file descriptors.  Each entry must have at least
                     ``name`` (str) and either ``bytes`` (raw bytes) or
                     ``content`` (pre-extracted text).  Optional ``id`` field
                     is stored in document metadata.

        Returns
        -------
        ``OrchestratorResponse`` with the combined content, intent, routing
        log, and updated ``session_id``.
        """
        files = files or []
        routing_log: List[str] = []

        # ── 1. Resolve session ───────────────────────────────────────────
        session = self._session_manager.get_or_create_session(session_id)
        sid = session.session_id
        logger.info(f"[session={sid}] handle_message called (files={len(files)})")

        # Store the user turn in conversation history
        self._session_manager.append_message(sid, "user", message)

        # ── 1b. Compute document IDs and filter re-sent files ─────────────
        # Open WebUI re-sends file references on EVERY request.
        # We compute a SHA-256 document_id for each file and skip those
        # already registered in session.documents (the document cache).
        logger.debug(
            f"[session={sid}] BEFORE _get_new_files: "
            f"session.documents keys={list(session.documents.keys())}, "
            f"active_document_id={session.active_document_id}, "
            f"last_analysis={'present' if session.last_analysis else 'None'}"
        )
        new_files = self._get_new_files(files, session)
        logger.debug(
            f"[session={sid}] AFTER _get_new_files: "
            f"input_files={len(files)}, new_files={len(new_files)}, "
            f"new_file_ids={[f.get('_document_id') for f in new_files]}"
        )
        if len(files) != len(new_files):
            logger.info(
                f"[session={sid}] Filtered {len(files) - len(new_files)} "
                f"already-analyzed file(s).  New files to process: {len(new_files)}"
            )

        # ── 2. Detect intent ─────────────────────────────────────────────
        # Only truly new/unanalyzed files count as "attachments".
        # Re-sent files from a prior turn are invisible to the intent detector.
        intent_result: IntentResult = self._intent_detector.detect_intent(
            message=message,
            has_attachments=bool(files),
            state=session,
        )
        intent = intent_result.intent
        
        AnalysisDiagnostics.log_event(
            "DECISION_POINT",
            f"Intent detected: {intent} (confidence={intent_result.confidence})",
            session_id=sid,
            extra_state={
                "intent": str(intent),
                "confidence": intent_result.confidence,
                "reasoning": intent_result.reasoning,
                "is_reanalysis_request": intent_result.is_reanalysis
            }
        )
        logger.info(f"[session={sid}] Intent={intent.value}, reasoning='{intent_result.reasoning}'")

        # ── 3. Route ─────────────────────────────────────────────────────
        analysis_output: Optional[Dict[str, Any]] = None
        qa_answer: Optional[str] = None
        sources: List[Dict[str, Any]] = []
        error: Optional[str] = None
        response_parts: List[str] = []

        # --- Analysis path (ANALYSIS or MIXED) ---
        # Executed when:
        # 1. Genuinely NEW files are attached.
        # 2. User explicitly asked for re-analysis (even if files were already seen).
        files_to_analyze = new_files
        if intent_result.is_reanalysis and not new_files and session.active_document_id:
            # Force re-analysis of the active document if it exists and no new files were sent
            active_doc = session.get_document(session.active_document_id)
            if active_doc:
                # We need the original bytes/content, which the cache might not have 
                # if it only stores the analysis_result. 
                # Wait, session.documents stores "analysis_result". 
                # If we don't have the original bytes, we can't "re-analyze" unless 
                # the UI re-sent them.
                pass 

        # Re-evaluating: If Open WebUI re-sends files, they are in `files` but filtered out of `new_files`.
        # If is_reanalysis is true, we should use all `files` as candidates.
        if intent_result.is_reanalysis and files:
            files_to_analyze = [{**f, "_document_id": _compute_document_id(f.get("content") or f.get("bytes") or b"", f.get("name", ""))} for f in files]

        if intent in (Intent.ANALYSIS, Intent.MIXED) and files_to_analyze:
            routing_log.append("DocumentOrchestrator")
            for file_info in files_to_analyze:
                filename = file_info.get("name") or file_info.get("filename", "document")
                file_bytes: Optional[bytes] = file_info.get("bytes")
                content: Optional[str] = file_info.get("content")
                file_id: Optional[str] = file_info.get("id")
                document_id: str = file_info.get("_document_id") or _compute_document_id(
                    content or (file_bytes or b""), filename
                )

                if not file_bytes and not content:
                    logger.warning(f"[session={sid}] Skipping '{filename}': no bytes or content provided.")
                    continue

                try:
                    AnalysisDiagnostics.log_event(
                        "ANALYSIS_START",
                        f"Initiating analysis for '{filename}'",
                        session_id=sid,
                        extra_state={"filename": filename, "doc_id": document_id}
                    )
                    logger.info(f"[session={sid}] Running DocumentOrchestrator on '{filename}' (doc_id={document_id})")
                    raw_result = self._doc_orchestrator.analyze_document(
                        file_bytes=file_bytes,
                        filename=filename,
                        content=content,
                    )

                    if raw_result.get("doc_type") == "invalid":
                        logger.warning(f"[session={sid}] Non-compliant filename: '{filename}'")
                        response_parts.append(raw_result["markdown"])
                        continue

                    # Wrap into structured output and extract keywords
                    structured = self._build_structured_output(raw_result, file_id=file_id)
                    analysis_output = structured

                    # Persist analysis entry (for RAG context pipeline)
                    entry = AnalysisEntry(
                        filename=structured["filename"],
                        doc_type=structured["doc_type"],
                        markdown=structured["raw_markdown"],
                        timestamp=structured["timestamp"],
                        analyzer_id=structured["analyzer_id"],
                    )
                    self._session_manager.append_extracted_context(sid, entry)

                    # Register document in the cache keyed by document_id
                    session = self._session_manager.get_session(sid)  # re-fetch
                    if session:
                        session.register_document(
                            document_id=document_id,
                            filename=filename,
                            file_id=file_id,
                            doc_type=structured["doc_type"],
                            content_hash=document_id,
                            analysis_result=structured,
                        )
                        self._session_manager.save_session(session)

                    # Include analysis in response ONLY if it's the primary intent or a new file
                    # If it's a MIXED intent follow-up where the file was already analyzed,
                    # we usually don't want to see the analysis again unless requested.
                    is_new_file = any(f.get("_document_id") == document_id for f in new_files)
                    if is_new_file or intent == Intent.ANALYSIS or intent_result.is_reanalysis:
                        response_parts.append(
                            f"### Análise de {filename}\n\n{structured['raw_markdown']}"
                        )

                except Exception as exc:
                    logger.error(f"[session={sid}] Analysis failed for '{filename}': {exc}", exc_info=True)
                    response_parts.append(f"**Erro ao analisar '{filename}':** {exc}")
                    error = str(exc)

        # --- Q&A path (QA or MIXED, or ANALYSIS intent with no new files) ---
        # When intent is ANALYSIS but new_files is empty (re-sent file), fall
        # through to QA so the cached analysis is used.
        should_run_qa = (
            intent in (Intent.QA, Intent.MIXED)
            or (intent == Intent.ANALYSIS and not files_to_analyze)
        )
        if should_run_qa:
            if not files_to_analyze and session.active_document_id:
                AnalysisDiagnostics.log_event(
                    "CACHE_HIT",
                    f"Using cached analysis for active document (doc_id={session.active_document_id})",
                    session_id=sid,
                    extra_state={"active_doc_id": session.active_document_id}
                )
            routing_log.append("RAGPipeline")
            try:
                fresh_session = self._session_manager.get_session(sid)
                conversation_history = fresh_session.conversation_history if fresh_session else []
                extracted_context = fresh_session.extracted_context if fresh_session else []

                logger.info(
                    f"[session={sid}] Running RAGPipeline "
                    f"(history={len(conversation_history)}, context={len(extracted_context)})"
                )

                rag_result = self._rag.query_with_context(
                    question=message,
                    conversation_history=conversation_history,
                    extracted_context=extracted_context,
                )
                qa_answer = rag_result.get("answer", "")
                sources = rag_result.get("sources", [])
                response_parts.append(qa_answer)

            except Exception as exc:
                logger.error(f"[session={sid}] Q&A failed: {exc}", exc_info=True)
                response_parts.append(f"**Erro no agente Q&A:** {exc}")
                error = str(exc)

        # ── 4. Build combined response ────────────────────────────────────
        separator = "\n\n---\n\n"
        combined_content = separator.join(p for p in response_parts if p)

        # ── 5. Persist assistant turn ─────────────────────────────────────
        # Store only a brief reference for analysis turns — NOT the full markdown.
        # The full analysis is preserved in extracted_context and session.documents.
        # Storing it in conversation_history would cause the RAG LLM to restate it.
        if analysis_output:
            doc_name = analysis_output.get("filename", "documento")
            doc_type_str = analysis_output.get("doc_type", "")
            history_msg = (
                f"[Análise de '{doc_name}' ({doc_type_str}) concluída e "
                f"armazenada. Q&A subsequente usará o resultado em cache.]"
            )
            if qa_answer:
                history_msg += f"\n\n{qa_answer}"
            self._session_manager.append_message(sid, "assistant", history_msg)
        else:
            self._session_manager.append_message(sid, "assistant", combined_content)

        # ── 6. Log routing event ──────────────────────────────────────────
        fresh_session = self._session_manager.get_session(sid)
        if fresh_session:
            fresh_session.append_routing_event(
                intent=intent.value,
                agents_called=routing_log,
                message_preview=message,
            )
            self._session_manager.save_session(fresh_session)

        logger.info(
            f"[session={sid}] Response built "
            f"(intent={intent.value}, agents={routing_log}, chars={len(combined_content)})"
        )

        return OrchestratorResponse(
            content=combined_content,
            intent=intent,
            routing_log=routing_log,
            session_id=sid,
            analysis_output=analysis_output,
            qa_answer=qa_answer,
            sources=sources,
            error=error,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_new_files(
        self,
        files: List[Dict[str, Any]],
        session: Any,
    ) -> List[Dict[str, Any]]:
        """Return only files whose document_id is NOT already in session.documents.

        A ``_document_id`` field is injected into each new-file descriptor so
        the analysis path can use it without re-computing the hash.
        Document identity is based on SHA-256 of content/bytes + filename.
        """
        if not files:
            return []

        new_files: List[Dict[str, Any]] = []
        for f in files:
            file_id = f.get("id") or f.get("file_id")
            file_bytes = f.get("bytes") or b""
            content = f.get("content") or ""
            filename = f.get("name") or f.get("filename", "document")
            
            # Priority: Use file_id for stability across turns if available
            document_id = _compute_document_id(content or file_bytes, filename, file_id=file_id)

            logger.debug(
                f"_get_new_files: file='{filename}', "
                f"has_bytes={bool(file_bytes)}, has_content={bool(content)}, "
                f"computed_doc_id={document_id}, "
                f"is_in_cache={session.is_document_analyzed(document_id)}"
            )

            if session.is_document_analyzed(document_id):
                AnalysisDiagnostics.log_event(
                    "DECISION_POINT",
                    f"File '{filename}' already analyzed (doc_id={document_id})",
                    session_id=getattr(session, "session_id", "-"),
                    extra_state={"filename": filename, "doc_id": document_id, "status": "CACHED"}
                )
                logger.info(
                    f"Skipping already-analyzed file '{filename}' (doc_id={document_id})"
                )
                continue

            # Inject the pre-computed document_id for the analysis path
            AnalysisDiagnostics.log_event(
                "DECISION_POINT",
                f"File '{filename}' is new (doc_id={document_id})",
                session_id=getattr(session, "session_id", "-"),
                extra_state={"filename": filename, "doc_id": document_id, "status": "NEW"}
            )
            new_files.append({**f, "_document_id": document_id})

        return new_files

    @staticmethod
    def _build_structured_output(
        raw_result: Dict[str, Any],
        file_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wrap a ``DocumentOrchestrator`` result into a standardised dict.

        The structured output preserves the full markdown while adding typed
        metadata and a best-effort keyword list extracted from the LLM output.

        Parameters
        ----------
        raw_result : Dict returned by ``DocumentOrchestrator.analyze_document()``.
        file_id    : Optional file identifier from the upload platform.

        Returns
        -------
        Dict with keys: summary, raw_markdown, filename, doc_type, timestamp,
        analyzer_id, keywords, metadata.
        """
        markdown: str = raw_result.get("markdown", "")
        keywords = _extract_keywords_from_markdown(markdown)

        return {
            "summary": markdown,                             # alias — same content
            "raw_markdown": markdown,
            "filename": raw_result.get("filename", "analysis.md"),
            "doc_type": raw_result.get("doc_type", "unknown"),
            "timestamp": raw_result.get("timestamp", time.time()),
            "analyzer_id": raw_result.get("analyzer_id", "unknown"),
            "keywords": keywords,
            "metadata": {
                "file_id": file_id,
                "original_filename": raw_result.get("filename", ""),
            },
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_keywords_from_markdown(markdown: str) -> List[str]:
    """Extract keywords from an analysis markdown string.

    Looks for a section headed with words like "keywords", "tags", "palavras-chave",
    and collects bullet/dash-list items below it.

    Returns a list of up to 30 keyword strings.  Returns an empty list when no
    keyword section is found.
    """
    import re

    # Match a line that looks like a "keywords / tags" section header
    header_re = re.compile(
        r"^#{1,4}\s*.*(keywords?|tags?|palavras[- ]chave|indicadores).*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Bullet or dash list items (possibly preceded by whitespace)
    item_re = re.compile(r"^\s*[-*•]\s+(.+)$", re.MULTILINE)

    keywords: List[str] = []
    header_match = header_re.search(markdown)

    if not header_match:
        return keywords

    # Take the text that follows the header
    text_after_header = markdown[header_match.end():]

    # Stop at the next section header (line starting with #)
    next_header = re.search(r"^#{1,6}\s", text_after_header, re.MULTILINE)
    if next_header:
        text_after_header = text_after_header[: next_header.start()]

    for m in item_re.finditer(text_after_header):
        kw = m.group(1).strip(" `*_")
        if kw:
            keywords.append(kw)
        if len(keywords) >= 30:
            break

    return keywords


def _compute_document_id(
    content: Union[str, bytes],
    filename: str = "",
    length: int = 16,
    file_id: Optional[str] = None
) -> str:
    """Compute a document_id based on a hash of the content and metadata.
    If file_id is provided (e.g. from Open WebUI), it is used as the primary
    identifier for stability across turns.
    """
    if file_id:
        return hashlib.sha256(file_id.encode()).hexdigest()[:length]

    if isinstance(content, str):
        content_bytes = content.encode("utf-8", errors="ignore")
    else:
        content_bytes = content
    prefix = filename.encode("utf-8", errors="ignore")
    digest = hashlib.sha256(prefix + b":" + content_bytes).hexdigest()
    return digest[:length]
