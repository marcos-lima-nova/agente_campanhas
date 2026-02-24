"""
Session Manager — in-memory (default) with optional Redis backend.

Provides conversation-state persistence so that analyzer outputs survive
across requests and can be injected into subsequent RAG queries.

Environment variables
---------------------
SESSION_BACKEND   : "memory" (default) | "redis"
REDIS_URL         : Redis connection URL (only when SESSION_BACKEND=redis)
SESSION_TTL_SECONDS : Session time-to-live in seconds (default 3600)
"""

from __future__ import annotations

import os
import time
import uuid
import json
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.utils.logging_config import setup_logging
from src.utils.redaction import redact_text

logger = setup_logging("session")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AnalysisEntry:
    """A single analyzer output stored in the session."""
    filename: str
    doc_type: str
    markdown: str
    timestamp: float = field(default_factory=time.time)
    analyzer_id: str = ""


@dataclass
class Session:
    """Represents a single conversation session with shared persistent state.

    Fields
    ------
    session_id          : Unique identifier for the session.
    created_at          : Unix timestamp when the session was created.
    last_accessed       : Unix timestamp of the most recent read/write.
    conversation_history: Ordered list of {role, content} message dicts.
    extracted_context   : List of AnalysisEntry dicts **kept for RAG compatibility**.
    documents           : Dict keyed by document_id (SHA256 snippet) mapping to document
                          metadata + cached analysis result.  Structure::

                            {
                              "<document_id>": {
                                "filename": str,
                                "file_id": str | None,
                                "doc_type": str,
                                "analyzed_at": float,
                                "content_hash": str,     # same as document_id
                                "analysis_result": {...}, # structured analysis dict
                              },
                              ...
                            }

    active_document_id  : The document_id of the document currently in focus.
                          Set after each analysis; used to look up cached results on
                          follow-up questions.
    last_analysis       : Convenience pointer to documents[active_document_id]["analysis_result"].
    routing_history     : Log of routing decisions.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    extracted_context: List[Dict[str, Any]] = field(default_factory=list)

    # -- document identity fields (backward-compatible: default to empty/None) --
    documents: Dict[str, Any] = field(default_factory=dict)
    active_document_id: Optional[str] = field(default=None)
    last_analysis: Optional[Dict[str, Any]] = field(default=None)
    routing_history: List[Dict[str, Any]] = field(default_factory=list)

    # -- helpers ----------------------------------------------------------
    def touch(self) -> None:
        """Update last_accessed timestamp."""
        self.last_accessed = time.time()

    def append_message(self, role: str, content: str) -> None:
        """Append a {role, content} message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        self.touch()

    def append_extracted_context(self, entry: AnalysisEntry) -> None:
        """Append an analysis entry and update last_analysis pointer."""
        entry_dict = asdict(entry)
        self.extracted_context.append(entry_dict)
        self.last_analysis = entry_dict  # keep a reference to the latest analysis
        self.touch()

    def register_document(
        self,
        document_id: str,
        filename: str,
        file_id: Optional[str],
        doc_type: str,
        content_hash: str,
        analysis_result: Dict[str, Any],
    ) -> None:
        """Cache a document and its analysis result under *document_id*.

        Also sets ``active_document_id`` and ``last_analysis``.
        """
        self.documents[document_id] = {
            "filename": filename,
            "file_id": file_id,
            "doc_type": doc_type,
            "analyzed_at": time.time(),
            "content_hash": content_hash,
            "analysis_result": analysis_result,
        }
        self.active_document_id = document_id
        self.last_analysis = analysis_result
        self.touch()
        logger.info(
            f"Session.register_document: doc_id={document_id}, "
            f"filename={filename}, active_document_id now={self.active_document_id}, "
            f"documents_keys={list(self.documents.keys())}"
        )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Return the cached document entry for *document_id*, or ``None``."""
        return self.documents.get(document_id)

    def is_document_analyzed(self, document_id: str) -> bool:
        """Return ``True`` if *document_id* is already in the document cache."""
        return document_id in self.documents

    def append_routing_event(self, intent: str, agents_called: List[str], message_preview: str) -> None:
        """Log a routing decision to the routing history."""
        self.routing_history.append({
            "timestamp": time.time(),
            "intent": intent,
            "agents_called": agents_called,
            "message_preview": message_preview[:120],
        })
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Reconstruct a Session from a dict (e.g., fetched from Redis).

        New fields default gracefully so sessions persisted before this
        upgrade remain loadable without errors.

        Note: ``documents`` changed from ``list`` to ``dict`` in this version.
        Old sessions stored as lists are silently converted to empty dicts.
        """
        raw_docs = data.get("documents", {})
        # Guard: convert old list format (from pre-upgrade sessions) to dict
        if isinstance(raw_docs, list):
            raw_docs = {}

        session = cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            conversation_history=data.get("conversation_history", []),
            extracted_context=data.get("extracted_context", []),
            documents=raw_docs,
            active_document_id=data.get("active_document_id"),
            last_analysis=data.get("last_analysis"),
            routing_history=data.get("routing_history", []),
        )
        return session


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class SessionBackend(ABC):
    """Abstract backend for session storage."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]:
        ...

    @abstractmethod
    def put(self, session: Session) -> None:
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    def cleanup_expired(self, ttl: int) -> int:
        """Remove sessions older than *ttl* seconds.  Return count removed."""
        ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class InMemoryBackend(SessionBackend):
    def __init__(self) -> None:
        self._store: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._store.get(session_id)
            if session is not None:
                session.touch()
            return session

    def put(self, session: Session) -> None:
        with self._lock:
            self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def cleanup_expired(self, ttl: int) -> int:
        cutoff = time.time() - ttl
        removed = 0
        with self._lock:
            expired = [
                sid for sid, s in self._store.items()
                if s.last_accessed < cutoff
            ]
            for sid in expired:
                del self._store[sid]
                removed += 1
        return removed


# ---------------------------------------------------------------------------
# Redis backend (optional — requires ``redis`` package)
# ---------------------------------------------------------------------------

class RedisBackend(SessionBackend):
    """Redis-backed session storage.  Requires the ``redis`` Python package."""

    def __init__(self, redis_url: str, ttl: int = 3600) -> None:
        try:
            import redis as _redis  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'redis' package is required for RedisBackend. "
                "Install it with: pip install redis"
            ) from exc

        self._client = _redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._prefix = "session:"

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def get(self, session_id: str) -> Optional[Session]:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        data = json.loads(raw)
        session = Session.from_dict(data)
        session.touch()
        # Refresh TTL on access
        self._client.setex(self._key(session_id), self._ttl, json.dumps(session.to_dict()))
        return session

    def put(self, session: Session) -> None:
        self._client.setex(
            self._key(session.session_id),
            self._ttl,
            json.dumps(session.to_dict()),
        )

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def cleanup_expired(self, ttl: int) -> int:
        # Redis handles expiry via SETEX; nothing to do here.
        return 0


# ---------------------------------------------------------------------------
# Session Manager (public API)
# ---------------------------------------------------------------------------

class SessionManager:
    """
    High-level session manager used by the server layer.

    Usage::

        sm = SessionManager()          # reads env vars
        session = sm.create_session()
        sm.get_session(session.session_id)
        sm.append_extracted_context(session.session_id, entry)
        sm.delete_session(session.session_id)
    """

    def __init__(
        self,
        backend: Optional[SessionBackend] = None,
        ttl: Optional[int] = None,
    ) -> None:
        self._ttl = ttl or int(os.getenv("SESSION_TTL_SECONDS", "3600"))

        if backend is not None:
            self._backend = backend
        else:
            backend_type = os.getenv("SESSION_BACKEND", "memory").lower()
            if backend_type == "redis":
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self._backend = RedisBackend(redis_url=redis_url, ttl=self._ttl)
                logger.info("SessionManager using Redis backend.")
            else:
                self._backend = InMemoryBackend()
                logger.info("SessionManager using in-memory backend.")

    # -- public API --------------------------------------------------------

    def create_session(self, session_id: Optional[str] = None) -> Session:
        """Create and persist a new session.  Optionally reuse a given id."""
        session = Session(session_id=session_id or str(uuid.uuid4()))
        self._backend.put(session)
        logger.info(f"Session created: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve an existing session or return ``None``."""
        return self._backend.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Return existing session or create a new one."""
        if session_id:
            session = self.get_session(session_id)
            if session is not None:
                return session
        return self.create_session(session_id)

    def save_session(self, session: Session) -> None:
        """Persist session state after mutations."""
        self._backend.put(session)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"append_message: session {session_id} not found.")
            return
        session.append_message(role, content)
        self._backend.put(session)

    def append_extracted_context(
        self, session_id: str, entry: AnalysisEntry
    ) -> None:
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"append_extracted_context: session {session_id} not found.")
            return
        # Apply redaction before persisting
        entry.markdown = redact_text(entry.markdown)
        session.append_extracted_context(entry)
        self._backend.put(session)

    def delete_session(self, session_id: str) -> None:
        self._backend.delete(session_id)
        logger.info(f"Session deleted: {session_id}")

    def cleanup_expired(self) -> int:
        removed = self._backend.cleanup_expired(self._ttl)
        if removed:
            logger.info(f"Cleaned up {removed} expired sessions.")
        return removed

    @property
    def ttl(self) -> int:
        return self._ttl
