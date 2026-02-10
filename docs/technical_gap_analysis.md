---
title: Technical Gap Analysis — Context Loss & Orchestration
date: 2026-02-10
author: Kilo Code (generated)
repo_path: agente_campanhas
---

Executive summary
-----------------
This analysis identifies and prioritizes technical gaps that cause context loss between analyzers (`gpt-4o-mini-analyzer`, `unified-analyzer-agent`) and the RAG agent (`marketing-rag-agent`). It provides evidence from the codebase, remediation recommendations, estimated effort, and a sequenced implementation plan for engineers.

Gaps (prioritized)
-------------------

1) Missing Session / Conversation State (Critical)
-------------------------------------------------
- Description: No session or conversation state is persisted after analyzer runs. Analyzer outputs are returned to the client but not stored server-side for later RAG queries.
- Impact/Severity: High — leads to loss of analysis-derived context between consecutive calls; poor UX and incorrect retrieval relevance.
- Evidence: `AppState` lacks session storage (`src/ui/server.py:67-72`), RAG uses only last user message (`src/ui/server.py:388-392`).
- Remediation:
  - Implement a session manager (in-memory or Redis-backed) at `src/utils/session_manager.py` that tracks conversation history and accumulated analyzer outputs (see suggested data model).
  - Modify `server.py` to accept and return a `session_id` in request/response metadata and persist analyzer outputs into session state when analyzers run.
  - Modify `RAGPipeline` to accept `conversation_history` and `extracted_context` and include them in the PromptBuilder template.
- Effort: Medium (3-5 days / 13-20 story points).
- Risks: Memory growth if in-memory sessions are used; mitigate with TTL and optional Redis for production.
- Validation:
  - Integration test: run unified analysis then a RAG query with returned session_id, assert RAG answer uses analysis content.

2) Stateless RAG queries (High)
------------------------------
- Description: RAG pipeline template contains only retrieved documents and question; no slot for pre-analyzed context or conversation history. (`src/rag/pipeline.py:24-42`).
- Impact: Medium — RAG answers not informed by prior analyzer results.
- Remediation:
  - Add a context-aware prompt template and implement `query_with_context(question, conversation_history, extracted_context)`.
- Effort: Small (1-2 days).
- Validation: Unit test for prompt rendering when `conversation_history` and `extracted_context` are present.

3) Session ID transport & API contract (Medium)
---------------------------------------------
- Description: No explicit API contract exists to carry a session identifier between client and server. Existing Open WebUI clients may not send metadata.
- Impact: Medium — Without a standard, clients cannot reliably re-use session state.
- Remediation:
  - Define a `session_id` field in the request body and return it in responses (`src/ui/server.py` response envelope). Document in README.
  - Support session_id in system message metadata as fallback (server already parses messages). Implement `extract_session_from_messages()`.
- Effort: Small (1-2 days).
- Validation: Backwards compatibility tests and client integration test.

4) Payload integrity for analyzer outputs (Medium)
------------------------------------------------
- Description: Analyzer outputs are saved to disk (data/summaries/) but not centrally indexed or JSON-serialized in sessions. This makes programmatic retrieval for RAG less convenient.
- Impact: Medium — Harder to programmatically consume analyzer outputs in RAG prompts.
- Remediation:
  - Store analyzer outputs inside session state as structured entries (filename, doc_type, markdown, timestamp).
  - Optionally add lightweight indexing to Chroma or a document cache for quick retrieval.
- Effort: Small to Medium (2-4 days).
- Validation: Ensure session.extracted_context is included in RAG queries and present in prompt.

5) Security & privacy (Medium)
------------------------------
- Description: Storing analyzed documents in session state or disk may expose PII or sensitive content if not protected (no explicit encryption or access controls in code).
- Impact: High (compliance/legal risk) depending on deployment.
- Remediation:
  - Add configuration for session persistence backend (in-memory vs Redis) and ensure data-at-rest encryption for Redis or disk (if necessary).
  - Add retention policy (TTL) and redaction options for sensitive fields before storing.
- Effort: Medium (2-5 days) + infra.
- Validation: Security review and privacy testing with synthetic PII data.

6) Observability / Logging (Low)
-------------------------------
- Description: While logging exists, there is no structured tracing to link analyzer runs to subsequent RAG queries.
- Impact: Low to Medium — makes debugging cross-request flows harder.
- Remediation:
  - Add a correlation_id/session_id on logs for each request lifecycle (inject into logger's contextual info).
  - Emit structured JSON logs for analyzer outputs and RAG inputs.
- Effort: Small (1-2 days).
- Validation: Logging tests and end-to-end trace demonstration.

Implementation plan (sequence)
-----------------------------
1. Create `src/utils/session_manager.py` (in-memory with TTL and pluggable Redis backend). [Medium]
2. Update `src/ui/server.py` to accept `session_id` in request, use `SessionManager`, and return `session_id` in responses; persist analyzer outputs to session. [Medium]
3. Update `src/agents/orchestrator.py` to allow callers to pass a session object (or return structured metadata) and to return structured analysis results (doc_type, filename, markdown) suitable for session storage. [Small]
4. Update `src/rag/pipeline.py` to add `query_with_context(question, conversation_history, extracted_context)` and a context-aware PromptBuilder template. [Small]
5. Add unit and integration tests for the above flows and update README with API contract changes. [Small]

Proposed PR structure
---------------------
- PR 1: feature/session-manager
  - Add `src/utils/session_manager.py`
  - Unit tests for session lifecycle

- PR 2: feature/server-session-integration
  - Modify `src/ui/server.py` to integrate SessionManager, accept/return `session_id`, persist analyzer outputs
  - Update API docs

- PR 3: feature/orchestrator-return-structure
  - Modify `src/agents/orchestrator.py` to return structured analysis metadata
  - Minor adjustments in analyzers if necessary

- PR 4: feature/rag-context-aware
  - Modify `src/rag/pipeline.py` to add `query_with_context`
  - Tests for context inclusion in prompts

Testing & validation
--------------------
- Integration: unified analysis -> RAG query using same session_id; assert that RAG prompt contains `extracted_context` (mock or inspect prompt builder input).
- Unit: session manager add/get/expiry behaviors.
- Security: ensure TTL and optional redaction can be toggled in env.

Appendix: Key evidence snippets
------------------------------
- `AppState` no session: [`src/ui/server.py:67-72`](src/ui/server.py:67).
- Single-message extraction for RAG: [`src/ui/server.py:388-392`](src/ui/server.py:388).
- Orchestrator analyze_unified: [`src/agents/orchestrator.py:69-76`](src/agents/orchestrator.py:69).
- UnifiedAnalyzer pipeline: [`src/agents/unified_analyzer.py:80-88`](src/agents/unified_analyzer.py:80).

End of file

