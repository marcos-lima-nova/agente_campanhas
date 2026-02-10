---
title: Implementation Plan — Resolve Context Loss and Orchestration Gaps
date: 2026-02-10
author: Kilo Code (generated)
repo_path: agente_campanhas
---

Executive summary

- Implement session-based conversation state to persist analyzer outputs across requests.
- Introduce `SessionManager` in `src/utils/session_manager.py` with an in-memory default and optional Redis backend.
- Modify `src/ui/server.py` to accept/return `session_id`, persist analysis outputs into session, and inject session context into RAG queries.
- Extend `src/agents/orchestrator.py` to return structured analysis metadata consumable by the session manager.
- Add `query_with_context(question, conversation_history, extracted_context)` to `src/rag/pipeline.py` and a context-aware PromptBuilder template.
- Add TTL, redaction options, and observability improvements (session_id in logs).
- Validate with unit tests and integration tests simulating analyze → RAG continuity using session_id.
- Rollout in phases: prerequisites → MVP (sessions) → reliability (Redis/chroma indexing) → hardening & observability.

Full plan details and tasks are documented in the repository docs and correspond to changes in:
- `src/ui/server.py` (endpoint routing, session integration)
- `src/agents/orchestrator.py` (structured returns)
- `src/rag/pipeline.py` (context-aware RAG)
- `src/utils/session_manager.py` (new)
- `src/utils/redaction.py` (optional)
- Logging config changes in `src/utils/logging_config.py`

See `docs/technical_gap_analysis.md` and `docs/orchestration_explanation.md` for background and code references.

End of file

