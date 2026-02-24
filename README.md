# Marketing RAG System (Haystack + Python)

This project provides a complete RAG (Retrieval-Augmented Generation) pipeline for processing marketing campaign documents in PDF and DOCX formats. It uses Haystack AI for ingestion, indexing, and retrieval.

## Features
- **Recursive Folder-based Ingestion**: Automatically processes files in `data/fichas_de_repertorio/` and its subdirectories, mapping them to clients based on folder names.
- **Anti-Duplication**: Uses a manifest file (`data/manifest.json`) to skip already processed files.
- **Support for PDF & DOCX**: Robust extraction and chunking.
- **Local Vector Store**: Persists embeddings locally for fast retrieval.
- **Open WebUI Integration**: FastAPI endpoint compatible with custom OpenAI-like providers.
- **Briefing Analysis Agent**: Specialized agent for marketing briefing summarization.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env` (use `.env.example` as a template).

## Usage

### 1. Ingestion
Place your PDF and DOCX files in `data/fichas_de_repertorio/`. You can organize them by client (e.g., `data/fichas_de_repertorio/ClientA/Project1/file.pdf`), and the system will automatically map the top-level folder name as the `client` metadata. Then run:
```bash
python -m src.ingestion.run
```
This will:
- Extract text.
- Create semantic chunks.
- Generate embeddings using `BAAI/bge-m3`.
- Store them in `vectorstore/doc_store.pkl`.

### 2. Querying (CLI)
To test the RAG system via terminal:
```bash
python -m src.rag.query "Your campaign question here"
```

### 3. Automatic Document Analysis
The system automatically classifies documents (Briefing vs. Edital) based on keywords in the filename.

To analyze a file locally:
```bash
python -m src.agents.auto_analyzer "path/to/briefing_xyz.pdf"
```

Keywords:
- **Briefing**: `briefing`, `brief`, `brf`
- **Edital**: `edital`, `tender`, `rfp`, `bid`, `procurement`

The summary will be saved to `data/summaries/` and displayed in the terminal.

### 4. Start API Service
To serve the RAG system and Document Analysis for Open WebUI:
```bash
python -m src.ui.server
```
*Note: This starts the main service on port 8000.*

### 5. Alternative Minimal API
If you only need basic RAG without document analysis, you can run:
```bash
python -m src.ui.api
```
*Note: This starts on port 8001 by default to avoid conflicts.*

## Open WebUI Integration

To connect Open WebUI to this service, follow the detailed guide in [OPENWEBUI.md](OPENWEBUI.md).

Quick start:
1. Run `python -m src.ui.server`.
2. Connect to `http://localhost:8000/v1` as an OpenAI provider.

- If the Briefing Agent times out, check your `OPENAI_API_KEY` and connection.

## Diagnostic Logging System (NEW)

The system includes a comprehensive tracing layer to debug file analysis and RAG triggers.

- **Trace Log**: Detailed decisions are recorded in `logs/file_analysis_trace.log`.
- **Event Types**: Monitors `REQUEST_ENTRY`, `DECISION_POINT`, `ANALYSIS_START`, and `CACHE_HIT`.
- **Stack Traces**: Each event includes a partial call stack and the current session state to identify why an analysis was triggered.

Usage via code:
```python
from src.utils.diagnostics import AnalysisDiagnostics
summary = AnalysisDiagnostics.get_summary() # Returns aggregated stats
```

## Session & Context Persistence (IMPROVED)

The API now supports **session-based conversation state** so that analyzer outputs survive across requests and are automatically injected into subsequent RAG queries.

### How it works
1. **Multi-layer Resolution**: The server identifies the conversation using `chat_id`, `session_id`, or by **hashing the first user message** (stable fingerprinting).
2. **State Management**: Analysis outputs are stored in the session and automatically injected into subsequent RAG queries via `query_with_context()`.
3. **Document Stability**: Documents are tracked via `file_id` (from UI) or content hash, ensuring they aren't re-analyzed even if the transmission format changes.

### API contract

**Request** (optional `session_id`):
```json
{
  "model": "marketing-rag-agent",
  "session_id": "optional-session-id-123",
  "messages": [
    { "role": "user", "content": "What campaigns match this briefing?" }
  ]
}
```

**Response** (always includes `session_id`):
```json
{
  "id": "chatcmpl-...",
  "session_id": "session-id-456",
  "choices": [ ... ]
}
```

Clients that do not send `session_id` will receive a new one. The server also supports extracting `session_id` from system-message metadata as a fallback.

### Configuration

Add these to your `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SESSION_BACKEND` | `memory` | `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL (only when backend=redis) |
| `SESSION_TTL_SECONDS` | `3600` | Session time-to-live in seconds |
| `ENABLE_REDACTION` | `false` | Enable PII redaction before session storage |

### Migration notes
- **Backwards compatible**: Existing clients that do not send `session_id` continue to work. They will simply get a new session per request.
- **New response field**: All `/v1/chat/completions` responses now include a top-level `session_id` field.
- **No database required**: Default in-memory backend works out of the box. For production, configure Redis.

## Project Structure
- `src/agents/`: Specialized agents (e.g., `briefing_analyzer.py`, `unified_analyzer.py`).
- `src/config/`: Configuration management.
- `src/ingestion/`: Text extraction and chunking.
- `src/indexing/`: Vector store and embedding logic.
- `src/rag/`: Retrieval and generation pipelines (including `query_with_context()`).
- `src/ui/`: FastAPI integration with session support and robust identifier resolution.
- `src/utils/`: Hashing, diagnostics, session management, and manifest management.
- `logs/file_analysis_trace.log`: Diagnostic trace logs for debugging RAG flow.
- `data/summaries/`: Output storage for the Briefing Analysis Agent.
- `data/fichas_de_repertorio/`: Default folder for bulk ingestion.
