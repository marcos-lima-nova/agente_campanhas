# Marketing RAG System (Haystack + Python)

This project provides a complete RAG (Retrieval-Augmented Generation) pipeline for processing marketing campaign documents in PDF and DOCX formats. It uses Haystack AI for ingestion, indexing, and retrieval.

## Features
- **Folder-based Ingestion**: Automatically processes files in `data/inbox/`.
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
Place your PDF and DOCX files in `data/inbox/`. Then run:
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
To serve the RAG system for Open WebUI:
```bash
uvicorn src.ui.api:app --host 0.0.0.0 --port 8000
```

## Open WebUI Integration

To connect Open WebUI to this service, follow the detailed guide in [OPENWEBUI.md](OPENWEBUI.md).

Quick start:
1. Run `python -m src.ui.server`.
2. Connect to `http://localhost:8000/v1` as an OpenAI provider.

### Troubleshooting
- If no files are found, check the `INBOX_FOLDER` path in `.env`.
- If memory is an issue, reduce `CHUNK_SIZE`.
- The first run will download the embedding model (`BAAI/bge-m3`).
- If the Briefing Agent times out, check your `OPENAI_API_KEY` and connection.

## Project Structure
- `src/agents/`: Specialized agents (e.g., `briefing_analyzer.py`).
- `src/config/`: Configuration management.
- `src/ingestion/`: Text extraction and chunking.
- `src/indexing/`: Vector store and embedding logic.
- `src/rag/`: Retrieval and generation pipelines.
- `src/ui/`: FastAPI integration.
- `src/utils/`: Hashing, logging, and manifest management.
- `data/summaries/`: Output storage for the Briefing Analysis Agent.
- `data/inbox/`: Default folder for bulk ingestion.
