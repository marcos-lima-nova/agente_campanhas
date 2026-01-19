# Marketing RAG System (Haystack + Python)

This project provides a complete RAG (Retrieval-Augmented Generation) pipeline for processing marketing campaign documents in PDF and DOCX formats. It uses Haystack AI for ingestion, indexing, and retrieval.

## Features
- **Folder-based Ingestion**: Automatically processes files in `data/inbox/`.
- **Anti-Duplication**: Uses a manifest file (`data/manifest.json`) to skip already processed files.
- **Support for PDF & DOCX**: Robust extraction and chunking.
- **Local Vector Store**: Persists embeddings locally for fast retrieval.
- **Open WebUI Integration**: FastAPI endpoint compatible with custom OpenAI-like providers.

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

3. Configure environment variables in `.env` (or use the defaults).

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
To test the system via terminal:
```bash
python -m src.rag.query "Your campaign question here"
```

### 3. Start API Service
To serve the RAG system for Open WebUI:
```bash
uvicorn src.ui.api:app --host 0.0.0.0 --port 8000
```

## Open WebUI Integration

To connect Open WebUI to this service:

1. In Open WebUI, go to **Settings > Connections**.
2. Add a new **OpenAI API** connection (or a custom one).
3. Set the **API Base URL** to `http://localhost:8000/query` (Note: You might need a wrapper or use the Functions/Tools feature if Open WebUI expects a strict OpenAI format, but the `/query` endpoint is ready for custom integration).
4. Alternatively, use a "Function" in Open WebUI to call this API.

### Troubleshooting
- If no files are found, check the `INBOX_FOLDER` path in `.env`.
- If memory is an issue, reduce `CHUNK_SIZE`.
- The first run will download the embedding model (`BAAI/bge-m3`).

## Project Structure
- `src/config/`: Configuration management.
- `src/ingestion/`: Text extraction and chunking.
- `src/indexing/`: Vector store and embedding logic.
- `src/rag/`: Retrieval and generation pipelines.
- `src/ui/`: FastAPI integration.
- `src/utils/`: Hashing, logging, and manifest management.
