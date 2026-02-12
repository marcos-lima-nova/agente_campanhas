# Late Chunking

## Overview

Late Chunking is an embedding strategy that preserves cross-chunk context by generating token-level embeddings from the **full document** before pooling them into chunk-level vectors. This contrasts with the traditional "chunk-then-embed" approach where each chunk is embedded independently.

## Why Late Chunking?

| Problem with Naive Chunking | How Late Chunking Helps |
|---|---|
| Each chunk is embedded in isolation | Tokens see the full document context during the model forward pass |
| Boundary tokens lose surrounding meaning | Overlapping sliding windows preserve boundary context |
| Word-based splitting ignores token boundaries | Splitting uses sentence/paragraph boundaries with token-aware span mapping |

## Architecture

```
Document Text
    │
    ├── 1. Split into chunks (sentence / paragraph)
    │       └── Record character offsets per chunk
    │
    ├── 2. Tokenize full document
    │       └── Get offset mapping (char → token)
    │
    ├── 3. Map chunk char-spans → token-spans
    │
    ├── 4. Run model forward pass (sliding window if needed)
    │       └── Produces token-level embeddings
    │
    └── 5. Pool token embeddings per chunk span
            └── One L2-normalized vector per chunk
```

## Quick Start

### Enable via Environment Variables

Add to your `.env` file:

```bash
USE_LATE_CHUNKING=true
EMBEDDING_MODEL=BAAI/bge-m3
MAX_TOKENS=8192
CHUNK_METHOD=sentence       # sentence | paragraph | semantic
POOLING_STRATEGY=mean       # mean | max | weighted_mean
OVERLAP_TOKENS=50
```

Then run ingestion as usual:

```bash
python -m src.ingestion.run
```

### Programmatic Usage

```python
from src.config import LateChunkingConfig
from src.indexing.late_chunker import LateChunker

config = LateChunkingConfig(
    MAX_TOKENS=8192,
    CHUNK_METHOD="sentence",
    POOLING_STRATEGY="mean",
)
chunker = LateChunker(config)

documents = chunker.process_document(
    text="Your full document text here...",
    metadata={"filename": "report.pdf", "client": "Acme"},
)

# Each Document has .content (chunk text) and .embedding (list[float])
for doc in documents:
    print(f"Chunk: {doc.content[:80]}...")
    print(f"Embedding dim: {len(doc.embedding)}")
```

### Using the Factory Function

```python
from src.ingestion.processors import get_processor

# Returns LateChunkerProcessor when True, DocumentProcessor when False
processor = get_processor(use_late_chunking=True)

text = processor.extract_text("path/to/file.pdf")
chunks = processor.create_chunks(text, metadata={"source": "file.pdf"})
```

## Configuration Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `USE_LATE_CHUNKING` | bool | `false` | Enable late chunking pipeline |
| `MAX_TOKENS` | int | `8192` | Model context window limit |
| `CHUNK_METHOD` | str | `"sentence"` | How to split: `sentence`, `paragraph`, `semantic` |
| `POOLING_STRATEGY` | str | `"mean"` | How to pool token embeddings: `mean`, `max`, `weighted_mean` |
| `OVERLAP_TOKENS` | int | `50` | Token overlap between sliding windows |
| `EMBEDDING_MODEL` | str | `"BAAI/bge-m3"` | HuggingFace model for embeddings |

## Pooling Strategies

- **mean**: Simple average of all token embeddings in the chunk span. Best general-purpose option.
- **max**: Element-wise maximum across tokens. Good for capturing salient features.
- **weighted_mean**: Linearly increasing weights (later tokens weighted more). Useful when recency matters.

All strategies produce L2-normalized output vectors.

## Context Window Handling

When a document exceeds `MAX_TOKENS`:

1. A sliding window moves across the token sequence with stride = `MAX_TOKENS - OVERLAP_TOKENS`.
2. Each window produces token-level embeddings independently.
3. Overlapping regions are averaged to produce smooth transitions.
4. The final result is a single embedding tensor covering all tokens.

## Backward Compatibility

- The existing `DocumentProcessor` and `IndexingPipeline` classes are unchanged.
- `LateChunkerProcessor` and `LateChunkingPipeline` are additive — they don't modify existing APIs.
- The `get_processor()` factory function selects the right processor at runtime.
- ChromaDB storage is fully compatible (same embedding format).

## File Structure

```
src/
├── config/__init__.py          # LateChunkingConfig dataclass
├── indexing/
│   ├── pipeline.py             # IndexingPipeline + LateChunkingPipeline
│   └── late_chunker.py         # LateChunker, ChunkSpan, TokenizedOutput
├── ingestion/
│   ├── processors.py           # DocumentProcessor + LateChunkerProcessor + get_processor()
│   └── run.py                  # USE_LATE_CHUNKING env var support
└── utils/
    └── migration.py            # Re-indexing utility

tests/
├── test_late_chunker.py                # 25 unit tests
└── test_late_chunking_integration.py   # 12 integration tests
```

## Running Tests

```bash
# Unit tests
python -m pytest tests/test_late_chunker.py -v

# Integration tests
python -m pytest tests/test_late_chunking_integration.py -v

# All late chunking tests
python -m pytest tests/test_late_chunker.py tests/test_late_chunking_integration.py -v
```
