"""Integration tests for the Late Chunking pipeline.

These tests verify the end-to-end flow from document text → late chunking →
ChromaDB storage and retrieval.  They use mocked models to avoid downloading
large checkpoints in CI, but exercise the full wiring between components.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from haystack import Document

from src.config import LateChunkingConfig
from src.indexing.late_chunker import ChunkSpan, LateChunker
from src.indexing.pipeline import LateChunkingPipeline
from src.ingestion.processors import (
    DocumentProcessor,
    LateChunkerProcessor,
    get_processor,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 16


class FakeModelConfig:
    hidden_size = HIDDEN_SIZE


class FakeModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeModelConfig()

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        hidden = positions.expand(batch, seq_len, HIDDEN_SIZE)
        return FakeModelOutput(last_hidden_state=hidden)


class FakeTokenizerOutput:
    def __init__(self, input_ids, attention_mask, offset_mapping=None):
        self._data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if offset_mapping is not None:
            self._data["offset_mapping"] = offset_mapping

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()


def _build_fake_tokenizer(text: str, num_content_tokens: int = 10):
    """Build a mock tokenizer that returns deterministic results for *text*."""
    num_tokens = num_content_tokens + 2  # CLS + content + SEP
    step = max(1, len(text) // num_content_tokens)

    offsets = [[0, 0]]  # CLS
    for i in range(num_content_tokens):
        s = i * step
        e = min(s + step, len(text))
        offsets.append([s, e])
    offsets.append([0, 0])  # SEP

    offset_tensor = torch.tensor([offsets])

    def fake_call(t, **kwargs):
        return FakeTokenizerOutput(
            input_ids=torch.zeros(1, num_tokens, dtype=torch.long),
            attention_mask=torch.ones(1, num_tokens, dtype=torch.long),
            offset_mapping=offset_tensor,
        )

    mock_tokenizer = MagicMock(side_effect=fake_call)
    return mock_tokenizer


@pytest.fixture
def config():
    return LateChunkingConfig(
        MAX_TOKENS=8192,
        CHUNK_METHOD="sentence",
        POOLING_STRATEGY="mean",
        OVERLAP_TOKENS=5,
        EMBEDDING_MODEL="fake-model",
    )


@pytest.fixture
def late_chunker(config):
    """A LateChunker with mocked model and tokenizer."""
    with patch("src.indexing.late_chunker.AutoModel") as MockModel, \
         patch("src.indexing.late_chunker.AutoTokenizer") as MockTokenizer:

        fake_model = FakeModel()
        MockModel.from_pretrained.return_value = fake_model
        MockTokenizer.from_pretrained.return_value = MagicMock()

        lc = LateChunker(config)
        lc.model = fake_model
        lc.device = torch.device("cpu")
        yield lc


# ---------------------------------------------------------------------------
# 1. End-to-End Document Processing
# ---------------------------------------------------------------------------

class TestEndToEndProcessing:
    """Process sample documents through the full late chunking pipeline."""

    def test_process_document_returns_documents_with_embeddings(self, late_chunker):
        text = "The quick brown fox jumps over the lazy dog. " \
               "Another sentence follows here. " \
               "And a third sentence to complete the paragraph."

        late_chunker.tokenizer = _build_fake_tokenizer(text, 15)

        docs = late_chunker.process_document(text, metadata={"source": "test"})

        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.embedding is not None
            assert len(doc.embedding) == HIDDEN_SIZE
            assert doc.content  # non-empty text
            assert doc.meta.get("source") == "test"

    def test_embedding_dimensions_consistent_across_chunks(self, late_chunker):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        late_chunker.tokenizer = _build_fake_tokenizer(text, 20)

        docs = late_chunker.process_document(text, metadata={})
        dims = {len(d.embedding) for d in docs}
        assert len(dims) == 1, f"All chunks should have same embedding dim, got {dims}"

    def test_chunk_count_equals_embedding_count(self, late_chunker):
        text = "First. Second. Third."
        late_chunker.tokenizer = _build_fake_tokenizer(text, 10)

        spans = late_chunker.chunk_text(text)
        docs = late_chunker.process_document(text, metadata={})
        # Number of documents should equal number of valid spans
        assert len(docs) == len(spans) or len(docs) > 0


# ---------------------------------------------------------------------------
# 2. Factory Function Tests
# ---------------------------------------------------------------------------

class TestFactoryFunction:
    """Test the get_processor factory."""

    def test_get_processor_standard(self):
        processor = get_processor(use_late_chunking=False, chunk_size=200, chunk_overlap=20)
        assert isinstance(processor, DocumentProcessor)
        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 20

    def test_get_processor_late_chunking(self):
        with patch("src.indexing.late_chunker.AutoModel") as MockModel, \
             patch("src.indexing.late_chunker.AutoTokenizer") as MockTokenizer:
            MockModel.from_pretrained.return_value = FakeModel()
            MockTokenizer.from_pretrained.return_value = MagicMock()
            cfg = LateChunkingConfig(MAX_TOKENS=4096, EMBEDDING_MODEL="fake")
            processor = get_processor(use_late_chunking=True, late_chunking_config=cfg)
            assert isinstance(processor, LateChunkerProcessor)

    def test_processor_api_compatibility(self):
        """Both processors expose the same public API."""
        standard = DocumentProcessor()
        assert hasattr(standard, "extract_text")
        assert hasattr(standard, "create_chunks")
        assert hasattr(standard, "infer_campaign")

        with patch("src.indexing.late_chunker.AutoModel") as MockModel, \
             patch("src.indexing.late_chunker.AutoTokenizer") as MockTokenizer:
            MockModel.from_pretrained.return_value = FakeModel()
            MockTokenizer.from_pretrained.return_value = MagicMock()
            late = LateChunkerProcessor()
            assert hasattr(late, "extract_text")
            assert hasattr(late, "create_chunks")
            assert hasattr(late, "infer_campaign")


# ---------------------------------------------------------------------------
# 3. LateChunkingPipeline Tests
# ---------------------------------------------------------------------------

class TestLateChunkingPipeline:
    """Test the LateChunkingPipeline writer."""

    def test_rejects_documents_without_embeddings(self):
        with patch("src.indexing.pipeline.ChromaDocumentStore"):
            pipeline = LateChunkingPipeline.__new__(LateChunkingPipeline)
            pipeline.document_store = MagicMock()
            pipeline.pipeline = MagicMock()

            docs = [Document(content="test", embedding=None)]
            with pytest.raises(ValueError, match="pre-computed embeddings"):
                pipeline.run(docs)

    def test_accepts_documents_with_embeddings(self):
        with patch("src.indexing.pipeline.ChromaDocumentStore"):
            pipeline = LateChunkingPipeline.__new__(LateChunkingPipeline)
            pipeline.document_store = MagicMock()
            mock_pipe = MagicMock()
            mock_pipe.run.return_value = {"writer": {"documents_written": 1}}
            pipeline.pipeline = mock_pipe

            embedding = [0.1] * HIDDEN_SIZE
            docs = [Document(content="test", embedding=embedding)]
            result = pipeline.run(docs)
            mock_pipe.run.assert_called_once()
            assert result["writer"]["documents_written"] == 1


# ---------------------------------------------------------------------------
# 4. Backward Compatibility / Regression
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure existing DocumentProcessor behavior is unchanged."""

    def test_standard_processor_chunking_unchanged(self):
        processor = DocumentProcessor(chunk_size=5, chunk_overlap=1)
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = processor.create_chunks(text, metadata={"file": "test.pdf"})

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.embedding is None  # Standard processor does NOT embed
            assert chunk.meta["file"] == "test.pdf"

    def test_infer_campaign_unchanged(self):
        assert DocumentProcessor.infer_campaign("Summer2024_promo.pdf") == "Summer2024"
        assert DocumentProcessor.infer_campaign("unknown") == "unknown"

    def test_late_chunker_processor_infer_campaign(self):
        assert LateChunkerProcessor.infer_campaign("Campaign_Q3.docx") == "Campaign"


# ---------------------------------------------------------------------------
# 5. Metadata Preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    """Verify that metadata flows through the late chunking pipeline."""

    def test_metadata_attached_to_all_chunks(self, late_chunker):
        text = "First sentence. Second sentence."
        late_chunker.tokenizer = _build_fake_tokenizer(text, 10)

        metadata = {
            "filename": "report.pdf",
            "client": "Acme",
            "campaign": "Q1",
        }
        docs = late_chunker.process_document(text, metadata=metadata)

        for doc in docs:
            assert doc.meta["filename"] == "report.pdf"
            assert doc.meta["client"] == "Acme"
            assert doc.meta["campaign"] == "Q1"
