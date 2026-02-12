"""Unit tests for the Late Chunker module.

Covers:
- Chunk/span alignment
- Embedding count consistency
- Determinism
- Context window handling (sliding window)
- Pooling strategies
- Edge cases (empty input, single-chunk documents)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import LateChunkingConfig
from src.indexing.late_chunker import ChunkSpan, LateChunker, TokenizedOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small model we can use in tests without downloading a huge checkpoint.
# We mock the heavy HuggingFace model loading and use deterministic tensors.

HIDDEN_SIZE = 16
SEQ_LEN = 30  # short sequence for tests


class FakeModelConfig:
    hidden_size = HIDDEN_SIZE


class FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class FakeModel(torch.nn.Module):
    """Deterministic fake model that returns index-based embeddings."""

    def __init__(self):
        super().__init__()
        self.config = FakeModelConfig()

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        # Each token embedding = its position index repeated across hidden_size
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        hidden = positions.expand(batch, seq_len, HIDDEN_SIZE)
        return FakeModelOutput(last_hidden_state=hidden)


class FakeTokenizerOutput:
    """Mimics tokenizer output dict-like access."""

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


def _make_offset_mapping(text: str, token_texts: list[str]) -> torch.Tensor:
    """Build a fake offset mapping from token strings.

    Returns a tensor of shape (1, num_tokens, 2).
    """
    offsets = []
    # CLS token
    offsets.append([0, 0])
    search_start = 0
    for tok in token_texts:
        idx = text.find(tok, search_start)
        if idx == -1:
            idx = search_start
        offsets.append([idx, idx + len(tok)])
        search_start = idx + len(tok)
    # SEP token
    offsets.append([0, 0])
    return torch.tensor([offsets])


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
def chunker(config):
    """Create a LateChunker with mocked model/tokenizer."""
    with patch("src.indexing.late_chunker.AutoModel") as MockModel, \
         patch("src.indexing.late_chunker.AutoTokenizer") as MockTokenizer:

        fake_model = FakeModel()
        MockModel.from_pretrained.return_value = fake_model

        # We'll set up the tokenizer mock per-test where needed,
        # but provide a basic default
        mock_tokenizer = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer

        lc = LateChunker(config)
        # Overwrite with our fake model directly
        lc.model = fake_model
        lc.device = torch.device("cpu")
        lc.tokenizer = mock_tokenizer
        yield lc


# ---------------------------------------------------------------------------
# 1. Chunk/Span Alignment Tests
# ---------------------------------------------------------------------------

class TestChunkSpanAlignment:
    """Verify chunk text matches span slicings."""

    def test_char_span_to_token_span_basic(self):
        """Character offsets correctly map to token indices."""
        # Fake offset mapping: [CLS](0,0), tok0(0,5), tok1(5,10), tok2(10,15), [SEP](0,0)
        offsets = [(0, 0), (0, 5), (5, 10), (10, 15), (0, 0)]
        start, end = LateChunker._char_span_to_token_span(offsets, 0, 5)
        assert start == 1
        assert end == 2

    def test_char_span_to_token_span_multi_token(self):
        offsets = [(0, 0), (0, 5), (5, 10), (10, 15), (0, 0)]
        start, end = LateChunker._char_span_to_token_span(offsets, 0, 15)
        assert start == 1
        assert end == 4

    def test_char_span_to_token_span_middle(self):
        offsets = [(0, 0), (0, 5), (5, 10), (10, 15), (0, 0)]
        start, end = LateChunker._char_span_to_token_span(offsets, 5, 10)
        assert start == 2
        assert end == 3

    def test_char_span_to_token_span_no_match(self):
        offsets = [(0, 0), (0, 5), (5, 10), (0, 0)]
        start, end = LateChunker._char_span_to_token_span(offsets, 20, 30)
        assert start is None
        assert end is None

    def test_chunk_text_produces_valid_spans(self, chunker):
        """chunk_text returns spans where each text can be found in original."""
        text = "Hello world. This is a test. Another sentence here."

        # Set up the tokenizer mock to return proper offset mapping
        token_texts = ["Hello", " world", ".", " This", " is", " a", " test", ".",
                       " Another", " sentence", " here", "."]
        offsets = _make_offset_mapping(text, token_texts)
        num_tokens = offsets.shape[1]

        def fake_tokenize(t, **kwargs):
            return FakeTokenizerOutput(
                input_ids=torch.zeros(1, num_tokens, dtype=torch.long),
                attention_mask=torch.ones(1, num_tokens, dtype=torch.long),
                offset_mapping=offsets,
            )

        chunker.tokenizer.side_effect = fake_tokenize

        spans = chunker.chunk_text(text)
        assert len(spans) > 0
        for span in spans:
            # Each chunk text should appear in the original text
            assert span.text in text or text.find(span.text.strip()) >= 0
            assert span.start_token < span.end_token


# ---------------------------------------------------------------------------
# 2. Embedding Count Consistency Tests
# ---------------------------------------------------------------------------

class TestEmbeddingCountConsistency:
    """Verify number of embeddings equals number of chunks."""

    def test_pool_embeddings_count_matches_spans(self, chunker):
        spans = [
            ChunkSpan(text="Hello", start_token=1, end_token=3),
            ChunkSpan(text="World", start_token=3, end_token=6),
            ChunkSpan(text="Test", start_token=6, end_token=8),
        ]
        token_emb = torch.randn(10, HIDDEN_SIZE)
        pooled = chunker.pool_embeddings(token_emb, spans)
        assert len(pooled) == len(spans)

    def test_embedding_dimensions_consistent(self, chunker):
        spans = [
            ChunkSpan(text="A", start_token=1, end_token=4),
            ChunkSpan(text="B", start_token=4, end_token=7),
        ]
        token_emb = torch.randn(10, HIDDEN_SIZE)
        pooled = chunker.pool_embeddings(token_emb, spans)
        for emb in pooled:
            assert emb.shape == (HIDDEN_SIZE,)


# ---------------------------------------------------------------------------
# 3. Determinism Tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same input produces same output across runs."""

    def test_embed_tokens_deterministic(self, chunker):
        input_ids = torch.randint(0, 100, (1, 20))
        mask = torch.ones_like(input_ids)

        out1 = chunker.embed_tokens(input_ids, mask)
        out2 = chunker.embed_tokens(input_ids, mask)

        assert torch.allclose(out1, out2), "embed_tokens should be deterministic"

    def test_pool_embeddings_deterministic(self, chunker):
        token_emb = torch.randn(10, HIDDEN_SIZE)
        spans = [ChunkSpan(text="x", start_token=2, end_token=5)]

        p1 = chunker.pool_embeddings(token_emb, spans)
        p2 = chunker.pool_embeddings(token_emb, spans)

        np.testing.assert_array_equal(p1[0], p2[0])


# ---------------------------------------------------------------------------
# 4. Context Window Tests
# ---------------------------------------------------------------------------

class TestContextWindow:
    """Test documents under and over MAX_TOKENS."""

    def test_short_document_single_pass(self, chunker):
        """Documents under MAX_TOKENS use a single forward pass."""
        chunker.config.MAX_TOKENS = 100
        input_ids = torch.randint(0, 100, (1, 50))
        mask = torch.ones_like(input_ids)

        result = chunker.embed_tokens(input_ids, mask)
        assert result.shape == (50, HIDDEN_SIZE)

    def test_long_document_sliding_window(self, chunker):
        """Documents over MAX_TOKENS trigger sliding window."""
        chunker.config.MAX_TOKENS = 10
        chunker.config.OVERLAP_TOKENS = 3
        input_ids = torch.randint(0, 100, (1, 25))
        mask = torch.ones_like(input_ids)

        result = chunker.embed_tokens(input_ids, mask)
        assert result.shape == (25, HIDDEN_SIZE)

    def test_sliding_window_overlap_averaging(self, chunker):
        """Overlapping regions should be averaged (not duplicated)."""
        chunker.config.MAX_TOKENS = 10
        chunker.config.OVERLAP_TOKENS = 5
        input_ids = torch.randint(0, 100, (1, 15))
        mask = torch.ones_like(input_ids)

        result = chunker.embed_tokens(input_ids, mask)
        # Should still have embeddings for all 15 tokens
        assert result.shape == (15, HIDDEN_SIZE)
        # No NaN values
        assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# 5. Pooling Strategy Tests
# ---------------------------------------------------------------------------

class TestPoolingStrategies:
    """Test different pooling strategies."""

    def test_mean_pooling(self, chunker):
        chunker.config.POOLING_STRATEGY = "mean"
        token_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        spans = [ChunkSpan(text="x", start_token=0, end_token=3)]
        pooled = chunker.pool_embeddings(token_emb, spans)
        # Mean of [1,3,5]=3 and [2,4,6]=4 => [3,4], then normalized
        expected = np.array([3.0, 4.0], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(pooled[0], expected, atol=1e-6)

    def test_max_pooling(self, chunker):
        chunker.config.POOLING_STRATEGY = "max"
        token_emb = torch.tensor([[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]])
        spans = [ChunkSpan(text="x", start_token=0, end_token=3)]
        pooled = chunker.pool_embeddings(token_emb, spans)
        # Max of [1,3,5]=5 and [6,4,2]=6 => [5,6], normalized
        expected = np.array([5.0, 6.0], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(pooled[0], expected, atol=1e-6)

    def test_weighted_mean_pooling(self, chunker):
        chunker.config.POOLING_STRATEGY = "weighted_mean"
        token_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        spans = [ChunkSpan(text="x", start_token=0, end_token=2)]
        pooled = chunker.pool_embeddings(token_emb, spans)
        assert pooled[0].shape == (2,)
        # Weights are [1.0, 2.0], so weighted = [1*1+0*2, 0*1+1*2]/3 = [1/3, 2/3]
        expected = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(pooled[0], expected, atol=1e-5)

    def test_unknown_pooling_raises(self, chunker):
        chunker.config.POOLING_STRATEGY = "unknown"
        token_emb = torch.randn(5, HIDDEN_SIZE)
        spans = [ChunkSpan(text="x", start_token=0, end_token=3)]
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            chunker.pool_embeddings(token_emb, spans)


# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases: empty input, single-chunk documents."""

    def test_empty_text_returns_empty(self, chunker):
        docs = chunker.process_document("", metadata={"test": True})
        assert docs == []

    def test_whitespace_only_returns_empty(self, chunker):
        docs = chunker.process_document("   \n\n  ", metadata={})
        assert docs == []

    def test_empty_span_produces_zero_vector(self, chunker):
        token_emb = torch.randn(10, HIDDEN_SIZE)
        spans = [ChunkSpan(text="empty", start_token=5, end_token=5)]  # empty range
        pooled = chunker.pool_embeddings(token_emb, spans)
        assert len(pooled) == 1
        np.testing.assert_array_equal(pooled[0], np.zeros(HIDDEN_SIZE))

    def test_l2_normalization(self, chunker):
        """Pooled embeddings should be L2-normalized."""
        token_emb = torch.randn(10, HIDDEN_SIZE)
        spans = [ChunkSpan(text="x", start_token=0, end_token=5)]
        pooled = chunker.pool_embeddings(token_emb, spans)
        norm = np.linalg.norm(pooled[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Configuration Tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Test LateChunkingConfig defaults and customization."""

    def test_default_config(self):
        cfg = LateChunkingConfig()
        assert cfg.MAX_TOKENS == 8192
        assert cfg.CHUNK_METHOD == "sentence"
        assert cfg.POOLING_STRATEGY == "mean"
        assert cfg.OVERLAP_TOKENS == 50
        assert cfg.EMBEDDING_MODEL == "BAAI/bge-m3"

    def test_custom_config(self):
        cfg = LateChunkingConfig(
            MAX_TOKENS=4096,
            CHUNK_METHOD="paragraph",
            POOLING_STRATEGY="max",
            OVERLAP_TOKENS=100,
            EMBEDDING_MODEL="custom/model",
        )
        assert cfg.MAX_TOKENS == 4096
        assert cfg.CHUNK_METHOD == "paragraph"
        assert cfg.POOLING_STRATEGY == "max"
        assert cfg.OVERLAP_TOKENS == 100
        assert cfg.EMBEDDING_MODEL == "custom/model"


# ---------------------------------------------------------------------------
# 8. Paragraph Splitting Tests
# ---------------------------------------------------------------------------

class TestParagraphSplitting:
    """Test the paragraph-based chunking method."""

    def test_split_paragraphs(self, chunker):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        spans = chunker._split_paragraphs(text)
        assert len(spans) == 3
        assert spans[0][0] == "First paragraph."
        assert spans[1][0] == "Second paragraph."
        assert spans[2][0] == "Third paragraph."

    def test_split_paragraphs_single(self, chunker):
        text = "Only one paragraph here."
        spans = chunker._split_paragraphs(text)
        assert len(spans) == 1
        assert spans[0][0] == "Only one paragraph here."

    def test_split_paragraphs_char_offsets_valid(self, chunker):
        text = "Para A.\n\nPara B."
        spans = chunker._split_paragraphs(text)
        for chunk_text, start, end in spans:
            assert text[start:end].strip() == chunk_text
