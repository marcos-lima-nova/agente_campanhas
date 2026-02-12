"""Late Chunking module for context-preserving document embeddings.

Implements the "Late Chunking" strategy: tokenize the full document once to
produce token-level embeddings that carry cross-chunk context, then pool
those embeddings per chunk span to obtain one vector per chunk.

References:
    - Jina AI Late Chunking paper (2024)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from haystack import Document
from transformers import AutoModel, AutoTokenizer

from src.config import LateChunkingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkSpan:
    """Represents a text chunk and its token-level span within the full document.

    Attributes:
        text: The raw chunk text.
        start_token: Inclusive start index in the tokenized document.
        end_token: Exclusive end index in the tokenized document.
        metadata: Arbitrary metadata associated with this chunk.
    """

    text: str
    start_token: int
    end_token: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class TokenizedOutput:
    """Container for full-document tokenization results.

    Attributes:
        input_ids: Token id tensor of shape ``(1, seq_len)``.
        attention_mask: Attention mask tensor of shape ``(1, seq_len)``.
        offset_mapping: List of ``(start_char, end_char)`` tuples per token.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# LateChunker
# ---------------------------------------------------------------------------

class LateChunker:
    """Generates chunk embeddings using the Late Chunking algorithm.

    The workflow is:
    1. Split the document text into semantic chunks and record character spans.
    2. Tokenize the entire document to get token-level inputs and offset mappings.
    3. Map each chunk's character span to a *token* span.
    4. Run the model to produce token-level embeddings (with sliding-window
       handling for long documents).
    5. Pool the token embeddings within each chunk span to produce one vector
       per chunk.

    Args:
        config: A :class:`LateChunkingConfig` instance. Uses defaults when
            ``None``.
    """

    def __init__(self, config: Optional[LateChunkingConfig] = None) -> None:
        self.config = config or LateChunkingConfig()

        logger.info(
            "Initializing LateChunker with model=%s, max_tokens=%d",
            self.config.EMBEDDING_MODEL,
            self.config.MAX_TOKENS,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.config.EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("LateChunker ready on device=%s", self.device)

    # ------------------------------------------------------------------
    # 1. Chunk text → character spans
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> List[ChunkSpan]:
        """Split *text* into chunks and compute their **token** spans.

        The method first splits the text using the configured chunk method,
        records each chunk's character offsets, tokenizes the full document
        with ``return_offsets_mapping=True``, and then converts character
        offsets to token indices.

        Args:
            text: The full document text.

        Returns:
            A list of :class:`ChunkSpan` objects with populated
            ``start_token`` / ``end_token`` fields.
        """
        # --- character-level splits ---
        char_spans: List[Tuple[str, int, int]] = self._split_to_char_spans(text)

        # --- tokenize full document with offsets ---
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        offset_mapping: List[Tuple[int, int]] = encoding["offset_mapping"][0].tolist()

        # --- map character spans → token spans ---
        chunk_spans: List[ChunkSpan] = []
        for chunk_text, char_start, char_end in char_spans:
            tok_start, tok_end = self._char_span_to_token_span(
                offset_mapping, char_start, char_end
            )
            if tok_start is not None and tok_end is not None and tok_end > tok_start:
                chunk_spans.append(
                    ChunkSpan(
                        text=chunk_text,
                        start_token=tok_start,
                        end_token=tok_end,
                    )
                )
            else:
                logger.warning(
                    "Could not map chunk to token span; skipping chunk: %r",
                    chunk_text[:80],
                )

        return chunk_spans

    # ------------------------------------------------------------------
    # 2. Full-document tokenization
    # ------------------------------------------------------------------

    def tokenize_document(self, text: str) -> TokenizedOutput:
        """Tokenize the full document without truncation.

        Args:
            text: The full document text.

        Returns:
            A :class:`TokenizedOutput` containing input ids, attention mask,
            and character-offset mappings.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        return TokenizedOutput(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            offset_mapping=encoding["offset_mapping"][0].tolist(),
        )

    # ------------------------------------------------------------------
    # 3. Token-level embedding (with sliding window for long docs)
    # ------------------------------------------------------------------

    def embed_tokens(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate token-level embeddings, handling long documents via
        a sliding window when the sequence exceeds ``MAX_TOKENS``.

        Args:
            input_ids: Token ids of shape ``(1, seq_len)``.
            attention_mask: Optional attention mask of same shape.

        Returns:
            Tensor of shape ``(seq_len, hidden_size)`` with one embedding
            vector per token position.
        """
        seq_len = input_ids.shape[1]
        max_tokens = self.config.MAX_TOKENS
        overlap = self.config.OVERLAP_TOKENS

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if seq_len <= max_tokens:
            # Single-pass embedding
            return self._forward_pass(input_ids, attention_mask)

        # --- Sliding window ---
        logger.info(
            "Document has %d tokens (max %d); using sliding window with overlap=%d",
            seq_len,
            max_tokens,
            overlap,
        )

        # We'll accumulate embeddings and counts for averaging overlaps
        hidden_size = self._get_hidden_size()
        token_embeddings = torch.zeros(seq_len, hidden_size, device="cpu")
        token_counts = torch.zeros(seq_len, 1, device="cpu")

        start = 0
        while start < seq_len:
            end = min(start + max_tokens, seq_len)
            window_ids = input_ids[:, start:end].to(self.device)
            window_mask = attention_mask[:, start:end].to(self.device)

            window_embeddings = self._forward_pass(window_ids, window_mask)  # (window_len, hidden)
            window_embeddings = window_embeddings.cpu()

            token_embeddings[start:end] += window_embeddings
            token_counts[start:end] += 1.0

            if end >= seq_len:
                break
            start += max_tokens - overlap

        # Average overlapping regions
        token_counts = token_counts.clamp(min=1.0)
        token_embeddings = token_embeddings / token_counts

        return token_embeddings

    # ------------------------------------------------------------------
    # 4. Pool token embeddings per chunk span
    # ------------------------------------------------------------------

    def pool_embeddings(
        self, token_embeddings: torch.Tensor, spans: List[ChunkSpan]
    ) -> List[np.ndarray]:
        """Pool token-level embeddings into one vector per chunk span.

        Supports three strategies controlled by
        ``config.POOLING_STRATEGY``:

        - **mean**: Simple average of token embeddings in the span.
        - **max**: Element-wise maximum across tokens.
        - **weighted_mean**: Linearly increasing weights (later tokens
          weighted more).

        Args:
            token_embeddings: Tensor of shape ``(seq_len, hidden_size)``.
            spans: List of :class:`ChunkSpan` with token boundaries.

        Returns:
            A list of numpy arrays, one embedding per chunk.
        """
        strategy = self.config.POOLING_STRATEGY
        pooled: List[np.ndarray] = []

        for span in spans:
            span_emb = token_embeddings[span.start_token : span.end_token]

            if span_emb.shape[0] == 0:
                logger.warning(
                    "Empty span for chunk %r; using zero vector", span.text[:60]
                )
                pooled.append(np.zeros(token_embeddings.shape[1]))
                continue

            if strategy == "mean":
                vec = span_emb.mean(dim=0)
            elif strategy == "max":
                vec = span_emb.max(dim=0).values
            elif strategy == "weighted_mean":
                weights = torch.linspace(1.0, 2.0, steps=span_emb.shape[0]).unsqueeze(1)
                weights = weights.to(span_emb.device)
                vec = (span_emb * weights).sum(dim=0) / weights.sum()
            else:
                raise ValueError(f"Unknown pooling strategy: {strategy}")

            # L2-normalize
            vec_np = vec.detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vec_np)
            if norm > 0:
                vec_np = vec_np / norm
            pooled.append(vec_np)

        return pooled

    # ------------------------------------------------------------------
    # 5. High-level: process a complete document
    # ------------------------------------------------------------------

    def process_document(
        self, text: str, metadata: Optional[Dict] = None
    ) -> List[Document]:
        """End-to-end late chunking pipeline for a single document.

        1. Split text into chunks and compute token spans.
        2. Tokenize the full document.
        3. Generate token-level embeddings (with sliding window if needed).
        4. Pool embeddings per chunk span.
        5. Return Haystack :class:`Document` objects with pre-computed
           embeddings.

        Args:
            text: Full document text.
            metadata: Optional base metadata dict to attach to every chunk.

        Returns:
            List of Haystack ``Document`` objects, each with its ``embedding``
            field set.
        """
        if metadata is None:
            metadata = {}

        if not text or not text.strip():
            logger.warning("Empty text passed to process_document; returning empty list.")
            return []

        # 1. Chunk & span annotation
        spans = self.chunk_text(text)
        if not spans:
            logger.warning("No valid spans produced; returning entire text as single chunk.")
            spans = [ChunkSpan(text=text.strip(), start_token=0, end_token=0)]

        # 2. Tokenize full document
        tok_output = self.tokenize_document(text)

        # Fix single-chunk fallback span end
        if spans[-1].end_token == 0 and len(spans) == 1:
            spans[0].end_token = tok_output.input_ids.shape[1]

        # 3. Token-level embeddings
        token_embeddings = self.embed_tokens(
            tok_output.input_ids, tok_output.attention_mask
        )

        # 4. Pool
        chunk_embeddings = self.pool_embeddings(token_embeddings, spans)

        # 5. Build Haystack Documents
        documents: List[Document] = []
        for span, embedding in zip(spans, chunk_embeddings):
            chunk_meta = {**metadata, **span.metadata}
            doc = Document(
                content=span.text,
                meta=chunk_meta,
                embedding=embedding.tolist(),
            )
            documents.append(doc)

        logger.info(
            "Processed document into %d chunks (embedding dim=%d)",
            len(documents),
            chunk_embeddings[0].shape[0] if chunk_embeddings else 0,
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_to_char_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into chunks using the configured method and return
        ``(chunk_text, char_start, char_end)`` tuples.
        """
        method = self.config.CHUNK_METHOD

        if method == "sentence":
            return self._split_sentences(text)
        elif method == "paragraph":
            return self._split_paragraphs(text)
        elif method == "semantic":
            # Semantic chunking falls back to sentence-level for now.
            # A future enhancement could use embedding similarity.
            logger.info("Semantic chunking not yet implemented; falling back to sentence splitting.")
            return self._split_sentences(text)
        else:
            raise ValueError(f"Unknown chunk method: {method}")

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences using NLTK's sent_tokenize with
        character offset tracking.
        """
        try:
            import nltk
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            import nltk
            nltk.download("punkt_tab", quiet=True)

        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
        spans: List[Tuple[str, int, int]] = []
        search_start = 0

        for sent in sentences:
            idx = text.find(sent, search_start)
            if idx == -1:
                # Fallback: try to find a close match
                logger.warning("Could not locate sentence in text; skipping: %r", sent[:60])
                continue
            spans.append((sent, idx, idx + len(sent)))
            search_start = idx + len(sent)

        return spans

    def _split_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text on double-newline boundaries."""
        spans: List[Tuple[str, int, int]] = []
        for match in re.finditer(r"[^\n](?:[^\n]|\n(?!\n))*", text):
            chunk = match.group().strip()
            if chunk:
                spans.append((chunk, match.start(), match.end()))
        return spans

    @staticmethod
    def _char_span_to_token_span(
        offset_mapping: List[Tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Convert a character-level span to a token-level span using the
        tokenizer's offset mapping.

        Returns:
            ``(tok_start, tok_end)`` where ``tok_end`` is exclusive, or
            ``(None, None)`` if the span could not be mapped.
        """
        tok_start: Optional[int] = None
        tok_end: Optional[int] = None

        for i, (cs, ce) in enumerate(offset_mapping):
            # Skip special tokens (offset 0,0 that aren't the first real token)
            if cs == 0 and ce == 0:
                continue
            if cs >= char_end:
                break
            if ce <= char_start:
                continue
            if tok_start is None:
                tok_start = i
            tok_end = i + 1

        return tok_start, tok_end

    @torch.no_grad()
    def _forward_pass(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run a single forward pass and return the last hidden state.

        Args:
            input_ids: Shape ``(1, window_len)``.
            attention_mask: Shape ``(1, window_len)``.

        Returns:
            Tensor of shape ``(window_len, hidden_size)``.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state shape: (batch=1, seq_len, hidden_size)
        return outputs.last_hidden_state.squeeze(0).cpu()

    def _get_hidden_size(self) -> int:
        """Return the model's hidden size from its config."""
        return self.model.config.hidden_size
