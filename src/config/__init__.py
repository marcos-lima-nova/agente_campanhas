"""Configuration module for the Agente Campanhas project.

Contains dataclass-based configuration for late chunking and other settings.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LateChunkingConfig:
    """Configuration for the Late Chunking embedding strategy.

    Attributes:
        MAX_TOKENS: Maximum token count for the model's context window.
        CHUNK_METHOD: Method used to split text into chunks before span annotation.
            - "sentence": Split on sentence boundaries using NLTK.
            - "paragraph": Split on paragraph boundaries (double newlines).
            - "semantic": Use semantic similarity to determine chunk boundaries.
        POOLING_STRATEGY: Strategy for pooling token-level embeddings into chunk embeddings.
            - "mean": Average all token embeddings in the span.
            - "max": Take element-wise maximum across token embeddings.
            - "weighted_mean": Weighted average with linearly increasing weights.
        OVERLAP_TOKENS: Number of overlapping tokens between sliding windows
            when a document exceeds MAX_TOKENS.
        EMBEDDING_MODEL: HuggingFace model identifier for generating embeddings.
    """

    MAX_TOKENS: int = 8192
    CHUNK_METHOD: Literal["sentence", "paragraph", "semantic"] = "sentence"
    POOLING_STRATEGY: Literal["mean", "max", "weighted_mean"] = "mean"
    OVERLAP_TOKENS: int = 50
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
