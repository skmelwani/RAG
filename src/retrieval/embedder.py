"""Turn text into vectors with a sentence-transformers model."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DocumentEmbedder(Protocol):
    """Anything with embed_documents(). The indexer depends on this, not on
    the real model, so tests can pass a tiny fake embedder instead."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


class Embedder:
    """Wraps a sentence-transformers model, e.g. BAAI/bge-small-en-v1.5.

    Documents and queries are embedded differently ("asymmetric search"):
    bge was trained with an instruction before QUERIES only. Leaving it out
    raises no error, it just quietly makes retrieval worse.
    """

    def __init__(
        self, model_name: str, device: str = "cpu", batch_size: int = 32, query_prefix: str = ""
    ) -> None:
        logger.info("Loading embedding model %s on %s", model_name, device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix

    @property
    def dimension(self) -> int:
        """Length of each vector (384 for bge-small)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed chunks. Returns an array of shape (len(texts), dimension).

        normalize_embeddings=True scales every vector to length 1, so
        cosine similarity becomes a plain dot product.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,  # texts processed together: faster than one by one
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one question, with the query instruction in front. Shape: (dimension,)."""
        return self.model.encode(
            [query],
            prompt=self.query_prefix,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
