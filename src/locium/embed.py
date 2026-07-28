"""Embed search queries with the model that produced the stored vectors.

MemPalace does not override Chroma's default embedding function, so drawers
were embedded with ONNX all-MiniLM-L6-v2. Reusing that exact function keeps
query vectors in the same space and avoids a second model download — the
weights are already cached under ~/.cache/chroma/onnx_models/.
"""

import threading

import numpy as np

_embedder = None
_embedder_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                try:
                    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
                except ImportError:
                    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
                        ONNXMiniLM_L6_V2,
                    )
                _embedder = ONNXMiniLM_L6_V2()
    return _embedder


def embed_query(text: str) -> np.ndarray:
    """Return an L2-normalised float32 embedding for a search query."""
    raw = _get_embedder()([text])[0]
    vector = np.asarray(raw, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalised float32 embeddings for a batch of documents.

    Same model, same space as the stored drawer vectors. The model truncates
    around 256 tokens, so for a long stitched message this effectively embeds
    its head -- which is still strictly more than any single middle chunk of
    it ever represented.
    """
    raw = _get_embedder()(list(texts))
    matrix = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return matrix / norms
