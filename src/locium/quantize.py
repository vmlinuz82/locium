"""int8 quantisation for the vectors shipped to the browser.

MiniLM embeddings are L2-normalised, so every component already lies in
[-1, 1] and a single global scale is enough. This cuts the payload roughly
four-fold while keeping neighbour rankings intact.
"""

import numpy as np

SCALE = 127.0


def quantize(vectors: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(vectors * SCALE), -127, 127).astype(np.int8)


def dequantize(quantized: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float32) / SCALE
