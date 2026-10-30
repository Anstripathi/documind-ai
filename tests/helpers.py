"""Test-only helpers. HashingEmbedder is a deterministic bag-of-words embedder used so
retrieval logic can be tested offline. It is NOT used by the application."""
import hashlib
import re

import numpy as np

from src.chunking import Chunk
from src.embeddings import l2_normalize

_STOP = {"the", "a", "an", "of", "is", "are", "what", "and", "to", "in", "how", "does", "do"}


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                if tok in _STOP:
                    continue
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        return l2_normalize(out)


def make_chunk(doc_id, page, text, idx=0, name="doc.pdf"):
    return Chunk(f"{doc_id}-p{page}-c{idx}", doc_id, name, page, idx, text)


SAMPLE_CHUNKS = [
    make_chunk("d1", 1, "Photosynthesis converts sunlight into chemical energy using chlorophyll in leaves."),
    make_chunk("d1", 2, "The French Revolution began in 1789 and transformed the political landscape of France."),
    make_chunk("d1", 3, "Python lists are mutable sequences while tuples are immutable sequences."),
]
