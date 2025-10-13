import numpy as np
from typing import Tuple

def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int) -> Tuple[list, np.ndarray]:
    sims = matrix @ query / (np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) + 1e-10) + 1e-10)
    idxs = np.argsort(-sims)[:k]
    return idxs.tolist(), sims[idxs]
