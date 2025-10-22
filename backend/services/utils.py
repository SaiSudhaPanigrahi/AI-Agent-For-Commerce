
import numpy as np
from typing import Tuple

def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.T

def topk_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    idxs = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
    # sort within top-k
    part = np.take_along_axis(scores, idxs, axis=1)
    order = np.argsort(-part, axis=1)
    sorted_idxs = np.take_along_axis(idxs, order, axis=1)
    sorted_scores = np.take_along_axis(scores, sorted_idxs, axis=1)
    return sorted_idxs, sorted_scores
