from kcenter import greedy_k_center_indices, assign_labels_topL
from binary import binary_search_kernel, pack_signs_to_uint64

import numpy as np
from typing import Optional

def greedy_k_center(
    docs: np.ndarray,
    K: int,
    num_centers: int=3,
    normalized: bool = True,
    start_index: int = 0,
    dtype = np.float32
):
    ''''
    Greedy k-center clustering (Gonzalez).
    Args:
        docs : (N, D) array-like
            Document/embedding matrix.
        K : int
            Number of clusters (clamped to N).
        normalized : bool, default False
            If True, uses the unit-norm shortcut d^2 = 2 - 2 * dot(x, c).
            Only set True if rows of X are L2-normalized.
        start_index : int, default 0
            Index of the initial center. Any index is valid with the 2-approx guarantee.
        dtype : numpy dtype, default float32
            X is cast to this dtype before clustering.

    Returns:
        centers : (K, D) ndarray
            Selected centers (subset of X).
        labels : (N,) ndarray of int64
            Index into 0..K-1 for each row of X.
        centers_idx : (K,) ndarray of int64
            Indices into X for the chosen centers.
    '''
    docs = np.asarray(docs, dtype=dtype)
    centers_idx = greedy_k_center_indices(docs, int(K), bool(normalized), int(start_index))
    labels = assign_labels_topL(docs, centers_idx,num_centers, bool(normalized))
    centers = docs[centers_idx].copy()
    return centers, labels, centers_idx

def binary_quantize_batch(vectors: np.ndarray, Q:Optional[np.ndarray]=None):
    '''
    Quantize a batch of vectors to binary format using random projections.

    For consistency, the random projection matrix Q can be provided 
    and if not provided, it is still seeded to ensure reproducibility.
    The vectors are packed into np.uint64 format, where each bit represents a sign.

    Args:
        vectors: (N, D) np.ndarray, input vectors to quantize
        Q: (D, D) np.ndarray, optional precomputed random projection matrix
    Returns:
        packed: (N, W) np.uint64, packed binary vectors
    '''
    _, dims = vectors.shape

    if Q is None:
        rng = np.random.default_rng(0)
        A=rng.standard_normal((dims, dims))
        Q, _ = np.linalg.qr(A, mode="reduced")
    Q=np.ascontiguousarray(Q, dtype=np.float32)

    projections = vectors @ Q

    packed = pack_signs_to_uint64(projections)
    return packed


def binary_vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int = 10):
    '''
    Perform a binary vector search to find the top-k closest documents for each query.
    This is a thin wrapper around the binary_search_kernel function written in Numba.
    Args:
        queries: (Q, W) np.ndarray, binary query vectors
        docs: (D, W) np.ndarray, binary document vectors
        top_k: int, number of top results to return for each query
    Returns:
        idxs: (Q, top_k) np.ndarray, indices of the top-k closest documents for each query
    '''
    k = min(top_k, docs.shape[0])
    idxs = binary_search_kernel(docs, queries, k)
    return idxs