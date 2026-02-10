"""Minimal replacement for `pytorch3d.ops.knn_points`.

This implementation is O(N*M) using `torch.cdist` and is intended only for
small inputs during GVHMR inference utilities.
"""

from __future__ import annotations

from typing import Tuple

import torch


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int = 1,
    return_nn: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Args:
        p1: (B, N, D)
        p2: (B, M, D)
        K: number of nearest neighbors
        return_nn: if True, also return the neighbor points (B, N, K, D)

    Returns:
        dists: (B, N, K) squared L2 distances
        idx: (B, N, K) indices into p2
        nn: (B, N, K, D) neighbor points if requested, else None
    """
    if p1.dim() != 3 or p2.dim() != 3:
        raise ValueError("p1 and p2 must be 3D tensors (B, N, D)")
    if p1.shape[0] != p2.shape[0] or p1.shape[-1] != p2.shape[-1]:
        raise ValueError("p1 and p2 must have the same batch size and point dimension")
    if K <= 0:
        raise ValueError("K must be >= 1")

    # (B, N, M)
    dist = torch.cdist(p1, p2, p=2)
    dist2 = dist * dist
    dists, idx = torch.topk(dist2, k=K, dim=-1, largest=False, sorted=True)

    nn = None
    if return_nn:
        B, N, _ = idx.shape
        D = p1.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(B, N, K, D)
        p2_exp = p2.unsqueeze(1).expand(B, N, p2.shape[1], D)
        nn = torch.gather(p2_exp, dim=2, index=idx_exp)

    return dists, idx, nn

