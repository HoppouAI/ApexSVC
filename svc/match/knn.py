"""kNN feature matcher.

For each (T_src, D) source frame, find the k nearest reference frames in
cosine distance and average them. Optional `alpha` blends a fraction of the
original source feature back in (alpha=0 = pure retrieval, like RVC index=1).

This is the zero-shot stand-in for RVC's index lookup: the output features
literally come from frames the reference singer actually produced, which is
why the timbre tracks the reference faithfully and avoids the over-smoothed
character of diffusion-based zero-shot SVC.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..utils.tools import fast_cosine_dist


@torch.inference_mode()
def knn_match(query_seq: Tensor, matching_pool: Tensor, *, topk: int = 4,
              alpha: float = 0.0, chunk: int = 4096,
              device: str | torch.device | None = None,
              on_chunk=None) -> Tensor:
    """Returns (T_src, D) matched features.

    `chunk` controls how many source frames are scored against the matching
    pool per cdist call, so peak VRAM stays bounded regardless of how long the
    source is. With 5 min reference (~15k frames at 50 Hz) and a 30 s source
    (~1.5k frames) the whole thing fits in <200 MB at float32.

    `on_chunk(done, total)` is called after each chunk if provided.
    """
    if device is None:
        device = query_seq.device
    device = torch.device(device)

    query_seq = query_seq.to(device)
    matching_pool = matching_pool.to(device)

    out_chunks: list[Tensor] = []
    n_total = (query_seq.shape[0] + chunk - 1) // chunk
    for i, start in enumerate(range(0, query_seq.shape[0], chunk)):
        q = query_seq[start:start + chunk]
        dists = fast_cosine_dist(q, matching_pool, device=device)
        nn = dists.topk(k=topk, largest=False, dim=-1)
        matched = matching_pool[nn.indices].mean(dim=1)  # (chunk, D)
        if alpha > 0:
            matched = (1.0 - alpha) * matched + alpha * q
        out_chunks.append(matched)
        if on_chunk is not None:
            on_chunk(i + 1, n_total)
    return torch.cat(out_chunks, dim=0)
