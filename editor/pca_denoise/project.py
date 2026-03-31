from __future__ import annotations

from typing import Optional

import torch

from .config import PCADenoiseConfig
from .select_k import choose_k_from_singular_values

def pca_project_denoise_by_threshold(
    H: torch.Tensor, 
    cfg: Optional[PCADenoiseConfig] = None,
) -> torch.Tensor:

    orig_dtype = H.dtype
    orig_device = H.device

    H_work = H.float() if H.dtype != torch.float32 else H
    mean = H_work.mean(dim=0, keepdim=True)
    Hc = H_work - mean

    # SVD：Hc = U S Vh
    _, S, Vh = torch.linalg.svd(Hc, full_matrices=False)

    r = int(S.numel())
    k, ratio, cumsum = choose_k_from_singular_values(
        S,
        var_threshold=cfg.var_threshold,
        eps=cfg.eps,
        min_k=cfg.min_k,
    )
    Uk_full = Vh[:k, :].T

    H_k = Hc @ Uk_full
    H_k.add_(mean @ Uk_full)
    H_k = H_k @ Uk_full.T

    return H_k.to(device=orig_device, dtype=orig_dtype)
