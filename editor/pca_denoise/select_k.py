# 根据阈值选择主成分
from __future__ import annotations

from typing import Optional, Tuple

import torch

# Compute explained variance ratio and cumulative sum from singular values
def explained_variance_ratio_from_singular_values(
    S: torch.Tensor,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:

    S_f = S.float()
    eig = S_f * S_f
    total = eig.sum() + eig.new_tensor(eps)
    ratio = eig / total
    cumsum = torch.cumsum(ratio, dim=0)
    return ratio, cumsum

# Select optimal number of components k based on variance threshold
def choose_k_by_threshold(
    cumsum: torch.Tensor,
    var_threshold: float,
    min_k: int = 1,
) -> int:

    tau = float(var_threshold)
    r = int(cumsum.numel())
    # Find minimal k with cumsum >= tau
    k = int((cumsum < tau).sum().item()) + 1
    k = max(min_k, min(k, r))
    return k


def choose_k_from_singular_values(
    S: torch.Tensor,
    var_threshold: float,
    eps: float = 1e-12,
    min_k: int = 1,
) -> Tuple[int, torch.Tensor, torch.Tensor]:

    ratio, cumsum = explained_variance_ratio_from_singular_values(S, eps=eps)
    k = choose_k_by_threshold(cumsum, var_threshold, min_k=min_k)
    return k, ratio, cumsum
