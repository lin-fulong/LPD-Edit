from dataclasses import dataclass
from typing import Optional


@dataclass
class PCADenoiseConfig:
    """
    PCA denoising config
    """

    # Explained variance threshold for selecting k
    var_threshold: float = 0.8

    # Minimum number of components
    min_k: int = 1

    # Small value to avoid division by zero
    eps: float = 1e-12