"""
Analysis module for quantitative evaluation.

This module provides tools for:
- Levenshtein distance utilities
- Encoder feature extraction (t-SNE / PCA, CTC posteriors)
"""

from hwrformer.analysis.metrics import lev_dist
from hwrformer.analysis.encoder_features import (
    extract_encoder_features,
    extract_ctc_posteriors,
    load_model_from_checkpoint,
)

__all__ = [
    # Metrics
    "lev_dist",
    # Encoder features
    "extract_encoder_features",
    "extract_ctc_posteriors",
    "load_model_from_checkpoint",
]
