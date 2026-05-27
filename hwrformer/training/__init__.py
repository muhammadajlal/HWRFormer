"""
Training module for handwriting recognition.

This module provides:
- Training loop functions for CTC, AR, and hybrid CTC-AR models
- Evaluation/testing functions
- Training utilities (freeze/unfreeze, optimizer coverage, etc.)
"""

from hwrformer.training.loops import (
    train_one_epoch,
    train_one_epoch_hybrid,
    test,
    test_hybrid,
)
from hwrformer.training.utils import (
    build_ar_batch,
    count_params,
    maybe_log_trainability,
    set_decoder_frozen,
    log_decoder_pretrain_load,
    maybe_log_optimizer_coverage,
)

__all__ = [
    # Training loops
    "train_one_epoch",
    "train_one_epoch_hybrid",
    "test",
    "test_hybrid",
    # Utilities
    "build_ar_batch",
    "count_params",
    "maybe_log_trainability",
    "set_decoder_frozen",
    "log_decoder_pretrain_load",
    "maybe_log_optimizer_coverage",
]
