"""
Tokenizer module for handwriting recognition.

This module provides:
- CharTokenizer: Character-level tokenizer with special tokens
- Utilities for building tokenizer vocabularies

All tokenizers follow a consistent interface:
- encode(text) -> list[int]
- decode(ids) -> str
- PAD, BOS, EOS, UNK special token IDs
- vocab_size property
"""

from hwrformer.tokenizer.base import BaseTokenizer
from hwrformer.tokenizer.char import CharTokenizer
from hwrformer.tokenizer.utils import normalize_text

__all__ = [
    "BaseTokenizer",
    "CharTokenizer",
    "normalize_text",
]
