"""
HWRFormer: IMU Handwriting Recognition Library

Core modules:
- hwrformer.model: Neural network architectures (1D CNN encoder, AR Transformer
  decoder, Transformer-CTC, hybrid CTC-AR dual head)
- hwrformer.dataset: Data loading and preprocessing
- hwrformer.training: Training loops and utilities (CTC, AR, hybrid)
- hwrformer.analysis: Quantitative analysis tools (metrics, encoder features)
- hwrformer.tokenizer: Character-level text tokenization
- hwrformer.evaluate: Evaluation metrics
- hwrformer.visualize: Result visualization
"""

__version__ = "1.0.0"
