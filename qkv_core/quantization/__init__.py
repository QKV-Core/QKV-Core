"""
Quantization Module - Adaptive Hybrid Compression

This module implements adaptive compression strategies for model quantization,
including surgical trimming and hybrid compression techniques.

Key Components:
- adaptive.py: Adaptive compression logic
- pruning.py: Surgical trimming logic for model optimization
"""

try:
    from qkv_core.quantization.adaptive import AdaptiveCompressor, CompressionMethod
except ImportError:
    AdaptiveCompressor = None
    CompressionMethod = None

try:
    from qkv_core.quantization.pruning import PruningEngine
except ImportError:
    PruningEngine = None

__all__ = [
    "AdaptiveCompressor",
    "CompressionMethod",
    "PruningEngine",
]

