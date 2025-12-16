"""
Model Format Handlers

This module provides handlers for different model formats:
- QKV: Custom QKV Core format with adaptive compression
- GGUF: Quantized model format optimized for CPU/GPU inference (llama.cpp compatibility)
- PyTorch: Native PyTorch model format (.pt files)
- Smart Loading: Universal loader that auto-detects format
"""

from qkv_core.formats.gguf_loader import GGUFModelLoader, GGUFTokenizerWrapper
from qkv_core.formats.smart_loader import SmartLoader
from qkv_core.formats.huggingface_converter import HuggingFaceConverter

try:
    from qkv_core.formats.qkv_handler import QKVReader, QKVWriter
    QKV_FORMAT_AVAILABLE = True
except ImportError:
    QKV_FORMAT_AVAILABLE = False
    QKVReader = None
    QKVWriter = None

try:
    from qkv_core.formats.model_loader import UniversalModelLoader
except ImportError:
    UniversalModelLoader = None

__all__ = [
    "GGUFModelLoader",
    "GGUFTokenizerWrapper",
    "SmartLoader",
    "HuggingFaceConverter",
    "QKVReader",
    "QKVWriter",
    "QKV_FORMAT_AVAILABLE",
    "UniversalModelLoader",
]
