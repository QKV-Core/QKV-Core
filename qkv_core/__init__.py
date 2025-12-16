"""
QKV Core - Professional Python Package

Query-Key-Value Core - The Core of Transformer Intelligence

This package provides a comprehensive framework for training, fine-tuning,
and deploying Large Language Models (LLMs) built on the fundamental
Query-Key-Value attention mechanism.

Main Components:
- core: Transformer architecture implementations
- formats: Model format handlers (GGUF, PyTorch)
- inference: Inference engines for text generation
- training: Training implementations (full training, LoRA, RLHF)
- tokenization: Tokenizer implementations (BPE, etc.)
- storage: Database implementations (SQLite, PostgreSQL)
- utils: Utility functions (logging, memory management, etc.)
"""

__version__ = "1.0.0"
__author__ = "QKV Core Team"

# Core exports
from qkv_core.core.transformer import TransformerModel

# Format handlers
from qkv_core.formats.gguf_loader import GGUFModelLoader, GGUFTokenizerWrapper
from qkv_core.formats.smart_loader import SmartLoader

# Quantization
try:
    from qkv_core.quantization.adaptive import AdaptiveCompressor, CompressionMethod
    from qkv_core.quantization.pruning import PruningEngine
except ImportError:
    AdaptiveCompressor = None
    CompressionMethod = None
    PruningEngine = None

# Kernels
try:
    from qkv_core.kernels.numba_engine import compress_chunk, decompress_chunk, NUMBA_AVAILABLE
except ImportError:
    compress_chunk = None
    decompress_chunk = None
    NUMBA_AVAILABLE = False

try:
    from qkv_core.formats.model_loader import UniversalModelLoader
except ImportError:
    UniversalModelLoader = None

# Inference engines
from qkv_core.inference.inference import InferenceEngine

# Training
try:
    from qkv_core.training.trainer import Trainer
except ImportError:
    Trainer = None

try:
    from qkv_core.training.incremental_trainer import IncrementalTrainer
except ImportError:
    IncrementalTrainer = None

# Tokenization
from qkv_core.tokenization.bpe import BPETokenizer

# Storage
try:
    from qkv_core.storage.db import DatabaseManager
except ImportError:
    DatabaseManager = None

# Utils
from qkv_core.utils.logger import get_logger, setup_logging

__all__ = [
    # Core
    "TransformerModel",
    # Formats
    "GGUFModelLoader",
    "GGUFTokenizerWrapper",
    "UniversalModelLoader",
    "SmartLoader",
    # Quantization
    "AdaptiveCompressor",
    "CompressionMethod",
    "PruningEngine",
    # Kernels
    "compress_chunk",
    "decompress_chunk",
    "NUMBA_AVAILABLE",
    # Inference
    "InferenceEngine",
    # Training
    "Trainer",
    "IncrementalTrainer",
    # Tokenization
    "BPETokenizer",
    # Storage
    "DatabaseManager",
    # Utils
    "get_logger",
    "setup_logging",
]

