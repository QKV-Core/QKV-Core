"""
Utility Functions

This module provides utility functions:
- Logger: Logging utilities
- MemoryUtils: Memory management utilities
- TokenizerLoader: Tokenizer loading utilities
"""

from qkv_core.utils.logger import get_logger, setup_logging

try:
    from qkv_core.utils.memory_utils import cleanup_memory, optimize_for_gtx1050, get_memory_info
except ImportError:
    cleanup_memory = None
    optimize_for_gtx1050 = None
    get_memory_info = None

try:
    from qkv_core.utils.tokenizer_loader import load_tokenizer_with_fallback
except ImportError:
    load_tokenizer_with_fallback = None

__all__ = [
    "get_logger",
    "setup_logging",
    "cleanup_memory",
    "optimize_for_gtx1050",
    "get_memory_info",
    "load_tokenizer_with_fallback",
]
