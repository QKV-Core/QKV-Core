"""
Performance-Focused Kernels

This module contains low-level, performance-optimized code for critical operations.
Includes Numba JIT compilation and CUDA kernels for maximum performance.

Why separate kernels?
- Performance-critical code needs special optimization
- JIT compilation (Numba) provides near C/C++ speeds
- CUDA kernels enable GPU acceleration
- Clear separation of high-performance code from business logic
"""

try:
    from qkv_core.kernels.numba_engine import compress_chunk, decompress_chunk
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    compress_chunk = None
    decompress_chunk = None

__all__ = [
    "compress_chunk",
    "decompress_chunk",
    "NUMBA_AVAILABLE",
]

