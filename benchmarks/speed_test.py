"""
Speed Test Benchmarks - Performance Proofs

This module provides performance benchmarks to demonstrate the effectiveness
of QKV Core's optimizations. Critical for visa assessment as it shows
measurable improvements in speed and efficiency.
"""

import time
import torch
from typing import Dict, Any
from pathlib import Path


def benchmark_inference_speed(model, tokenizer, prompt: str, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark inference speed in tokens per second.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        prompt: Input prompt for generation
        num_runs: Number of runs to average
    
    Returns:
        Dictionary with performance metrics:
        - tokens_per_second: Average generation speed
        - total_time: Total time for all runs
        - avg_time_per_run: Average time per run
    """
    from qkv_core.inference.inference import InferenceEngine
    
    engine = InferenceEngine(model, tokenizer)
    
    total_tokens = 0
    total_time = 0.0
    
    for _ in range(num_runs):
        start_time = time.time()
        tokens = list(engine.generate_stream(prompt, max_length=100))
        elapsed = time.time() - start_time
        
        total_tokens += len(tokens)
        total_time += elapsed
    
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0
    
    return {
        "tokens_per_second": avg_tokens_per_second,
        "total_time": total_time,
        "avg_time_per_run": total_time / num_runs,
        "total_tokens": total_tokens
    }


def benchmark_compression_ratio(original_size: int, compressed_size: int) -> Dict[str, float]:
    """
    Calculate compression ratio metrics.
    
    Args:
        original_size: Original size in bytes
        compressed_size: Compressed size in bytes
    
    Returns:
        Dictionary with compression metrics:
        - compression_ratio: Ratio of compressed to original
        - space_saved_percent: Percentage of space saved
        - space_saved_mb: Space saved in megabytes
    """
    compression_ratio = compressed_size / original_size if original_size > 0 else 0.0
    space_saved = original_size - compressed_size
    space_saved_percent = (space_saved / original_size * 100) if original_size > 0 else 0.0
    space_saved_mb = space_saved / (1024 * 1024)
    
    return {
        "compression_ratio": compression_ratio,
        "space_saved_percent": space_saved_percent,
        "space_saved_mb": space_saved_mb,
        "original_size_mb": original_size / (1024 * 1024),
        "compressed_size_mb": compressed_size / (1024 * 1024)
    }


if __name__ == "__main__":
    print("🚀 QKV Core Speed Test Benchmarks")
    print("=" * 60)
    print("This module provides performance benchmarks for QKV Core.")
    print("Run specific benchmarks using the functions above.")

