"""
Helper utility functions for the web UI.
"""
from pathlib import Path


def get_available_tokenizers():
    """
    Get list of available tokenizer files.
    
    Returns:
        List of tokenizer filenames
    """
    tokenizer_dir = Path("tokenizer")
    if not tokenizer_dir.exists():
        return []
    return [f.name for f in tokenizer_dir.glob("*.pkl")]


def get_available_checkpoints():
    """
    Get list of available model checkpoint files.
    Includes .pt (PyTorch) and .gguf (GGUF format) files.
    
    Returns:
        List of checkpoint filenames
    """
    weights_dir = Path("model_weights")
    if not weights_dir.exists():
        return []
    # Get .pt and .gguf files
    pt_files = [f.name for f in weights_dir.glob("*.pt")]
    gguf_files = [f.name for f in weights_dir.glob("*.gguf")]
    return pt_files + gguf_files


def format_time(seconds):
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1.5s", "2.3m", "1.2h")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
