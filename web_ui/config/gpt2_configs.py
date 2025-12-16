"""
GPT-2 model configurations and compatibility utilities.
"""
from typing import Optional


GPT2_CONFIGS = {
    "gpt2": {
        "vocab_size": 50257,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "max_seq_length": 1024,
        "dropout": 0.1
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
        "max_seq_length": 1024,
        "dropout": 0.1
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
        "max_seq_length": 1024,
        "dropout": 0.1
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
        "max_seq_length": 1024,
        "dropout": 0.1
    },
    "distilgpt2": {
        "vocab_size": 50257,
        "d_model": 768,
        "num_layers": 6,
        "num_heads": 12,
        "d_ff": 3072,
        "max_seq_length": 1024,
        "dropout": 0.1
    }
}


def detect_gpt2_model_type(checkpoint_config: dict) -> Optional[str]:
    """
    Detect GPT-2 model type from checkpoint configuration.
    
    Args:
        checkpoint_config: Dictionary containing model configuration
        
    Returns:
        Model type string (e.g., "gpt2", "gpt2-medium") or None if not detected
    """
    d_model = checkpoint_config.get('d_model')
    num_layers = checkpoint_config.get('num_layers')
    num_heads = checkpoint_config.get('num_heads')
    d_ff = checkpoint_config.get('d_ff')
    vocab_size = checkpoint_config.get('vocab_size')
    
    if vocab_size != 50257:
        return None
    
    for model_type, config in GPT2_CONFIGS.items():
        if (d_model == config['d_model'] and 
            num_layers == config['num_layers'] and
            num_heads == config['num_heads'] and
            d_ff == config['d_ff']):
            return model_type
    
    if d_model == 768:
        if num_layers == 12:
            return "gpt2"
        elif num_layers == 6:
            return "distilgpt2"
    elif d_model == 1024 and num_layers == 24:
        return "gpt2-medium"
    elif d_model == 1280 and num_layers == 36:
        return "gpt2-large"
    elif d_model == 1600 and num_layers == 48:
        return "gpt2-xl"
    
    return None


def get_gpt2_compatible_config(checkpoint_config: dict, detected_type: Optional[str] = None) -> dict:
    """
    Get GPT-2 compatible configuration from checkpoint config.
    
    Args:
        checkpoint_config: Original checkpoint configuration
        detected_type: Optional detected model type
        
    Returns:
        GPT-2 compatible configuration dictionary
    """
    from qkv_core.utils.logger import get_logger
    _logger = get_logger()
    
    if detected_type and detected_type in GPT2_CONFIGS:
        gpt2_config = GPT2_CONFIGS[detected_type].copy()
        return gpt2_config
    
    config = checkpoint_config.copy()
    
    if config.get('vocab_size') != 50257:
        _logger.warning(f"⚠️  vocabulary size mismatch: {config.get('vocab_size')} != 50257 (GPT-2 standard)")
        if abs(config.get('vocab_size', 0) - 50257) < 10:
            config['vocab_size'] = 50257
    
    d_model = config.get('d_model', 768)
    expected_d_ff = 4 * d_model
    if config.get('d_ff') != expected_d_ff:
        _logger.warning(f"⚠️  d_ff mismatch: {config.get('d_ff')} != {expected_d_ff} (expected 4 * d_model)")
        config['d_ff'] = expected_d_ff
    
    expected_num_heads = d_model // 64
    if config.get('num_heads') != expected_num_heads:
        _logger.warning(f"⚠️  num_heads mismatch: {config.get('num_heads')} != {expected_num_heads} (expected d_model / 64)")
        config['num_heads'] = expected_num_heads
    
    return config
