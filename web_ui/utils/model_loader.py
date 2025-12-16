"""
Model loading utilities.
"""
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from qkv_core.formats.smart_loader import SmartLoader
from qkv_core.utils.logger import get_logger


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'CPU', logger=None):
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('CPU' or 'CUDA')
        logger: Optional logger instance
        
    Returns:
        Tuple of (model, config, metadata) or (None, None, None) on error
    """
    try:
        if logger:
            logger.info(f"Loading model: {checkpoint_path}")
        else:
            _logger = get_logger()
            _logger.info(f"Loading model: {checkpoint_path}")
            
        model, config = SmartLoader.load_model(checkpoint_path, device=device)
        
        if model is None:
            if logger:
                logger.error("SmartLoader failed to load model.")
            else:
                _logger = get_logger()
                _logger.error("SmartLoader failed to load model.")
            return None, None, None
            
        return model, config, {}
        
    except Exception as e:
        if logger:
            logger.error(f"model loading error: {e}")
        else:
            _logger = get_logger()
            _logger.error(f"model loading error: {e}")
        return None, None, None
