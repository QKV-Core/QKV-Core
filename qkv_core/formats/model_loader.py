# utils/model_loader.py
"""
Universal model and tokenizer loader with registry integration.
Supports both custom trained models and HuggingFace models.
"""

import torch
import sys
from pathlib import Path

# Add project root to path for external dependencies
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from model_registry import ModelRegistry
except ImportError:
    ModelRegistry = None

from qkv_core.core.transformer import TransformerModel
from qkv_core.tokenization.bpe import BPETokenizer
from qkv_core.formats.huggingface_converter import HuggingFaceConverter

class UniversalModelLoader:
    """
    Loads models and tokenizers from registry entries.
    Handles both custom models and HuggingFace conversions.
    """
    
    def __init__(self):
        if ModelRegistry is None:
            raise ImportError("ModelRegistry is not available. Install model_registry package.")
        self.registry = ModelRegistry()
        self.hf_converter = HuggingFaceConverter()
    
    def load_model(self, model_id: str, device: str = None):
        """Load model from registry entry"""
        model_entry = self.registry.get_model(model_id)
        if not model_entry:
            raise ValueError(f"model {model_id} not found in registry")
        
        model_path = model_entry.get('model_path')
        model_type = model_entry.get('model_type', '')
        
        if not device:
            device = 'CUDA' if torch.CUDA.is_available() else 'CPU'
        
        # Custom model loading
        if model_type in ['incremental-finetune', 'custom']:
            return self._load_custom_model(model_path, device)
        
        # HuggingFace model loading
        elif model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'distilgpt2']:
            return self._load_hf_model(model_path, device)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_custom_model(self, model_path: str, device: str):
        """Load custom trained model with shape mismatch fix"""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            config = checkpoint.get('config', {})
            
            model = TransformerModel(config)
            
            # Shape mismatch correction - transpose weights from HF model
            state_dict = checkpoint['model_state_dict']
            fixed_state_dict = {}
            
            from qkv_core.utils.logger import get_logger
            logger = get_logger()
            
            for key, weight in state_dict.items():
                if 'mlp.c_proj.weight' in key and weight.dim() == 2:
                    # HF format: [out_features, in_features] -> Our format: [in_features, out_features]
                    if weight.shape[0] > weight.shape[1]:  # If in HF format
                        fixed_weight = weight.t()  # Transpose
                        logger.info(f"Shape fixed: {key} {weight.shape} -> {fixed_weight.shape}")
                        fixed_state_dict[key] = fixed_weight
                    else:
                        fixed_state_dict[key] = weight
                else:
                    fixed_state_dict[key] = weight
            
            # Load model (strict=False to tolerate missing keys)
            model.load_state_dict(fixed_state_dict, strict=False)
            model.to(device)
            model.eval()
            
            # Check for missing keys
            model_state = model.state_dict()
            missing_keys = [key for key in model_state.keys() if key not in fixed_state_dict]
            if missing_keys:
                logger.warning(f"Missing keys (default values will be used): {missing_keys}")
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load custom model {model_path}: {e}")
    
    def _load_hf_model(self, model_path: str, device: str):
        """Load HuggingFace model (converted or original)"""
        try:
            # Try loading as converted model first
            if model_path.endswith('.pt'):
                return self._load_custom_model(model_path, device)
            
            # Fallback to direct HF loading
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_path)
            model.to(device)
            model.eval()
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model {model_path}: {e}")
    
    def load_tokenizer(self, model_id: str):
        """Load tokenizer from registry entry"""
        model_entry = self.registry.get_model(model_id)
        if not model_entry:
            raise ValueError(f"model {model_id} not found in registry")
        
        # Custom tokenizer loading
        tokenizer_path = model_entry.get('tokenizer_path')
        if tokenizer_path and tokenizer_path.endswith('.pkl'):
            return BPETokenizer.load(tokenizer_path)
        
        # HuggingFace tokenizer fallback
        try:
            from transformers import GPT2Tokenizer
            return GPT2Tokenizer.from_pretrained('gpt2')
        except:
            raise RuntimeError("No suitable tokenizer found")
    
    def get_available_models(self):
        """Get list of all available models from registry"""
        models = self.registry.list_models()
        return [model['model_id'] for model in models]

# Global loader instance for easy access (lazy initialization)
_loader = None

def get_loader():
    """Get or create the global loader instance."""
    global _loader
    if _loader is None:
        _loader = UniversalModelLoader()
    return _loader
