"""
utils/smart_loader.py
Robust Model Loader v3
(Fix: Explicitly handles mlp.c_proj dimension mismatch for Medium/Large GPT-2 models)
"""

import torch
import os
import pickle
import sys
import re
from pathlib import Path

from qkv_core.core.transformer import TransformerModel
from qkv_core.utils.logger import get_logger

# Try to import GGUF loader
try:
    from qkv_core.formats.gguf_loader import GGUFModelLoader
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    GGUFModelLoader = None

try:
    from config.model_config import ModelConfig
except ImportError:
    class ModelConfig:
        @staticmethod
        def get_custom(vocab_size, d_model, num_layers, num_heads, max_seq_len):
            return {
                "vocab_size": vocab_size, "d_model": d_model,
                "num_layers": num_layers, "num_heads": num_heads,
                "d_ff": 4 * d_model, "max_seq_length": max_seq_len,
                "dropout": 0.1
            }

logger = get_logger()

class SmartLoader:
    
    @staticmethod
    def _infer_config_from_state_dict(state_dict):
        detected_vocab = 50257
        detected_dim = 768
        detected_layers = 6 
        
        try:
            layer_indices = set()
            for key in state_dict.keys():
                match = re.search(r'h\.(\d+)\.', key)
                if match:
                    layer_indices.add(int(match.group(1)))
            
            if layer_indices:
                detected_layers = max(layer_indices) + 1
            
            if 'wte.weight' in state_dict:
                detected_vocab, detected_dim = state_dict['wte.weight'].shape
            elif 'output_projection.weight' in state_dict:
                if state_dict['output_projection.weight'].shape[0] > 20000:
                     detected_vocab, detected_dim = state_dict['output_projection.weight'].shape
            
        except Exception as e:
            logger.warning(f"⚠️ Could not infer config: {e}")
        
        return {
            "vocab_size": detected_vocab,
            "d_model": detected_dim,
            "num_layers": detected_layers,
            "num_heads": 12, 
            "max_seq_length": 1024
        }

    @staticmethod
    def load_tokenizer(path):
        path = str(path)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"❌ Tokenizer load error: {e}")
            return None

    @staticmethod
    def load_model(path, device='cpu'):
        path = str(path)
        if not os.path.exists(path):
            logger.error(f"❌ Model file not found: {path}")
            return None, None

        logger.info(f"🔍 Smart Loading (model): {os.path.basename(path)}")
        device_str = str(device) if isinstance(device, torch.device) else device

        # Check if this is a GGUF model
        if path.endswith('.gguf'):
            if not GGUF_AVAILABLE or not GGUFModelLoader.is_available():
                logger.error("❌ GGUF model detected but llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
                return None, None
            
            logger.info("🔍 Detected GGUF format model")
            result = GGUFModelLoader.load_model(path, n_ctx=2048, n_gpu_layers=-1)
            if result is None:
                return None, None
            
            # GGUFModelLoader returns (model, tokenizer) tuple
            if isinstance(result, tuple):
                gguf_model, gguf_tokenizer = result
            else:
                gguf_model = result
                from qkv_core.formats.gguf_loader import GGUFTokenizerWrapper
                gguf_tokenizer = GGUFTokenizerWrapper(gguf_model)
            
            # Store tokenizer in model for InferenceEngine access
            gguf_model._gguf_tokenizer = gguf_tokenizer
            
            # Return GGUF model with a dummy config
            # The InferenceEngine will handle GGUF models specially
            config = {
                "vocab_size": 151936,  # Qwen2.5 tokenizer vocab size
                "d_model": 1536,  # Qwen2.5-1.5B config
                "num_layers": 24,
                "num_heads": 12,
                "d_ff": 6144,
                "max_seq_length": 2048,
                "dropout": 0.1,
                "model_type": "gguf"
            }
            return gguf_model, config

        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                saved_config = checkpoint.get('config', None)
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
                saved_config = None
            else:
                return None, None

            # Config Setup
            if saved_config:
                final_config = saved_config
            else:
                inferred = SmartLoader._infer_config_from_state_dict(state_dict)
                final_config = ModelConfig.get_custom(
                    vocab_size=inferred['vocab_size'],
                    d_model=inferred['d_model'],
                    num_layers=inferred['num_layers'],
                    num_heads=inferred['num_heads'],
                    max_seq_len=inferred.get('max_seq_length', 1024)
                )

            # Ensure d_ff exists
            if 'd_ff' not in final_config:
                final_config['d_ff'] = 4 * final_config['d_model']

            model = TransformerModel(final_config)
            
            # --- FORMAT DETECTION & CORRECTION ---
            
            # Global Heuristic Check (Previously implemented)
            is_huggingface_format = False
            sample_key = None
            for key in state_dict.keys():
                if "attn.c_attn.weight" in key or "c_attn.weight" in key:
                    sample_key = key
                    break
            
            if sample_key:
                weight = state_dict[sample_key]
                if weight.shape[0] == final_config['d_model'] and weight.shape[1] == 3 * final_config['d_model']:
                    is_huggingface_format = True
                    logger.warning("⚠️ Detected Hugging Face Conv1D format globally.")

            new_state_dict = {}
            transpose_count = 0
            
            for key, value in state_dict.items():
                new_key = key
                if new_key.startswith("transformer."): new_key = new_key.replace("transformer.", "")
                if new_key.startswith("module."): new_key = new_key.replace("module.", "")
                
                should_transpose = False
                
                # 1. Global HF Format check
                if is_huggingface_format and len(value.shape) == 2:
                    if "weight" in new_key and "wte" not in new_key and "wpe" not in new_key and "ln_" not in new_key:
                         should_transpose = True

                # 2. Specific Check for MLP Output Projection (The error you encountered)
                # Checkpoint has [d_ff, d_model] e.g. [4096, 1024]
                # PyTorch needs [d_model, d_ff] e.g. [1024, 4096]
                if "mlp.c_proj.weight" in new_key and not should_transpose:
                    if value.shape == (final_config['d_ff'], final_config['d_model']):
                        logger.info(f"🔧 Fixing dimension mismatch for {new_key}")
                        should_transpose = True

                # 3. Specific Check for MLP Input (c_fc)
                # Checkpoint has [d_model, d_ff] e.g. [1024, 4096] (HF Conv1D: in, out)
                # PyTorch needs [d_ff, d_model] e.g. [4096, 1024]
                if "mlp.c_fc.weight" in new_key and not should_transpose:
                    if value.shape == (final_config['d_model'], final_config['d_ff']):
                        logger.info(f"🔧 Fixing dimension mismatch for {new_key}")
                        should_transpose = True

                if should_transpose:
                    value = value.t()
                    transpose_count += 1

                new_state_dict[new_key] = value

            if transpose_count > 0:
                logger.info(f"✅ Transposed {transpose_count} matrices to fix Linear/Conv1D mismatch.")

            # Output Projection Fix
            if "output_projection.weight" not in new_state_dict:
                wte_key = next((k for k in new_state_dict if "wte.weight" in k), None)
                if wte_key:
                    new_state_dict["output_projection.weight"] = new_state_dict[wte_key]

            # Load
            model.load_state_dict(new_state_dict, strict=False)
            model.to(device_str)
            model.eval()
            
            return model, final_config

        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            # Detaylı hata göster ki hangi katmanda patladığını görelim
            import traceback
            traceback.print_exc()
            return None, None