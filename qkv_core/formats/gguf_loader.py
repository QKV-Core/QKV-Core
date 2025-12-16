"""
GGUF Model Loader
Supports loading and running GGUF format models using llama-cpp-python.
Optimized for Qwen2.5-1.5B-Instruct-GGUF model.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

from qkv_core.utils.logger import get_logger

logger = get_logger()


class GGUFTokenizerWrapper:
    """
    Wrapper for GGUF model's built-in tokenizer.
    Makes it compatible with the existing tokenizer interface.
    """
    
    def __init__(self, llama_model):
        """
        Initialize tokenizer wrapper.
        
        Args:
            llama_model: Llama instance from llama-cpp-python
        """
        self.llama_model = llama_model
        self.eos_token_id = llama_model.token_eos() if hasattr(llama_model, 'token_eos') else 151643
        self.bos_token_id = llama_model.token_bos() if hasattr(llama_model, 'token_bos') else 151643
        self.unk_token_id = llama_model.token_unk() if hasattr(llama_model, 'token_unk') else 0
        self.pad_token_id = llama_model.token_pad() if hasattr(llama_model, 'token_pad') else self.eos_token_id
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs"""
        try:
            if isinstance(text, str):
                text_bytes = text.encode('utf-8')
            else:
                text_bytes = text
            
            tokens = self.llama_model.tokenize(text_bytes, add_bos=add_special_tokens)
            return tokens if tokens else []
        except Exception as e:
            logger.warning(f"⚠️ Tokenizer encode error: {e}")
            return []
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        try:
            if not token_ids:
                return ""
            
            # Filter out special tokens if requested
            if skip_special_tokens:
                token_ids = [tid for tid in token_ids if tid not in [self.eos_token_id, self.bos_token_id, self.pad_token_id]]
            
            text_bytes = self.llama_model.detokenize(token_ids)
            if isinstance(text_bytes, bytes):
                return text_bytes.decode('utf-8', errors='ignore')
            return str(text_bytes)
        except Exception as e:
            logger.warning(f"⚠️ Tokenizer decode error: {e}")
            return ""


class GGUFModelLoader:
    """
    Loader for GGUF format models using llama-cpp-python.
    Provides a wrapper to make GGUF models compatible with the existing inference system.
    """
    
    @staticmethod
    def is_available():
        """Check if llama-cpp-python is installed"""
        return LLAMA_CPP_AVAILABLE
    
    @staticmethod
    def load_model(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, verbose: bool = False):
        """
        Load a GGUF model file.
        
        Args:
            model_path: Path to the .gguf model file
            n_ctx: Context window size (default: 2048)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
            verbose: Enable verbose logging
            
        Returns:
            Llama model instance or None if failed
        """
        if not LLAMA_CPP_AVAILABLE:
            logger.error("❌ llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
            return None
        
        model_path = str(model_path)
        if not os.path.exists(model_path):
            logger.error(f"❌ GGUF model file not found: {model_path}")
            return None
        
        if not model_path.endswith('.gguf'):
            logger.error(f"❌ File is not a GGUF model: {model_path}")
            return None
        
        try:
            logger.info(f"🔍 Loading GGUF model: {os.path.basename(model_path)}")
            
            # Auto-detect GPU layers if CUDA is available
            import torch
            if torch.cuda.is_available() and n_gpu_layers == -1:
                # Try to offload all layers to GPU for better performance
                n_gpu_layers = 99  # Large number to offload all layers
                logger.info(f"🚀 CUDA detected, offloading layers to GPU")
            elif n_gpu_layers == -1:
                n_gpu_layers = 0  # CPU only
                logger.info(f"💻 Using CPU mode")
            
            model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                n_threads=None,  # Auto-detect
                n_batch=512,
            )
            
            logger.info(f"✅ GGUF model loaded successfully")
            
            # Create tokenizer wrapper
            tokenizer = GGUFTokenizerWrapper(model)
            
            # Return both model and tokenizer as a tuple for compatibility
            # The model itself contains the tokenizer, but we also provide it separately
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def download_gguf_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF", 
                            file_name: str = "qwen2.5-1.5b-instruct-q3_k_m.gguf",
                            output_dir: str = "model_weights",
                            progress_callback=None):
        """
        Download GGUF model from Hugging Face.
        
        Args:
            model_name: Hugging Face model repository name
            file_name: Specific GGUF file to download (e.g., qwen2.5-1.5b-instruct-q3_k_m.gguf)
            output_dir: Directory to save the model
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            from huggingface_hub import hf_hub_download
            HF_HUB_AVAILABLE = True
        except ImportError:
            logger.error("❌ huggingface_hub is not installed. Install with: pip install huggingface_hub")
            return None
        
        def log(msg):
            logger.info(f"[GGUF-DOWNLOAD] {msg}")
            if progress_callback:
                progress_callback(msg)
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            final_path = output_path / file_name
            
            # Check if file already exists
            if final_path.exists():
                log(f"✅ Model already exists: {final_path}")
                return str(final_path)
            
            log(f"🚀 Downloading GGUF model: {model_name}/{file_name}")
            log(f"📁 Saving to: {final_path}")
            
            downloaded_path = hf_hub_download(
                repo_id=model_name,
                filename=file_name,
                local_dir=str(output_path),
                local_dir_use_symlinks=False,
            )
            
            log(f"✅ Download completed: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            error_msg = f"Download Error: {str(e)}"
            log(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return None


class GGUFModelWrapper:
    """
    Wrapper class to make GGUF models compatible with InferenceEngine.
    Provides a PyTorch-like interface for the llama-cpp-python model.
    """
    
    def __init__(self, llama_model, tokenizer=None):
        """
        Initialize wrapper.
        
        Args:
            llama_model: Llama instance from llama-cpp-python
            tokenizer: Optional tokenizer (if None, creates GGUFTokenizerWrapper)
        """
        self.llama_model = llama_model
        if tokenizer is None:
            self.tokenizer = GGUFTokenizerWrapper(llama_model)
        else:
            self.tokenizer = tokenizer
        self.eval()  # Set to eval mode
    
    def eval(self):
        """Set model to evaluation mode (no-op for GGUF models)"""
        pass
    
    def to(self, device):
        """Device placement (handled by llama-cpp-python internally)"""
        return self
    
    def __call__(self, input_ids, past_key_values=None):
        """
        Forward pass compatible with InferenceEngine.
        
        Note: This is a simplified wrapper. For full compatibility,
        we'll use the GGUF model's native generate method in InferenceEngine.
        """
        # This method is not used when using GGUF models directly
        # The InferenceEngine will detect GGUF models and use their native API
        raise NotImplementedError(
            "GGUF models should use native generate() method, not this wrapper. "
            "Use GGUFInferenceEngine instead."
        )

