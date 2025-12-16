import torch
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from qkv_core.tokenization.bpe import BPETokenizer
from qkv_core.utils.logger import get_logger

logger = get_logger()

class HuggingFaceConverter:
    
    RECOMMENDED_MODELS = {
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF": {
            "description": "Qwen2.5-1.5B-Instruct-GGUF - Optimized GGUF Format (Turkish & English, Great Coding)",
            "size": "1.5B (GGUF Q3_K_M)",
            "params": 1_500_000_000,
            "download_size_mb": 800,
            "license": "Apache 2.0",
            "model_type": "gguf",
            "file_name": "qwen2.5-1.5b-instruct-q3_k_m.gguf",
            "note": "🚀 GGUF format - Optimized for GTX 1050! Q3_K_M quantization for even better efficiency. Only ~0.8GB VRAM usage. Supports 100+ languages including excellent Turkish and English. Outstanding coding capabilities. Perfect for multilingual conversations, code generation, and general-purpose tasks. Uses llama-cpp-python for fast inference."
        },
        "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF": {
            "description": "Qwen2.5-Coder-3B-Instruct-GGUF - Specialized Code Generation Model (GGUF Format, 3B) - Champion Model",
            "size": "3B (GGUF Q5_K_M)",
            "params": 3_000_000_000,
            "download_size_mb": 2400,
            "license": "Apache 2.0",
            "model_type": "gguf",
            "file_name": "qwen2.5-coder-3b-instruct-q5_k_m.gguf",
            "note": "🏆 Champion Model - Best for GTX 1050 4GB! Specialized for code generation and programming tasks. Q5_K_M quantization provides excellent balance between speed and quality. Excellent at understanding and generating code in multiple programming languages. Based on Qwen2.5 architecture with 3B parameters. Great for code completion, debugging, and technical documentation. Requires ~2.4GB VRAM - perfect fit for 4GB cards with Windows overhead. Fast, intelligent, and reliable. Uses llama-cpp-python for fast inference."
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF": {
            "description": "Qwen2.5-Coder-7B-Instruct-GGUF - Specialized Code Generation Model (GGUF Format, 7B)",
            "size": "7B (GGUF Q3_K_M)",
            "params": 7_000_000_000,
            "download_size_mb": 3200,
            "license": "Apache 2.0",
            "model_type": "gguf",
            "file_name": "qwen2.5-coder-7b-instruct-q3_k_m.gguf",
            "note": "💻 Specialized for code generation and programming tasks! GGUF format optimized for efficiency. Q3_K_M quantization for better resource efficiency. Excellent at understanding and generating code in multiple programming languages. Based on Qwen2.5 architecture with 7B parameters (quantized to Q3_K_M). Great for code completion, debugging, and technical documentation. Requires ~3.2GB VRAM (GGUF optimized). Best for developers and advanced coding tasks. Uses llama-cpp-python for fast inference."
        },
        "microsoft/Phi-3-mini-4k-instruct-gguf": {
            "description": "Phi-3-mini-4k-instruct-GGUF - Microsoft's Efficient Small Model (GGUF Format)",
            "size": "3.8B (GGUF Q4)",
            "params": 3_800_000_000,
            "download_size_mb": 2400,
            "license": "MIT",
            "model_type": "gguf",
            "file_name": "Phi-3-mini-4k-instruct-q4.gguf",
            "note": "🚀 Microsoft's Phi-3-mini model in GGUF format! Excellent instruction-following and reasoning capabilities. Q4 quantization provides good balance between quality and efficiency. Great for general-purpose tasks, coding, and conversations. Requires ~2.4GB VRAM. Optimized for speed and quality. Perfect balance between performance and resource usage."
        },
        "gpt2": {
            "description": "GPT-2 Small (124M)",
            "size": "124M",
            "params": 124_000_000,
            "download_size_mb": 500,
            "license": "MIT"
        },
        "distilgpt2": {
            "description": "DistilGPT-2 (82M)",
            "size": "82M",
            "params": 82_000_000,
            "download_size_mb": 350,
            "license": "Apache 2.0"
        },
        "microsoft/DialoGPT-small": {
            "description": "DialoGPT Small - Chat Optimized (117M)",
            "size": "117M",
            "params": 117_000_000,
            "download_size_mb": 450,
            "license": "MIT"
        },
    }

    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("This module requires the 'transformers' library.")
            
        self.weights_dir = Path("model_weights")
        self.tokenizer_dir = Path("tokenizer")
        self.weights_dir.mkdir(exist_ok=True)
        self.tokenizer_dir.mkdir(exist_ok=True)

    def download_and_convert(self, model_name: str, output_name: str = None, progress_callback=None) -> Dict[str, Any]:
        
        if output_name is None:
            output_name = model_name.split("/")[-1].replace("-", "_")
            
        def log(msg):
            logger.info(f"[HF-CONVERT] {msg}")
            if progress_callback:
                progress_callback(msg)

        try:
            log(f"🚀 Download Started: {model_name}")
            
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            log("✅ Download completed. Starting conversion...")

            config = hf_model.config
            d_model = getattr(config, 'n_embd', getattr(config, 'hidden_size', 768))
            num_layers = getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 12))
            num_heads = getattr(config, 'n_head', getattr(config, 'num_attention_heads', 12))
            max_seq_length = getattr(config, 'n_positions', getattr(config, 'max_position_embeddings', 1024))
            dropout = getattr(config, 'attn_pdrop', getattr(config, 'hidden_dropout_prob', 0.1))

            model_config = {
                'vocab_size': config.vocab_size,
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'd_ff': d_model * 4,
                'max_seq_length': max_seq_length,
                'dropout': dropout
            }
            
            log(f"📊 Detected Config: d_model={d_model}, layers={num_layers}")

            new_state_dict = {}
            old_sd = hf_model.state_dict()

            for key, value in old_sd.items():
                new_key = key
                prefixes_to_remove = ["transformer.", "gpt_neox.", "base_model."]
                for prefix in prefixes_to_remove:
                    if new_key.startswith(prefix):
                        new_key = new_key.replace(prefix, "")
                        break
                
                if "lm_head" in new_key or "embed_out" in new_key:
                    new_key = "output_projection.weight"
                
                if "layers." in new_key:
                    new_key = new_key.replace("layers.", "h.")
                
                # Attention and MLP mapping
                if "attention.query_key_value" in new_key: new_key = new_key.replace("attention.query_key_value", "attn.c_attn")
                if "attention.dense" in new_key: new_key = new_key.replace("attention.dense", "attn.c_proj")
                if "mlp.dense_h_to_4h" in new_key: new_key = new_key.replace("mlp.dense_h_to_4h", "mlp.c_fc")
                if "mlp.dense_4h_to_h" in new_key: new_key = new_key.replace("mlp.dense_4h_to_h", "mlp.c_proj")

                # CRITICAL FIX: Always transpose Conv1D weights to match nn.Linear
                # This applies to c_attn, c_proj, and c_fc weights.
                # In HF GPT-2, these are Conv1D (in, out). PyTorch Linear expects (out, in).
                # Even if square (like c_proj), we MUST transpose.
                if any(x in key for x in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                    if value.dim() == 2:
                        value = value.t()
                        # log(f"   🔄 Transposed {new_key} {value.t().shape} -> {value.shape}")
                    
                new_state_dict[new_key] = value

            if "output_projection.weight" not in new_state_dict:
                wte_keys = ["wte.weight", "embed_in.weight"]
                found_wte = None
                for k in wte_keys:
                    if k in new_state_dict:
                        found_wte = new_state_dict[k]
                        break
                if found_wte is not None:
                    new_state_dict["output_projection.weight"] = found_wte.clone()

            checkpoint = {'model_state_dict': new_state_dict, 'config': model_config}
            save_path_pt = self.weights_dir / f"{output_name}.pt"
            torch.save(checkpoint, save_path_pt)
            log(f"✅ Model saved successfully: {save_path_pt}")

            log("🔤 Converting tokenizer (Adapter Mode)...")
            
            try:
                from qkv_core.tokenization.bpe import HuggingFaceTokenizerAdapter
                tokenizer_wrapper = HuggingFaceTokenizerAdapter(hf_tokenizer)
                log("   ✨ Using HuggingFaceTokenizerAdapter.")
            except ImportError:
                class TemporaryAdapter:
                    def __init__(self, hf_tok):
                        self._tokenizer = hf_tok
                        self.vocabulary = hf_tok.get_vocab()
                        self.reverse_vocab = {v: k for k, v in self.vocabulary.items()}
                        self.eos_token_id = hf_tok.eos_token_id or 50256
                        self.bos_token_id = hf_tok.bos_token_id or 50256
                        self.unk_token_id = hf_tok.unk_token_id or 50256
                        self.pad_token_id = hf_tok.pad_token_id or 50256
                        self.is_trained = True
                        self.merges = []
                        self.from_huggingface = True
                        
                    def encode(self, text, add_special_tokens=True):
                        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
                    def decode(self, token_ids, skip_special_tokens=True):
                        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                    def get_vocab_size(self):
                        return self._tokenizer.vocab_size

                tokenizer_wrapper = TemporaryAdapter(hf_tokenizer)
                log("   ⚠️ Adapter class not found, temporary adapter created.")

            save_path_tok = self.tokenizer_dir / f"{output_name}.pkl"
            
            with open(save_path_tok, 'wb') as f:
                pickle.dump(tokenizer_wrapper, f)
                
            log(f"✅ Tokenizer saved successfully: {save_path_tok}")
            log(f"🎉 COMPLETE! You can now select '{output_name}' in the Chat tab.")

            return {
                "checkpoint_path": str(save_path_pt),
                "tokenizer_path": str(save_path_tok),
                "config": model_config
            }

        except Exception as e:
            error_msg = f"Conversion Error: {str(e)}"
            log(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            raise Exception(error_msg)