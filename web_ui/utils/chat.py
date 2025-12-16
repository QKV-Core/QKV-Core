"""
Chat Module for Web UI - Fixed Version
Uses InferenceEngine to match debug_chat.py performance.
"""
import torch
import sys
import os
from pathlib import Path

# Add project path
# Updated path calculation for utils/ subdirectory
_current_file = Path(__file__).resolve()
_utils_dir = _current_file.parent
_web_ui_dir = _utils_dir.parent
_project_root = _web_ui_dir.parent

# Ensure project root is in path for absolute imports
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qkv_core.formats.smart_loader import SmartLoader
from qkv_core.inference.inference import InferenceEngine

# --- SETTINGS ---
# Updated paths for utils/ subdirectory
MODEL_PATH = _project_root / "model_weights" / "distilgpt2.pt"
TOKENIZER_PATH = _project_root / "tokenizer" / "distilgpt2.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
model = None
tokenizer = None
engine = None
is_initialized = False

def initialize_model():
    """Load model and tokenizer using SmartLoader and setup InferenceEngine"""
    global model, tokenizer, engine, is_initialized

    # 1. Try to grab from Web UI global cache first (to save memory)
    try:
        import sys
        if 'web_ui.app' in sys.modules:
            app_module = sys.modules['web_ui.app']
            if hasattr(app_module, '_ui_model_loaded') and app_module._ui_model_loaded:
                # Using shared model from Web UI (no logging needed for UI components)
                model = app_module._ui_model
                tokenizer = app_module._ui_tokenizer
                
                # Setup Engine
                engine = InferenceEngine(model, tokenizer, device=DEVICE)
                is_initialized = True
                return True
    except:
        pass

    if is_initialized and engine is not None:
        return True

    # 2. Load manually if not in cache
    try:
        # Loading model (UI components handle their own logging)
        model, config = SmartLoader.load_model(str(MODEL_PATH), device=DEVICE)
        if model is None:
            return False

        tokenizer = SmartLoader.load_tokenizer(str(TOKENIZER_PATH))
        if tokenizer is None:
            return False

        # Initialize Inference Engine (Critically important for stability)
        engine = InferenceEngine(model, tokenizer, device=DEVICE)
        is_initialized = True
        return True

    except Exception as e:
        # Error will be handled by UI
        return False

def process_message(user_input, stream_callback=None):
    """
    Process message using InferenceEngine (Matches debug_chat.py logic).
    """
    global engine, is_initialized

    if not is_initialized or engine is None:
        if not initialize_model():
            return "❌ Model could not be loaded!"

    # Critical: Prompt template for chat mode
    # Base models require this format for proper chat functionality
    prompt = f"User: {user_input}\nAI:"

    # Settings from debug_chat.py
    gen_kwargs = {
        "max_length": 100,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "use_kv_cache": True,
        "method": "sample"
    }

    generated_text = ""
    
    try:
        # Use the robust generate_stream from InferenceEngine
        for token_chunk in engine.generate_stream(prompt, **gen_kwargs):
            generated_text += token_chunk
            if stream_callback:
                stream_callback(token_chunk)
                
        return generated_text.strip()
        
    except Exception as e:
        return f"❌ Generation Error: {str(e)}"

if __name__ == "__main__":
    # Test block
    print("Chat Module Test")
    if initialize_model():
        while True:
            u_in = input("\nYou: ")
            if u_in.lower() == 'q': break
            
            print("AI: ", end="", flush=True)
            process_message(u_in, stream_callback=lambda x: print(x, end="", flush=True))
            print()
