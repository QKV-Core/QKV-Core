"""
Debug Chat Terminal Interface

Interactive command-line interface for testing QKV Core models.
Provides a simple chat interface for debugging and testing inference.
"""

import sys
import os
import torch
from pathlib import Path

# Environment setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qkv_core.formats.smart_loader import SmartLoader
from qkv_core.inference.inference import InferenceEngine
from qkv_core.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging("logs")
logger = get_logger()


def main():
    """
    Main function for debug chat terminal.
    
    Provides interactive model selection and chat interface for testing
    QKV Core models in both PyTorch and GGUF formats.
    """
    logger.info("=" * 60)
    logger.info("QKV Core - Debug Chat Terminal")
    logger.info("   Query-Key-Value Core - The Core of Transformer Intelligence")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model_dir = Path("model_weights")
    if not model_dir.exists():
        logger.error("'model_weights' directory not found!")
        return

    checkpoints = list(model_dir.glob("*.pt"))
    gguf_models = list(model_dir.glob("*.gguf"))
    all_models = checkpoints + gguf_models
    
    if not all_models:
        logger.error("No model files found (.pt or .gguf)!")
        return

    # If multiple models found, let user choose
    if len(all_models) > 1:
        logger.info("\nAvailable Models:")
        for i, model in enumerate(all_models, 1):
            model_type = "GGUF" if model.suffix == '.gguf' else "PyTorch"
            logger.info(f"  {i}. {model.name} ({model_type})")
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(all_models)}) or press Enter for first: ").strip()
                if not choice:
                    default_model = all_models[0]
                    break
                choice_num = int(choice)
                if 1 <= choice_num <= len(all_models):
                    default_model = all_models[choice_num - 1]
                    break
                else:
                    logger.warning(f"Please enter a number between 1 and {len(all_models)}")
            except ValueError:
                logger.warning("Please enter a valid number")
            except KeyboardInterrupt:
                logger.info("\nExiting...")
                return
    else:
        default_model = all_models[0]

    if default_model.suffix == '.gguf':
        model_type = "GGUF"
    else:
        model_type = "PyTorch"
    logger.info(f"\nSelected Model: {default_model.name} ({model_type})")

    logger.info("Loading model (SmartLoader)...")
    try:
        model, config = SmartLoader.load_model(str(default_model), device=device)
        if model is None:
            logger.error("Failed to load model")
            return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # For GGUF models, tokenizer is built-in
    if default_model.suffix == '.gguf':
        logger.info("GGUF model detected - using built-in tokenizer")
        # GGUF models have tokenizer built-in, InferenceEngine will handle it
        tokenizer = None  # Will be created by InferenceEngine
    else:
        tokenizer_name = default_model.stem + ".pkl"
        tokenizer_path = Path("tokenizer") / tokenizer_name
        
        if not tokenizer_path.exists():
            tokenizers = list(Path("tokenizer").glob("*.pkl"))
            if tokenizers:
                tokenizer_path = tokenizers[0]
            else:
                logger.error("No tokenizer found!")
                return

        logger.info(f"Loading tokenizer: {tokenizer_path.name}")
        tokenizer = SmartLoader.load_tokenizer(str(tokenizer_path))
    
    engine = InferenceEngine(model, tokenizer, device=device)
    logger.info("Engine ready! Starting chat session...")
    logger.info("-" * 60)

    # Chat history can be maintained here if needed
    # chat_history = ""

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                logger.info("Exiting...")
                break
            
            if not user_input.strip():
                continue

            print("AI: ", end="", flush=True)

            # Critical: Prompt template for chat mode
            # Base models require "User:" and "AI:" labels for proper chat formatting
            prompt = f"User: {user_input}\nAI:"

            gen_kwargs = {
                "max_length": 2048,  # High default (model will stop automatically)
                "temperature": 0.6,  # Slightly reduced for more consistent responses
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "use_kv_cache": True,
                "method": "sample"
            }

            response_received = False
            for token in engine.generate_stream(prompt, **gen_kwargs):
                if token:
                    print(token, end="", flush=True)
                    response_received = True
            
            if not response_received:
                logger.warning("No response generated - this might indicate a GGUF stream format issue")
            
            print()

        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
