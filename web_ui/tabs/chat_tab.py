"""
Chat tab for model inference and conversation.
"""
from pathlib import Path
import torch
import gradio as gr

from qkv_core.formats.smart_loader import SmartLoader
from ..utils.helpers import get_available_checkpoints, get_available_tokenizers
from qkv_core.inference.inference import InferenceEngine
from ..state.app_state import state
from qkv_core.utils.logger import get_logger

logger = get_logger()

# Global UI model variables (legacy support)
_ui_model = None
_ui_tokenizer = None
_ui_device = None
_ui_model_loaded = False


def create_chat_tab():
    """Create the chat interface tab."""
    
    with gr.Tab("💬 Chat"):
        gr.Markdown("# Chat Interface")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")
                
                with gr.Accordion("📖 Checkpoint Explanation", open=False):
                    gr.Markdown("""
**What are checkpoints?**

Checkpoints are saved model states during or after training. They contain:
- Model weights (learned parameters)
- Training configuration
- Optimizer state (for resuming training)

**Model Formats:**
- **`.pt` files** - PyTorch format (requires separate tokenizer)
  - `best_model.pt` - Best performing model during training
  - `final_model.pt` - Final model after all epochs
  - `checkpoint_epoch_X.pt` - Model saved at epoch X

- **`.gguf` files** - GGUF format (optimized, built-in tokenizer)
  - Optimized for low VRAM (e.g., GTX 1050)
  - Built-in tokenizer (no separate tokenizer file needed)
  - Example: `qwen2.5-1.5b-instruct-q3_k_m.gguf`

**How to use:**
1. Download a pre-trained model OR train your own
2. Select the checkpoint file from the dropdown
3. For `.pt` models: Select the matching tokenizer
4. For `.gguf` models: Tokenizer is built-in (skip this step)
5. Click "Load Model" to start chatting!
                    """)
                
                with gr.Row():
                    chat_checkpoint = gr.Dropdown(
                        choices=get_available_checkpoints(),
                        label="Select Model Checkpoint",
                        value=None,
                        scale=4,
                        info="Select trained model checkpoint (best_model.pt recommended)"
                    )
                    refresh_checkpoints_btn = gr.Button("🔄 Refresh", scale=1, size="sm")
                
                with gr.Row():
                    chat_tokenizer = gr.Dropdown(
                        choices=get_available_tokenizers(),
                        label="Select Tokenizer",
                        value=None,
                        scale=4,
                        info="⚠️ Required for .pt models. Not needed for .gguf models (built-in tokenizer)"
                    )
                    refresh_tokenizers_btn = gr.Button("🔄 Refresh", scale=1, size="sm")
                
                load_model_btn = gr.Button("🚀 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", lines=3, interactive=False)

                gr.Markdown("### Generation Settings")

                gen_method = gr.Radio(
                    choices=["greedy", "sample", "beam"],
                    value="greedy",
                    label="Generation Method",
                    info="Greedy: Most consistent (best for poorly trained models). Sample: More creative. Beam: Better quality but slower."
                )
                
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature", info="Lower = more consistent. 0.5-0.7 recommended for fine-tuned models.")
                max_length = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=2048,  # Varsayılanı yüksek tutuyoruz (Otomatik durması için)
                    step=64,
                    label="Max Token Limit",
                    info="Bu bir üst sınırdır. Model cevabı bittiğinde (ister 1 kelime, ister 1 sayfa olsun) kendisi otomatik duracaktır."
                )
                top_k = gr.Slider(0, 100, value=50, step=5, label="Top-K")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-P")
                repetition_penalty = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Repetition Penalty", info="Higher = less repetition. 1.5-2.0 recommended to prevent loops.")
                
                clear_btn = gr.Button("🗑️ Clear Chat")
            
            with gr.Column(scale=2):
                # Using dictionary format (required by newer Gradio versions)
                chatbot = gr.Chatbot(
                    label="AI Chat",
                    height=500
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message and press Enter to send...",
                        lines=1,
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1)
        
        def load_chat_model(checkpoint_path, tokenizer_name=None):
            global _ui_model, _ui_tokenizer, _ui_device, _ui_model_loaded
            
            try:
                if _ui_device is None:
                    _ui_device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if not Path(checkpoint_path).is_absolute() and not checkpoint_path.startswith('model_weights/'):
                    checkpoint_path = f"model_weights/{checkpoint_path}"
                
                # Check if this is a GGUF model
                is_gguf = checkpoint_path.endswith('.gguf')
                
                # For GGUF models, tokenizer is built-in (optional)
                # For .pt models, tokenizer is required
                if not is_gguf:
                    if tokenizer_name and not tokenizer_name.startswith('tokenizer/'):
                        tokenizer_path = f"tokenizer/{tokenizer_name}"
                    else:
                        tokenizer_path = None
                    
                    if not tokenizer_path:
                        return "❌ Tokenizer is required for PyTorch models (.pt files). Please select a tokenizer."
                else:
                    # GGUF models have built-in tokenizer
                    tokenizer_path = None
                    if tokenizer_name:
                        logger.info("⚠️ GGUF model detected - tokenizer is built-in, ignoring selected tokenizer")

                loaded_model, config = SmartLoader.load_model(checkpoint_path, _ui_device)
                if loaded_model is None:
                    return "❌ Model could not be loaded (File corrupted or incompatible)."
                
                _ui_model = loaded_model
                _ui_model_loaded = True

                # Load tokenizer for non-GGUF models
                if not is_gguf:
                    if tokenizer_path:
                        loaded_tokenizer = SmartLoader.load_tokenizer(tokenizer_path)
                        if loaded_tokenizer:
                            _ui_tokenizer = loaded_tokenizer
                        else:
                            return "❌ Tokenizer could not be loaded."
                    else:
                        return "❌ Tokenizer is required for PyTorch models."
                    
                    return f"✅ Model and tokenizer successfully loaded! ({_ui_device})"
                else:
                    # For GGUF models, tokenizer is stored in the model
                    if hasattr(loaded_model, '_gguf_tokenizer'):
                        _ui_tokenizer = loaded_model._gguf_tokenizer
                    else:
                        # Create tokenizer wrapper if not already created
                        try:
                            from qkv_core.formats.gguf_loader import GGUFTokenizerWrapper
                            _ui_tokenizer = GGUFTokenizerWrapper(loaded_model)
                        except Exception as e:
                            logger.warning(f"⚠️ Could not create GGUF tokenizer wrapper: {e}")
                            _ui_tokenizer = None
                    
                    return f"✅ GGUF model successfully loaded! ({_ui_device})\n💡 Note: GGUF models use built-in tokenizer, no separate tokenizer file needed."

            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"❌ System Error: {str(e)}"

        def refresh_checkpoints():
            return gr.update(choices=get_available_checkpoints())
        
        def refresh_tokenizers():
            return gr.update(choices=get_available_tokenizers())
        
        def add_user_message(user_message, history):
            if not user_message:
                return gr.update(), history

            if history is None:
                history = []
            
            # Convert old tuple format to dictionary format if needed
            if len(history) > 0 and isinstance(history[0], list):
                # Convert old format [[user, bot], ...] to new format
                history = [
                    {"role": "user", "content": msg[0]} if msg[0] else {"role": "assistant", "content": msg[1] or ""}
                    for msg in history if isinstance(msg, list) and len(msg) >= 2
                ]

            # Dictionary format: [{"role": "user", "content": "..."}]
            new_history = history + [{"role": "user", "content": user_message}]

            return "", new_history

        def bot_response(history, temp, max_len, top_k, top_p, rep_pen):
            global _ui_model, _ui_tokenizer, _ui_device, _ui_model_loaded

            if not history: 
                yield history
                return

            # Get last message - handle both formats for compatibility
            last_message = history[-1]
            if isinstance(last_message, dict):
                user_message = last_message.get("content", "")
            elif isinstance(last_message, list) and len(last_message) >= 1:
                # Support old format temporarily
                user_message = last_message[0] if last_message[0] else ""
            else:
                user_message = str(last_message)

            if not _ui_model_loaded or _ui_model is None:
                # Add error message as assistant response
                error_msg = "❌ Model not loaded! Please load a model first."
                history.append({"role": "assistant", "content": error_msg})
                yield history
                return

            try:
                # Check if this is a GGUF model
                try:
                    from llama_cpp import Llama
                    is_gguf = isinstance(_ui_model, Llama)
                except:
                    is_gguf = False
                
                engine = InferenceEngine(_ui_model, _ui_tokenizer, state.db, device=_ui_device)
                
                # Prompt format - InferenceEngine will handle GGUF format conversion
                # For GGUF models, it will convert "User: ..." to Qwen format automatically
                prompt = f"User: {user_message}\nAI:"
                
                gen_kwargs = {
                    "max_length": int(max_len),
                    "temperature": float(temp),
                    "top_k": int(top_k),
                    "top_p": float(top_p),
                    "repetition_penalty": float(rep_pen),
                    "use_kv_cache": True,
                    "method": "sample"
                }

                accumulated_text = ""
                response_received = False
                # Add assistant message placeholder
                history.append({"role": "assistant", "content": ""})

                for text_chunk in engine.generate_stream(prompt, **gen_kwargs):
                    if text_chunk:
                        accumulated_text += text_chunk
                        response_received = True
                        # Update assistant message content
                        history[-1]["content"] = accumulated_text
                        yield history
                
                # If no response was received, show a helpful message
                if not response_received:
                    if is_gguf:
                        error_msg = "⚠️ No response generated. This might be a GGUF stream format issue. Try increasing max_length or check model compatibility."
                    else:
                        error_msg = "⚠️ No response generated. Try increasing max_length or adjusting generation parameters."
                    history[-1]["content"] = error_msg
                    yield history

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"❌ Error: {str(e)}"
                if len(history) > 0 and isinstance(history[-1], dict):
                    history[-1]["content"] = error_msg
                else:
                    history.append({"role": "assistant", "content": error_msg})
                yield history

        load_model_btn.click(
            fn=load_chat_model,
            inputs=[chat_checkpoint, chat_tokenizer],
            outputs=model_status
        )
        
        refresh_checkpoints_btn.click(
            fn=refresh_checkpoints,
            inputs=[],
            outputs=[chat_checkpoint]
        )
        
        refresh_tokenizers_btn.click(
            fn=refresh_tokenizers,
            inputs=[],
            outputs=[chat_tokenizer]
        )

        input_list = [chatbot, temperature, max_length, top_k, top_p, repetition_penalty]

        msg.submit(
            add_user_message, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot], 
            queue=False
        ).then(
            bot_response,
            inputs=input_list,
            outputs=[chatbot]
        )

        send_btn.click(
            add_user_message, 
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot], 
            queue=False
        ).then(
            bot_response,
            inputs=input_list,
            outputs=[chatbot]
        )

        clear_btn.click(lambda: [], None, chatbot, queue=False)