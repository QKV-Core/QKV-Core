"""
Download model tab for HuggingFace model conversion.
"""
from pathlib import Path
import gradio as gr

from ..utils.helpers import get_available_checkpoints
from ..config.feature_flags import HF_CONVERTER_AVAILABLE, HuggingFaceConverter
from qkv_core.utils.logger import get_logger

# Try to import GGUF loader
try:
    from qkv_core.formats.gguf_loader import GGUFModelLoader
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    GGUFModelLoader = None

logger = get_logger()


def create_download_model_tab():
    """Create the download model tab."""
    
    with gr.Tab("📥 Download Model"):
        if not HF_CONVERTER_AVAILABLE:
            gr.Markdown("## ⚠️ Transformers Library Not Available\n\nInstall with: `pip install transformers`")
            return
        
        gr.Markdown("# Download Pre-trained Models from Hugging Face\n\n**Download and convert GPT-2 compatible models**")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Selection")

                model_selection = gr.Dropdown(
                    choices=list(HuggingFaceConverter.RECOMMENDED_MODELS.keys()),
                    value="Qwen/Qwen2.5-1.5B-Instruct-GGUF" if "Qwen/Qwen2.5-1.5B-Instruct-GGUF" in HuggingFaceConverter.RECOMMENDED_MODELS else list(HuggingFaceConverter.RECOMMENDED_MODELS.keys())[0] if HuggingFaceConverter.RECOMMENDED_MODELS else None,
                    label="Recommended Models",
                    info="🏆 Champion: Qwen2.5-Coder-3B-Instruct-GGUF (Q5_K_M, ~2.4GB VRAM) - Best for GTX 1050 4GB! 💻 For Coding: Qwen2.5-Coder-7B-Instruct-GGUF (~3.2GB VRAM). 🤖 General Purpose: Phi-3-mini-4k-instruct-GGUF (~2.4GB VRAM) or Qwen2.5-1.5B-Instruct-GGUF (~0.8GB VRAM). For GTX 1050: DistilGPT-2 (very fast) or GPT-2 Small (recommended)."
                )
                
                custom_model = gr.Textbox(
                    label="Or Custom Model Name",
                    placeholder="e.g.: gpt2, distilgpt2, EleutherAI/gpt-neo-125M, cenkersisman/gpt2-turkish-50m",
                    info="Any GPT-2 compatible model name from Hugging Face model hub"
                )
                
                output_name = gr.Textbox(
                    label="Output Name (Optional)",
                    placeholder="gpt2_small",
                    info="Name for checkpoint and tokenizer (uses model name if left empty)"
                )
                
                download_btn = gr.Button(
                    "🚀 Download & Convert Model",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                gr.Markdown("### Download Progress")

                download_status = gr.Textbox(
                    label="Status",
                    lines=20,
                    interactive=False,
                    value="💡 Select a model and click the 'Download and Convert Model' button."
                )
        
        def download_hf_model(model_choice, custom_model_name, output_name_input):
            
            try:
                if custom_model_name and custom_model_name.strip():
                    model_name = custom_model_name.strip()
                    status_messages = [f"📥 Custom model selectildi: {model_name}\n"]
                    status_messages.append("⚠️ Note: Custom model GPT-2 compatible mimariye sahip should be\n\n")
                    is_gguf = False
                else:
                    model_name = model_choice
                    model_info = HuggingFaceConverter.RECOMMENDED_MODELS.get(model_name, {})
                    model_desc = model_info.get('description', model_name)
                    model_size = model_info.get('size', 'Unknown')
                    download_size = model_info.get('download_size_mb', 0)
                    is_gguf = model_info.get('model_type') == 'gguf'
                    
                    status_messages = [
                        f"📥 Model Selected: {model_desc}\n",
                        f"📊 Parameters: {model_size}\n",
                        f"💾 Download Size: ~{download_size} MB\n",
                        f"📜 License: {model_info.get('license', 'Unknown')}\n"
                    ]
                    
                    if is_gguf:
                        status_messages.append("🚀 GGUF Format Model - Optimized for low VRAM!\n")
                    
                    # Add note if available
                    if model_info.get('note'):
                        status_messages.append(f"💡 {model_info.get('note')}\n")
                    
                    params = model_info.get('params', 0)
                    if params > 1_000_000_000:
                        if params <= 1_500_000_000:
                            # 1B-1.5B models: May work but need optimization
                            if is_gguf:
                                status_messages.append(f"✅ GGUF Model ({model_size}) - Perfect for GTX 1050 with 4GB VRAM (~0.9-1GB usage)!\n")
                            else:
                                status_messages.append(f"⚠️ Medium-Large Model ({model_size}) - Should work on GTX 1050 with 4GB VRAM\n")
                                status_messages.append(f"💡 These models (Qwen2.5-1.5B) are optimized for small GPUs!\n")
                        elif params <= 7_000_000_000:
                            # 2B-7B models: Large but may work with optimization (especially GGUF)
                            if is_gguf:
                                status_messages.append(f"✅ GGUF Model ({model_size}) - Optimized format! Requires ~4GB VRAM (much less than PyTorch version)\n")
                                status_messages.append(f"💡 Qwen2.5-Coder-7B-Instruct-GGUF is specialized for code generation. GGUF format makes it efficient for lower VRAM GPUs.\n")
                            else:
                                status_messages.append(f"⚠️ Large Model ({model_size}) - Requires ~6-8GB VRAM for optimal performance\n")
                                status_messages.append(f"💡 Consider using GGUF format for better VRAM efficiency.\n")
                        else:
                            status_messages.append(f"⚠️ ⚠️ ⚠️ VERY LARGE MODEL ({model_size}) - May not work on GTX 1050 or be extremely slow!\n")
                            status_messages.append(f"💡 Recommended: Models under 500M (DistilGPT-2, GPT-2 Small, Pythia 70M)\n")
                    elif params > 500_000_000:
                        status_messages.append(f"⚠️ Large model - May be slow on GTX 1050!\n")
                    elif params < 200_000_000:
                        status_messages.append(f"✅ Small model - Ideal for GTX 1050!\n")
                    
                    status_messages.append("\n")
                
                # Handle GGUF model download
                if is_gguf:
                    if not GGUF_AVAILABLE or not GGUFModelLoader.is_available():
                        status_messages.append("❌ ERROR: llama-cpp-python is not installed!\n")
                        status_messages.append("💡 Install with: pip install llama-cpp-python\n")
                        yield "\n".join(status_messages)
                        return
                    
                    model_info = HuggingFaceConverter.RECOMMENDED_MODELS.get(model_name, {})
                    file_name = model_info.get('file_name', 'qwen2.5-1.5b-instruct-q3_k_m.gguf')
                    
                    def progress_callback(msg):
                        status_messages.append(msg + "\n")
                        return "\n".join(status_messages)
                    
                    status_messages.append("🚀 Starting GGUF model download...\n")
                    yield "\n".join(status_messages)
                    
                    import time as time_module
                    download_start = time_module.time()
                    
                    downloaded_path = GGUFModelLoader.download_gguf_model(
                        model_name=model_name,
                        file_name=file_name,
                        output_dir="model_weights",
                        progress_callback=progress_callback
                    )
                    
                    download_time = time_module.time() - download_start
                    
                    if downloaded_path:
                        status_messages.append("\n" + "=" * 60 + "\n")
                        status_messages.append("✅ GGUF MODEL SUCCESSFULLY DOWNLOADED!\n")
                        status_messages.append("=" * 60 + "\n\n")
                        status_messages.append(f"📁 Model File: {Path(downloaded_path).name}\n")
                        status_messages.append(f"⏱️ Download Time: {download_time:.1f} seconds\n\n")
                        status_messages.append("💡 You can now load and use this model from the Chat tab!\n")
                        status_messages.append(f"   Model: {Path(downloaded_path).name}\n")
                        status_messages.append("   Note: GGUF models use their built-in tokenizer, no separate tokenizer file needed.\n\n")
                        status_messages.append("💡 Note: If you don't see the dropdowns in the Chat tab, click the 🔄 Refresh button!\n")
                    else:
                        status_messages.append("\n❌ Failed to download GGUF model. Check the error messages above.\n")
                    
                    yield "\n".join(status_messages)
                    return
                
                # Handle regular PyTorch model download
                if not output_name_input or not output_name_input.strip():
                    output_name_final = model_name.replace('/', '_').replace('-', '_')
                else:
                    output_name_final = output_name_input.strip()
                
                status_messages.append(f"🔧 Output Name: {output_name_final}\n")
                status_messages.append("─" * 60 + "\n")
                
                def progress_callback(msg):
                    status_messages.append(msg + "\n")
                    return "\n".join(status_messages)
                
                converter = HuggingFaceConverter()
                status_messages.append("🚀 Starting download...\n")
                yield "\n".join(status_messages)
                
                import time as time_module
                download_start = time_module.time()
                
                result = converter.download_and_convert(
                    model_name,
                    output_name_final,
                    progress_callback=lambda m: status_messages.append(m + "\n")
                )
                
                download_time = time_module.time() - download_start
                
                status_messages.append("\n" + "=" * 60 + "\n")
                status_messages.append("✅ MODEL SUCCESSFULLY DOWNLOADED AND CONVERTED!\n")
                status_messages.append("=" * 60 + "\n\n")
                status_messages.append(f"📁 Checkpoint: {result['checkpoint_path']}\n")
                status_messages.append(f"🔤 Tokenizer: {result['tokenizer_path']}\n\n")
                status_messages.append("📊 Model Configuration:\n")
                config = result['config']
                status_messages.append(f"   • Vocabulary Size: {config['vocab_size']:,}\n")
                status_messages.append(f"   • d_model: {config['d_model']}\n")
                status_messages.append(f"   • Layers: {config['num_layers']}\n")
                status_messages.append(f"   • Heads: {config['num_heads']}\n")
                status_messages.append(f"   • d_ff: {config['d_ff']}\n\n")
                status_messages.append("💡 You can now load and use this model from the Chat tab!\n")
                status_messages.append(f"   Checkpoint: {Path(result['checkpoint_path']).name}\n")
                status_messages.append(f"   Tokenizer: {Path(result['tokenizer_path']).name}\n\n")
                status_messages.append("💡 Note: If you don't see the dropdowns in the Chat tab, click the 🔄 Refresh button!\n")
                
                yield "\n".join(status_messages)
                
            except Exception as e:
                import traceback
                error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
                logger.error(f"[DOWNLOAD] ❌ Error during download/convert: {str(e)}", exc_info=True)
                yield error_msg
        
        download_btn.click(
            fn=download_hf_model,
            inputs=[model_selection, custom_model, output_name],
            outputs=[download_status]
        )
