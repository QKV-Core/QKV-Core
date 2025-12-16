"""
Training tab for the QKV Core Web Interface.
"""
import sys
from pathlib import Path
from datetime import datetime
import gradio as gr

# Add project root to path before importing config
# Use absolute path resolution to ensure correct path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

from qkv_core.core.transformer import TransformerModel
from qkv_core.tokenization.bpe import BPETokenizer
from qkv_core.training.trainer import Trainer
from qkv_core.training.dataset import TextDataset

# Now import config with fallback
try:
    from config.model_config import ModelConfig
except ImportError as e:
    # Fallback: try importing with explicit path
    import importlib.util
    config_path = _project_root / "config" / "model_config.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("model_config", config_path)
        model_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_config_module)
        ModelConfig = model_config_module.ModelConfig
    else:
        raise ImportError(f"Cannot import config.model_config: {e}")

from ..state.app_state import state
from ..utils.helpers import get_available_tokenizers
from ..config.feature_flags import (
    MAMBA_AVAILABLE, MambaModel,
    FLASH_ATTN_AVAILABLE, FlashMultiHeadAttention,
    QUANTIZATION_AVAILABLE, QuantizationConfig, quantize_model, measure_model_size,
    SCALING_LAWS_AVAILABLE, ScalingLawsConfig
)


def create_training_tab():
    """Create the training tab."""
    
    with gr.Tab("🎓 Training"):
        gr.Markdown("### Model Training")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Data Input")
                training_corpus = gr.File(label="Training Corpus (.txt)", file_types=[".txt"])
                training_text = gr.Textbox(
                    label="Or Enter Text Directly",
                    lines=5,
                    placeholder="Training text..."
                )

                gr.Markdown("#### Tokenizer Configuration")
                tokenizer_choice = gr.Radio(
                    choices=["Use Existing tokenizer", "Train New tokenizer"],
                    value="Use Existing tokenizer",
                    label="tokenizer Selection"
                )
                
                existing_tokenizer = gr.Dropdown(
                    choices=get_available_tokenizers(),
                    label="Select tokenizer",
                    visible=True
                )

                gr.Markdown("#### Model Configuration")

                model_architecture = gr.Radio(
                    choices=["Transformer (Classic)", "Mamba (O(N) - New!)" + (" ⚠️ einops required" if not MAMBA_AVAILABLE else "")],
                    value="Transformer (Classic)",
                    label="model Architecture"
                )
                
                model_preset = gr.Radio(
                    choices=["Tiny (Test)", "Small (Fast)", "Medium (Balanced)", "Large (Powerful)", "Custom"],
                    value="Tiny (Test)",
                    label="model Size (Tiny recommended for CPU!)"
                )
                
                with gr.Accordion("model Parameters", open=False):
                    d_model = gr.Slider(64, 512, value=128, step=64, label="d_model")
                    num_layers = gr.Slider(2, 6, value=2, step=1, label="Layers")
                    num_heads = gr.Slider(4, 8, value=4, step=2, label="Attention Heads")
                    d_ff = gr.Slider(256, 2048, value=512, step=256, label="FFN Dimension")
                
                with gr.Accordion("🚀 Advanced Features (Research)", open=False):
                    use_flash_attention = gr.Checkbox(
                        label="⚡ FlashAttention (2-4x faster, 10-20x less memory)",
                        value=False
                    )
                    use_scaling_laws = gr.Checkbox(
                        label="📈 Scaling Laws Optimizer (Auto hyperparameters)",
                        value=False
                    )
                    enable_quantization = gr.Checkbox(
                        label="📦 Post-Training Quantization",
                        value=False
                    )
                    quantization_bits = gr.Radio(
                        choices=["8-bit (4x smaller)", "4-bit (8x smaller)"],
                        value="8-bit (4x smaller)",
                        label="Quantization Level",
                        visible=False
                    )
                    enable_quantization.change(
                        lambda x: gr.update(visible=x),
                        inputs=enable_quantization,
                        outputs=quantization_bits
                    )
                
                with gr.Accordion("Training Parameters", open=True):
                    batch_size = gr.Slider(2, 32, value=4, step=2, label="batch Size (4-8 recommended for CPU)")
                    epochs = gr.Slider(1, 20, value=3, step=1, label="Epochs (3 is enough for initial testing)")
                    learning_rate = gr.Number(value=0.0003, label="Learning Rate")
                
                session_name = gr.Textbox(
                    label="Session Name",
                    value=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="stop")
            
            with gr.Column(scale=2):
                gr.Markdown("#### Training Progress")
                training_progress = gr.Textbox(
                    label="Console Output",
                    lines=20,
                    interactive=False,
                    max_lines=30
                )
                
                with gr.Row():
                    current_epoch = gr.Number(label="Current epoch", value=0, interactive=False)
                    current_loss = gr.Number(label="Current Loss", value=0.0, interactive=False)
                    best_loss = gr.Number(label="Best Loss", value=0.0, interactive=False)
        
        def update_model_params(preset):
            
            presets = {
                "Tiny (Test)": (64, 2, 4, 256),
                "Small (Fast)": (128, 2, 4, 512),
                "Medium (Balanced)": (256, 4, 8, 1024),
                "Large (Powerful)": (512, 6, 8, 2048),
                "Custom": (128, 2, 4, 512)
            }
            
            if preset in presets:
                d_m, layers, heads, ff = presets[preset]
                return d_m, layers, heads, ff
            return 128, 2, 4, 512
        
        model_preset.change(
            fn=update_model_params,
            inputs=model_preset,
            outputs=[d_model, num_layers, num_heads, d_ff]
        )
        
        def start_training(
            corpus_file_obj, corpus_text_input,
            tokenizer_choice_val, existing_tokenizer_val,
            model_arch_val, d_model_val, num_layers_val, num_heads_val, d_ff_val,
            batch_size_val, epochs_val, lr_val,
            session_name_val,
            use_flash_attn, use_scaling_laws, enable_quant, quant_bits
        ):
            try:
                output = "🚀 Starting training...\n\n"
                
                corpus = []
                if corpus_file_obj is not None:
                    with open(corpus_file_obj.name, 'r', encoding='utf-8') as f:
                        corpus = [line.strip() for line in f if line.strip()]
                elif corpus_text_input:
                    corpus = [line.strip() for line in corpus_text_input.split('\n') if line.strip()]
                else:
                    return "❌ Corpus required!", 0, 0.0, 0.0
                
                output += f"📊 Corpus: {len(corpus)} samples\n"
                
                if tokenizer_choice_val == "Use Existing tokenizer":
                    if not existing_tokenizer_val:
                        return "❌ Select a tokenizer!", 0, 0.0, 0.0
                    
                    tokenizer = BPETokenizer.load(f"tokenizer/{existing_tokenizer_val}")
                    output += f"✅ tokenizer loaded: {existing_tokenizer_val}\n"
                else:
                    output += "🔤 Training tokenizer...\n"
                    tokenizer = BPETokenizer(vocab_size=10000)
                    tokenizer.train(corpus, verbose=False)
                    output += "✅ tokenizer trained\n"
                
                output += "📦 Creating dataset...\n"
                dataset = TextDataset(corpus, tokenizer, max_length=512)
                
                output += f"🏗️ Creating model...\n"
                output += f"   • Architecture: {model_arch_val}\n"
                output += f"   • d_model: {d_model_val}\n"
                output += f"   • layers: {num_layers_val}\n"
                output += f"   • heads: {num_heads_val}\n"
                
                config = ModelConfig()
                config.vocab_size = tokenizer.get_vocab_size()
                config.d_model = int(d_model_val)
                config.num_layers = int(num_layers_val)
                config.num_heads = int(num_heads_val)
                config.d_ff = int(d_ff_val)
                config.batch_size = int(batch_size_val)
                config.max_epochs = int(epochs_val)
                
                if use_scaling_laws:
                    if not SCALING_LAWS_AVAILABLE:
                        output += "⚠️  Scaling Laws not available. Using standard LR...\n"
                        config.learning_rate = float(lr_val)
                    else:
                        output += "\n📈 Scaling Laws optimizer aktif...\n"
                        scaling_config = ScalingLawsConfig(
                            model_params=config.d_model * config.num_layers * 1000,
                            compute_budget_flops=1e18
                        )
                        config.learning_rate = scaling_config.learning_rate
                        config.batch_size = scaling_config.batch_size
                        output += f"   • Optimal LR: {scaling_config.learning_rate:.2e}\n"
                        output += f"   • Optimal batch: {scaling_config.batch_size}\n"
                else:
                    config.learning_rate = float(lr_val)
                
                if "Mamba" in model_arch_val:
                    if not MAMBA_AVAILABLE:
                        output += "⚠️  Mamba not available (einops required). Using Transformer...\n"
                        output += "💡 To install: pip install einops\n"
                        model_arch_val = "Transformer (Classic)"
                    
                    if MAMBA_AVAILABLE:
                        output += "🐍 Creating Mamba (SSM) model creating...\n"
                        model = MambaModel(
                            vocab_size=config.vocab_size,
                            d_model=config.d_model,
                            num_layers=config.num_layers,
                            d_state=16,
                            d_conv=4,
                            expand_factor=2,
                            dropout=config.dropout,
                            max_seq_length=config.max_seq_length
                        )
                    else:
                        model = TransformerModel(
                            vocab_size=config.vocab_size,
                            d_model=config.d_model,
                            num_layers=config.num_layers,
                            num_heads=config.num_heads,
                            d_ff=config.d_ff,
                            max_seq_length=config.max_seq_length,
                            dropout=config.dropout
                        )
                else:
                    model = TransformerModel(
                        vocab_size=config.vocab_size,
                        d_model=config.d_model,
                        num_layers=config.num_layers,
                        num_heads=config.num_heads,
                        d_ff=config.d_ff,
                        max_seq_length=config.max_seq_length,
                        dropout=config.dropout
                    )
                    
                    if use_flash_attn:
                        if not FLASH_ATTN_AVAILABLE:
                            output += "⚠️  FlashAttention not available. Using standard attention...\n"
                        else:
                            output += "⚡ FlashAttention aktif elanguageiyor...\n"
                            try:
                                for layer in model.encoder_layers:
                                    layer.self_attention = FlashMultiHeadAttention(
                                        d_model=config.d_model,
                                        num_heads=config.num_heads,
                                        dropout=config.dropout
                                    )
                                output += "✅ FlashAttention successfully applied!\n"
                            except Exception as e:
                                output += f"⚠️  FlashAttention error: {str(e)}\n"
                                output += "   Using standard attention...\n"
                
                total_params = model.get_num_parameters()
                output += f"✅ model created: {total_params:,} parameters\n\n"
                
                trainer = Trainer(
                    model=model,
                    config=config.to_dict(),
                    tokenizer=tokenizer,
                    db_manager=state.db
                )
                
                state.current_trainer = trainer
                state.training_active = True
                
                output += "🎓 Starting training...\n"
                output += "─" * 50 + "\n"
                
                trainer.train(
                    train_dataset=dataset,
                    num_epochs=config.max_epochs,
                    session_name=session_name_val,
                    model_version_name=f"model_{session_name_val}"
                )
                
                output += "\n✅ Training completed!"
                
                if enable_quant:
                    if not QUANTIZATION_AVAILABLE:
                        output += "⚠️  Quantization not available...\n"
                    else:
                        output += "\n📦 model quantize elanguageiyor...\n"
                        try:
                            bits = 8 if "8-bit" in quant_bits else 4
                            quant_config = QuantizationConfig(bits=bits, group_size=128)
                            model = quantize_model(model, quant_config)
                            
                            original_size = measure_model_size(model)
                            output += f"✅ Quantization completed ({bits}-bit)!\n"
                            output += f"   • model size: {original_size['total_size_mb']:.2f} MB\n"
                        except Exception as e:
                            output += f"⚠️  Quantization error: {str(e)}\n"
                
                return output, config.max_epochs, trainer.best_loss, trainer.best_loss
                
            except Exception as e:
                state.training_active = False
                return f"❌ Error: {str(e)}", 0, 0.0, 0.0
        
        train_btn.click(
            fn=start_training,
            inputs=[
                training_corpus, training_text,
                tokenizer_choice, existing_tokenizer,
                model_architecture, d_model, num_layers, num_heads, d_ff,
                batch_size, epochs, learning_rate,
                session_name,
                use_flash_attention, use_scaling_laws, enable_quantization, quantization_bits
            ],
            outputs=[training_progress, current_epoch, current_loss, best_loss]
        )
