"""
Home tab for the QKV Core Web Interface.
"""
import sys
import io
import time
import traceback
from pathlib import Path
import torch
import gradio as gr

from qkv_core.core.transformer import TransformerModel
from qkv_core.tokenization.bpe import BPETokenizer
from qkv_core.training.trainer import Trainer
from qkv_core.training.dataset import TextDataset
from qkv_core.inference.inference import InferenceEngine

# Add project root to path before importing config
# Use absolute path resolution to ensure correct path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Now import config
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
from ..utils.helpers import get_available_tokenizers, get_available_checkpoints
from qkv_core.utils.logger import get_logger

logger = get_logger()


def create_home_tab():
    """Create the home tab with system status and auto training."""
    
    with gr.Tab("🏠 Home"):
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
        else:
            gpu_info = "CPU Mode"

        # Logo integration at the beginning of the title
        from ..utils.logo_helper import get_logo_path
        import base64
        
        logo_path = get_logo_path()
        logo_html = ""
        
        if logo_path and logo_path.exists():
            # Use base64 encoding for reliable logo display
            try:
                with open(logo_path, 'rb') as f:
                    logo_data = base64.b64encode(f.read()).decode('utf-8')
                    logo_html = f"""
                        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                            <img src="data:image/png;base64,{logo_data}" alt="QKV Core Logo" style="max-height: 60px; width: auto; object-fit: contain; background: transparent;">
                            <div>
                                <h1 style="margin: 0; font-size: 2em;">QKV Core - Web Interface</h1>
                                <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9em;"><strong>Query-Key-Value Core - The Core of Transformer Intelligence</strong></p>
                                <p style="margin: 5px 0 0 0; color: #888; font-size: 0.85em;">{gpu_info}</p>
                            </div>
                        </div>
                    """
            except Exception as e:
                logger.warning(f"Could not load logo: {e}")
        
        if logo_html:
            gr.HTML(logo_html)
        else:
            gr.Markdown(f"# QKV Core - Web Interface\n**Query-Key-Value Core - The Core of Transformer Intelligence**\n\n{gpu_info}")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### System Status")
                status_output = gr.Textbox(label="Status", lines=8, interactive=False)
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            with gr.Column():
                gr.Markdown("### Auto Training")

                with gr.Row():
                    with gr.Column():
                        auto_train_btn = gr.Button(
                            "🚀 Start Auto Quick Training",
                            variant="primary",
                            size="lg"
                        )
                        
                        auto_train_status = gr.Textbox(
                            label="📊 Auto Training Status",
                            lines=15,
                            interactive=False,
                            visible=True,
                            value="💡 Click 'Start Auto Quick Training' to begin.\n\nThis process will:\n• Train tokenizer\n• Create balanced model (128d, 3 layer)\n• Train for 5 epochs\n• Prepare model for chat\n\n⏱️ Estimated time: 5-10 minutes (for better quality)"
                        )
                        
                        auto_train_info = gr.Markdown(
                            value="",
                            visible=True
                        )
                
                gr.Markdown()
        
        def get_system_status():
            stats = state.db.get_statistics()
            tokenizers = get_available_tokenizers()
            checkpoints = get_available_checkpoints()
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                device_info = f"{device} ({gpu_name})"
                gpu_memory_info = f"\n• VRAM Total: {gpu_memory:.1f} GB\n• VRAM Used: {allocated:.2f} GB"
            else:
                device_info = device
                gpu_memory_info = ""
            
            status = f"""System Status:
• Device: {device_info}{gpu_memory_info}
• Available Tokenizers: {len(tokenizers)}
• Available Checkpoints: {len(checkpoints)}
• Total Training Sessions: {stats.get('total_sessions', 0)}
• Total Models: {stats.get('total_models', 0)}
• Storage Used: {stats.get('storage_used', 'N/A')}"""
            return status
        
        def auto_train_quick_start(progress=gr.Progress()):
            
            try:
                logger.info("🚀 Starting automatic quick training...")
                
                status = "🚀 Starting Automatic Quick Training...\n\n"
                status += "=" * 60 + "\n"
                info_text = "🔄 Starting process..."
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                try:
                    progress(0.1, desc="📚 Preparing corpus...")
                except:
                    pass
                status += "📚 Step 1: Preparing corpus...\n"
                info_text = "📚 Preparing corpus..."
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                corpus_path = Path("data/sample_corpus.txt")
                test_corpus_path = Path("data/test_corpus.txt")
                
                if corpus_path.exists():
                    status += f"📚 Step 1: Loading corpus from {corpus_path}...\n"
                    info_text = "📝 Loading corpus..."
                    yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                    
                    with open(corpus_path, 'r', encoding='utf-8') as f:
                        all_lines = [line.strip() for line in f if line.strip()]
                    
                    if len(all_lines) > 1000:
                        corpus = all_lines[:1000]
                        status += f"✅ Corpus loaded: {len(corpus)} sentences (first 1000 lines used)\n"
                    else:
                        corpus = all_lines
                        status += f"✅ Corpus loaded: {len(corpus)} sentences\n"
                elif test_corpus_path.exists():
                    status += f"📚 Step 1: Loading corpus from {test_corpus_path}...\n"
                    info_text = "📝 Loading test corpus..."
                    yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                    
                    with open(test_corpus_path, 'r', encoding='utf-8') as f:
                        corpus = [line.strip() for line in f if line.strip()]
                    
                    if len(corpus) > 1000:
                        corpus = corpus[:1000]
                    
                    status += f"✅ Test corpus loaded: {len(corpus)} sentences\n"
                    logger.info(f"Test corpus loaded: {len(corpus)} samples")
                else:
                    status += "📚 Step 1: Creating default corpus (100 sentences)...\n"
                    info_text = "📝 Creating default corpus..."
                    yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                    
                    sample_text = """Hello world! This is a sample sentence for training.
How are you doing today? I hope you're having a great day.
The weather is beautiful outside. Let's go for a walk.
Programming is fun and interesting. I enjoy learning new things.
Machine learning models can be very powerful tools.
Natural language processing helps computers understand text.
Training data quality is important for good results.
Small models can still perform well with good data.
Tokenization breaks text into smaller pieces called tokens.
Transformers are a type of neural network architecture."""

                    corpus = [line.strip() for line in sample_text.strip().split('\n') if line.strip()]
                    status += f"✅ Default corpus created: {len(corpus)} sentences\n"
                    status += "💡 Tip: For better results, add your custom corpus to 'data/sample_corpus.txt'!\n"
                
                status += f"✅ Corpus ready: {len(corpus)} sentences\n\n"
                info_text = f"✅ Corpus ready ({len(corpus)} lines)"
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                try:
                    progress(0.2, desc="🔤 Training tokenizer...")
                except:
                    pass
                status += "🔤 Step 2: Training tokenizer...\n"
                info_text = "🔤 Training tokenizer..."
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                tokenizer_name = f"auto_tokenizer_{int(time.time())}.pkl"
                try:
                    from tokenizer.fast_tokenizer import FastBPETokenizer
                    tokenizer = FastBPETokenizer(vocab_size=5000, min_frequency=1)
                    tokenizer.train(corpus)
                except ImportError:
                    tokenizer = BPETokenizer(vocab_size=5000, min_frequency=1)
                    tokenizer.train(corpus, verbose=False)
                
                Path("tokenizer").mkdir(exist_ok=True)
                tokenizer.save(f"tokenizer/{tokenizer_name}")
                state.current_tokenizer = tokenizer
                state.auto_trained_tokenizer = tokenizer_name
                
                status += f"✅ tokenizer trained: {tokenizer_name}\n"
                status += f"   Vocabulary size: {tokenizer.get_vocab_size()}\n\n"
                info_text = f"✅ tokenizer ready (vocabulary: {tokenizer.get_vocab_size()})"
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                try:
                    progress(0.3, desc="🏗️ Creating model...")
                except:
                    pass
                status += "🏗️ Step 3: Creating optimal model...\n"
                info_text = "🏗️ Creating model..."
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                config = ModelConfig()
                config.vocab_size = tokenizer.get_vocab_size()
                config.d_model = 128
                config.num_layers = 3
                config.num_heads = 4
                config.d_ff = 512
                config.batch_size = 16
                config.max_epochs = 5
                config.learning_rate = 3e-4
                config.max_seq_length = 64
                config.warmup_steps = 100
                config.accumulation_steps = 2
                config.save_every = 999999
                config.log_every = 999999
                config.eval_every = 999999
                config.use_gradient_checkpointing = False
                config.use_mixed_precision = True
                config.skip_database_logging = False
                
                model = TransformerModel(
                    vocab_size=config.vocab_size,
                    d_model=config.d_model,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    max_seq_length=config.max_seq_length,
                    dropout=config.dropout
                )
                
                total_params = model.get_num_parameters()
                status += f"✅ model created: {total_params:,} parameters\n\n"
                info_text = f"✅ model ready ({total_params:,} parameters)"
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                try:
                    progress(0.4, desc="🎓 Training model...")
                except:
                    pass
                status += "🎓 Step 4: Training model...\n"
                status += "─" * 60 + "\n"
                info_text = f"🎓 Starting training ({config.max_epochs} epochs)..."
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                dataset_start = time.time()
                
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                try:
                    dataset = TextDataset(corpus, tokenizer, max_length=config.max_seq_length, lazy_loading=True)
                finally:
                    sys.stdout = old_stdout
                
                dataset_time = time.time() - dataset_start
                status += f"   ✅ Dataset ready ({dataset_time:.2f}s)\n\n"
                
                status += "   🔧 Trainer initialization...\n"
                
                try:
                    trainer = Trainer(
                        model=model,
                        config=config.to_dict(),
                        tokenizer=tokenizer,
                        db_manager=state.db
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    status += f"   ❌ Trainer initialization error: {str(e)}\n"
                    yield status, gr.update(interactive=True, value="❌ Error"), "❌ Trainer initialization error"
                    return
                
                trainer_init_time = time.time() - dataset_start - dataset_time
                status += f"   ✅ Trainer initialized ({trainer_init_time:.2f}s)\n\n"
                
                state.current_trainer = trainer
                state.training_active = True
                
                start_time = time.time()
                
                last_update_time = [0]
                update_interval = 0.5
                
                def update_progress(epoch_num, total_epochs, epoch_progress):
                    
                    try:
                        current_time = time.time()
                        if current_time - last_update_time[0] < update_interval:
                            return
                        last_update_time[0] = current_time
                        
                        base_progress = 0.4
                        training_range = 0.5
                        
                        if total_epochs:
                            epoch_base = base_progress + (epoch_num - 1) * (training_range / total_epochs)
                            current_progress = epoch_base + (epoch_progress * (training_range / total_epochs))
                        else:
                            epoch_base = base_progress + (epoch_num - 1) * (training_range / config.max_epochs)
                            current_progress = epoch_base + (epoch_progress * (training_range / config.max_epochs))
                        
                        progress(min(current_progress, 0.9), desc=f"🎓 Training: epoch {epoch_num}/{config.max_epochs} ({epoch_progress*100:.0f}%)")
                    except:
                        pass
                
                status += f"📊 Starting {config.max_epochs} epoch training...\n"
                info_text = f"🎓 Starting training ({config.max_epochs} epoch)..."
                
                training_success = False
                training_error = None
                
                try:
                    trainer.train(
                        train_dataset=dataset,
                        num_epochs=config.max_epochs,
                        session_name="auto_quick_start",
                        model_version_name="auto_model_v1",
                        progress_callback=update_progress
                    )
                    training_success = True
                except Exception as e:
                    training_error = str(e)
                    logger.error(f"Training error: {training_error}", exc_info=True)
                
                training_time = time.time() - start_time
                state.training_active = False
                
                if training_success:
                    status += f"\n✅ Training completed! ({training_time:.1f} seconds)\n"
                    try:
                        status += f"   Best loss: {trainer.best_loss:.4f}\n\n"
                        logger.log_training_end("auto_quick_start", training_time, trainer.best_loss, trainer.best_loss)
                    except:
                        pass
                    info_text = f"✅ Training completed! ({training_time:.1f}s)"
                else:
                    status += f"\n⚠️  Training interrupted ({training_time:.1f} seconds)\n"
                    if training_error:
                        status += f"   Error: {training_error}\n"
                    status += "   checkpoint kontrol elanguageiyor...\n\n"
                    info_text = f"⚠️ Training interrupted ({training_time:.1f}s)"
                
                yield status, gr.update(interactive=False, value="✅ Training completed!" if training_success else "⚠️ Training interrupted"), info_text
                
                try:
                    progress(0.9, desc="📥 Loading model...")
                except:
                    pass
                status += "📥 Step 5: Preparing model for chat...\n"
                info_text = "📥 Preparing model for chat..."
                yield status, gr.update(interactive=False, value="⏳ Training in progress..."), info_text
                
                checkpoint_path = None
                checkpoint_name = None
                skip_checkpoint_load = False
                
                best_path = Path("model_weights/best_model.pt")
                final_path = Path("model_weights/final_model.pt")
                
                if best_path.exists():
                    checkpoint_path = str(best_path)
                    checkpoint_name = "best_model.pt"
                elif final_path.exists():
                    checkpoint_path = str(final_path)
                    checkpoint_name = "final_model.pt"
                
                if checkpoint_path and Path(checkpoint_path).exists():
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location='CPU')
                        
                        checkpoint_vocab_size = checkpoint.get('vocab_size', None)
                        if checkpoint_vocab_size is None and 'model_state_dict' in checkpoint:
                            if 'embedding.weight' in checkpoint['model_state_dict']:
                                checkpoint_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
                        
                        current_vocab_size = tokenizer.get_vocab_size()
                        
                        if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
                            status += f"⚠️  checkpoint vocabulary size ({checkpoint_vocab_size}) doesn't match current tokenizer ({current_vocab_size}).\n"
                            if training_success:
                                status += "✅ Training successful, using freshly trained model (no checkpoint load needed).\n"
                                logger.warning(f"checkpoint vocabulary size mismatch: {checkpoint_vocab_size} != {current_vocab_size}, using trained model")
                                skip_checkpoint_load = True
                            else:
                                status += "❌ Training failed and checkpoint incompatible. Cannot load model.\n"
                                status += "   Old checkpoint was trained with different tokenizer. New training required.\n"
                                logger.error(f"Training failed and checkpoint incompatible (vocabulary: {checkpoint_vocab_size} != {current_vocab_size})")
                                skip_checkpoint_load = True
                        
                        if not skip_checkpoint_load:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        
                        if training_success or (not skip_checkpoint_load):
                            state.auto_trained_checkpoint = checkpoint_name
                            state.auto_trained_tokenizer = tokenizer_name
                            state.auto_training_complete = True
                            
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            model = model.to(device)
                            state.inference_engine = InferenceEngine(model, tokenizer, state.db, device=device)
                            status += "✅ model automatically loaded for chat!\n"
                            
                            status += "─" * 60 + "\n"
                            status += "🎉 Automatic training completed successfully!\n"
                            status += "💬 Now you can go to the Chat tab to chat!\n"
                            
                            logger.info(f"checkpoint: {checkpoint_path}, tokenizer: {tokenizer_name}")
                            
                            yield status, gr.update(interactive=True, value="✅ Training Completed! You can go to Chat."), "🎉 Success!"
                        else:
                            status += "─" * 60 + "\n"
                            status += "❌ Training failed and no compatible checkpoint found.\n"
                            status += "🔄 Please restart training.\n"
                            yield status, gr.update(interactive=True, value="⚠️ Training Failed"), "❌ Error"
                    except Exception as e:
                        status += f"⚠️  Error loading model: {str(e)}\n"
                        status += "   You can manually load from the Chat tab.\n"
                        logger.error(f"model load failed: {str(e)}", exc_info=True)
                        yield status, gr.update(interactive=True, value="⚠️ model not loaded"), info_text
                else:
                    status += f"⚠️  checkpoint not found!\n"
                    status += f"   Check elanguageen: model_weights/best_model.pt\n"
                    status += f"   Check elanguageen: model_weights/final_model.pt\n"
                    if Path("model_weights").exists():
                        files = list(Path("model_weights").glob("*.pt"))
                        if files:
                            status += f"   Found files: {[f.name for f in files]}\n"
                        else:
                            status += f"   No .pt files in model_weights directory\n"
                    else:
                        status += f"   model_weights directory does not exist\n"
                    status += "   You can manually load model from the Chat tab.\n"
                    logger.warning(f"checkpoint not found. best_model.pt exists: {best_path.exists()}, final_model.pt exists: {final_path.exists()}")
                    yield status, gr.update(interactive=True, value="⚠️ checkpoint not found"), info_text
                    logger.warning("checkpoint not found after training")
                
                try:
                    progress(1.0, desc="✅ Completed!")
                except:
                    pass
                info_text = "✅ Completed! You can go to the Chat tab."
                yield status, gr.update(interactive=True, value="🚀 Start Automatic Quick Training"), info_text
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
                logger.error(f"Auto training failed: {str(e)}", exc_info=True)
                yield error_msg, gr.update(interactive=True, value="🚀 Start Automatic Quick Training"), "❌ An error occurred!"
        
        auto_train_btn.click(
            fn=auto_train_quick_start,
            inputs=[],
            outputs=[auto_train_status, auto_train_btn, auto_train_info]
        )
        
        refresh_btn.click(fn=get_system_status, outputs=status_output)
        
        status_output.value = get_system_status()
