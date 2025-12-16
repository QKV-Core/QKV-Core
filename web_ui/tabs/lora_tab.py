"""
LoRA fine-tuning tab for the QKV Core Web Interface.
"""
from datetime import datetime
import torch
import gradio as gr

from qkv_core.tokenization.bpe import BPETokenizer
from qkv_core.training.dataset import TextDataset
from ..utils.model_loader import load_model_from_checkpoint
from ..utils.helpers import get_available_checkpoints, get_available_tokenizers
from ..config.feature_flags import LORA_AVAILABLE, add_lora_to_model, freeze_base_model, get_lora_state_dict
from ..state.app_state import state
from qkv_core.utils.logger import get_logger

logger = get_logger()


def create_lora_tab():
    """Create the LoRA fine-tuning tab."""
    
    with gr.Tab("🎯 LoRA Fine-Tuning"):
        gr.Markdown("### LoRA Fine-Tuning")
        gr.Markdown("**100x fewer trainable parameters! Fine-tune large models efficiently.**")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Configuration")

                lora_base_checkpoint = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Base model checkpoint",
                    value=None
                )
                
                lora_base_tokenizer = gr.Dropdown(
                    choices=get_available_tokenizers(),
                    label="tokenizer",
                    value=None
                )

                gr.Markdown("#### LoRA Parameters")

                lora_target_modules = gr.CheckboxGroup(
                    choices=["q_proj", "k_proj", "v_proj", "o_proj"],
                    value=["q_proj", "v_proj"],
                    label="Target Modules"
                )
                
                lora_rank = gr.Slider(
                    minimum=4,
                    maximum=64,
                    value=8,
                    step=4,
                    label="LoRA Rank (r)"
                )
                
                lora_alpha = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=16,
                    step=8,
                    label="LoRA Alpha"
                )
                
                lora_dropout = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.05,
                    step=0.05,
                    label="LoRA Dropout"
                )

                gr.Markdown("#### Training Data")

                lora_corpus = gr.File(label="Fine-tuning Corpus (.txt)", file_types=[".txt"])
                lora_text = gr.Textbox(
                    label="Or Direct Text Input",
                    lines=5,
                    placeholder="Fine-tuning metni..."
                )
                
                with gr.Accordion("Training Parameters", open=True):
                    lora_batch_size = gr.Slider(2, 16, value=4, step=2, label="batch Size")
                    lora_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                    lora_lr = gr.Number(value=2e-5, label="Learning Rate")
                
                lora_session_name = gr.Textbox(
                    label="Session Name",
                    value=f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                apply_lora_btn = gr.Button("🎯 LoRA Trainingi Start", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("#### Training Progress")

                lora_output = gr.Textbox(
                    label="Training Output",
                    lines=20,
                    interactive=False
                )
                
                with gr.Row():
                    lora_trainable_params = gr.Number(label="Trainable Parameters", value=0, interactive=False)
                    lora_total_params = gr.Number(label="Total Parameters", value=0, interactive=False)
                    lora_reduction = gr.Number(label="Reduction Factor", value=0, interactive=False)
        
        def start_lora_training(
            checkpoint, tokenizer_name,
            target_modules, rank, alpha, dropout,
            corpus_file, corpus_text,
            batch_size, epochs, lr, session_name
        ):
            try:
                output = "🎯 LoRA Fine-tuning starting...\n\n"
                
                if not checkpoint or not tokenizer_name:
                    return "❌ Select checkpoint and tokenizer!", 0, 0, 0
                
                output += "📥 Base model loading...\n"
                tokenizer = BPETokenizer.load(f"tokenizer/{tokenizer_name}")
                
                checkpoint_path = f"model_weights/{checkpoint}"
                
                model, config, checkpoint_data = load_model_from_checkpoint(checkpoint_path, 'CPU', logger)
                
                if model is None:
                    return "❌ model could not be loaded!", 0, 0, 0
                
                total_params = model.get_num_parameters()
                output += f"✅ Base model loaded: {total_params:,} parameters\n\n"
                
                if not LORA_AVAILABLE:
                    return "❌ LoRA module not available! Please install the required dependencies.", 0, 0, 0

                output += "🔧 Adding LoRA adapters...\n"
                model_lora = add_lora_to_model(
                    model,
                    target_modules=target_modules,
                    r=int(rank),
                    lora_alpha=int(alpha),
                    lora_dropout=float(dropout)
                )
                
                freeze_base_model(model_lora)
                
                trainable_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
                reduction = total_params / trainable_params if trainable_params > 0 else 0
                
                output += f"✅ LoRA added!\n"
                output += f"   • Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)\n"
                output += f"   • Reduction: {reduction:.1f}x\n\n"
                
                corpus = []
                if corpus_file:
                    with open(corpus_file.name, 'r', encoding='utf-8') as f:
                        corpus = [line.strip() for line in f if line.strip()]
                elif corpus_text:
                    corpus = [line.strip() for line in corpus_text.split('\n') if line.strip()]
                
                if len(corpus) < 10:
                    return "❌ En az 10 example required!", 0, 0, 0
                
                output += f"📚 Training data: {len(corpus)} samples\n"
                output += "🎓 Training starting...\n"
                output += "─" * 50 + "\n"
                
                dataset = TextDataset(corpus, tokenizer, max_length=512)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
                
                optimizer = torch.optim.AdamW(
                    [p for p in model_lora.parameters() if p.requires_grad],
                    lr=float(lr)
                )
                
                model_lora.train()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_lora = model_lora.to(device)
                
                for epoch in range(int(epochs)):
                    epoch_loss = 0
                    for batch_idx, batch in enumerate(dataloader):
                        src, tgt_input, tgt_output = [b.to(device) for b in batch]
                        
                        optimizer.zero_grad()
                        logits = model_lora(src, tgt_input)
                        loss = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            tgt_output.reshape(-1),
                            ignore_index=0
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_lora.parameters(), 1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                        if batch_idx % 10 == 0:
                            output += f"epoch {epoch+1}/{epochs}, batch {batch_idx}, Loss: {loss.item():.4f}\n"
                    
                    avg_loss = epoch_loss / len(dataloader)
                    output += f"epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}\n"
                
                lora_state = get_lora_state_dict(model_lora)
                save_path = f"model_weights/lora_{session_name}.pt"
                torch.save(lora_state, save_path)
                
                output += f"\n✅ LoRA training completed!\n"
                output += f"📁 Saved: {save_path}\n"
                
                return output, trainable_params, total_params, reduction
                
            except Exception as e:
                import traceback
                return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", 0, 0, 0
        
        apply_lora_btn.click(
            fn=start_lora_training,
            inputs=[
                lora_base_checkpoint, lora_base_tokenizer,
                lora_target_modules, lora_rank, lora_alpha, lora_dropout,
                lora_corpus, lora_text,
                lora_batch_size, lora_epochs, lora_lr, lora_session_name
            ],
            outputs=[lora_output, lora_trainable_params, lora_total_params, lora_reduction]
        )
