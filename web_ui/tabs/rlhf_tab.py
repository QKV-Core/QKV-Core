"""
RLHF & DPO training tab.
"""
from datetime import datetime
import torch
import gradio as gr

from qkv_core.core.transformer import TransformerModel
from qkv_core.tokenization.bpe import BPETokenizer
from ..utils.model_loader import load_model_from_checkpoint
from ..utils.helpers import get_available_checkpoints, get_available_tokenizers
from ..config.feature_flags import RLHF_AVAILABLE, DPOTrainer, DPOConfig, PreferenceDataset
from ..state.app_state import state
from qkv_core.utils.logger import get_logger

logger = get_logger()


def create_rlhf_tab():
    """Create the RLHF & DPO training tab."""
    
    with gr.Tab("🤖 RLHF & DPO"):
        gr.Markdown("# RLHF & DPO Training")
        gr.Markdown("**ChatGPT-style alignment! Make your model helpful, harmless, and honest.**")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training Data")
                rlhf_base_checkpoint = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Base model (SFT)",
                    value=None
                )
                
                rlhf_tokenizer = gr.Dropdown(
                    choices=get_available_tokenizers(),
                    label="tokenizer",
                    value=None
                )
                
                gr.Markdown("### Alignment Method")

                alignment_method = gr.Radio(
                    choices=["DPO (Recommended - Simpler)", "RLHF (Advanced)"],
                    value="DPO (Recommended - Simpler)",
                    label="Method"
                )
                
                gr.Markdown("### Data Format")

                gr.Markdown("**Format:** Each line: `prompt | chosen_response | rejected_response`")
                
                preference_data = gr.Textbox(
                    label="Preference Data",
                    lines=10,
                    placeholder="Enter preference data here, one example per line..."
                )

                preference_file = gr.File(label="Or Load from File (.txt)", file_types=[".txt"])

                gr.Markdown("### Training Parameters")

                dpo_beta = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Beta (KL penalty)")
                dpo_lr = gr.Number(value=5e-7, label="Learning Rate")
                dpo_batch_size = gr.Slider(1, 8, value=4, step=1, label="batch Size")
                dpo_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                
                dpo_session_name = gr.Textbox(
                    label="Session Name",
                    value=f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                start_dpo_btn = gr.Button("🤖 DPO Training Start", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### Training Progress")

                dpo_output = gr.Textbox(
                    label="Training Output",
                    lines=20,
                    interactive=False
                )
                
                with gr.Row():
                    dpo_accuracy = gr.Number(label="Preference Accuracy", value=0.0, interactive=False)
                    dpo_loss = gr.Number(label="DPO Loss", value=0.0, interactive=False)
                    dpo_reward_margin = gr.Number(label="Reward Margin", value=0.0, interactive=False)
        
        def start_dpo_training(
            checkpoint, tokenizer_name, method,
            preference_text, preference_file_obj,
            beta, lr, batch_size, epochs, session_name
        ):
            try:
                output = "🤖 DPO Training starting...\n\n"
                
                if not checkpoint or not tokenizer_name:
                    return "❌ Select checkpoint and tokenizer!", 0.0, 0.0, 0.0
                
                preferences = []
                if preference_file_obj:
                    with open(preference_file_obj.name, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '|' in line:
                                parts = line.strip().split('|')
                                if len(parts) == 3:
                                    preferences.append([p.strip() for p in parts])
                elif preference_text:
                    for line in preference_text.split('\n'):
                        if '|' in line:
                            parts = line.strip().split('|')
                            if len(parts) == 3:
                                preferences.append([p.strip() for p in parts])
                
                if len(preferences) < 5:
                    return "❌ En az 5 preference pair required!", 0.0, 0.0, 0.0
                
                output += f"📊 Preference pairs: {len(preferences)}\n"
                
                output += "📥 Models loading...\n"
                tokenizer = BPETokenizer.load(f"tokenizer/{tokenizer_name}")
                
                checkpoint_path = f"model_weights/{checkpoint}"
                
                policy_model, config, checkpoint_data = load_model_from_checkpoint(checkpoint_path, 'CPU', logger)
                
                if policy_model is None:
                    return "❌ model could not be loaded!", 0.0, 0.0, 0.0
                
                ref_model = TransformerModel(config)
                ref_model.load_state_dict(checkpoint_data['model_state_dict'])
                for param in ref_model.parameters():
                    param.requires_grad = False
                
                output += "✅ Models loaded\n\n"
                
                if not RLHF_AVAILABLE:
                    return "❌ RLHF/DPO module not available! Please install the required dependencies.", 0.0, 0.0, 0.0
                
                output += "🔧 DPO trainer creating...\n"
                dpo_config = DPOConfig(
                    beta=float(beta),
                    learning_rate=float(lr),
                    batch_size=int(batch_size),
                    max_length=512
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                policy_model = policy_model.to(device)
                ref_model = ref_model.to(device)
                
                trainer = DPOTrainer(policy_model, ref_model, dpo_config, device=str(device))
                
                output += "🎓 DPO training starting...\n"
                output += "─" * 50 + "\n"
                
                prompts = [p[0] for p in preferences]
                chosen = [p[1] for p in preferences]
                rejected = [p[2] for p in preferences]
                
                dataset = PreferenceDataset(prompts, chosen, rejected, tokenizer)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=int(batch_size),
                    shuffle=True
                )
                
                for epoch in range(int(epochs)):
                    epoch_metrics = []
                    for batch in dataloader:
                        metrics = trainer.train_step(batch)
                        epoch_metrics.append(metrics)
                        
                        output += f"epoch {epoch+1}/{epochs}, Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2%}\n"
                    
                    avg_metrics = {
                        k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                        for k in epoch_metrics[0].keys()
                    }
                    
                    output += f"\nEpoch {epoch+1} Summary:\n"
                    output += f"  Loss: {avg_metrics['loss']:.4f}\n"
                    output += f"  Accuracy: {avg_metrics['accuracy']:.2%}\n"
                    output += f"  Reward Margin: {avg_metrics['reward_margin']:.4f}\n\n"
                
                save_path = f"model_weights/aligned_{session_name}.pt"
                torch.save({
                    'model_state_dict': policy_model.state_dict(),
                    'config': config,
                    'dpo_config': dpo_config.__dict__
                }, save_path)
                
                output += f"✅ DPO training completed!\n"
                output += f"📁 Aligned model saved: {save_path}\n"
                
                final_metrics = avg_metrics
                return output, final_metrics['accuracy'], final_metrics['loss'], final_metrics['reward_margin']
                
            except Exception as e:
                import traceback
                return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", 0.0, 0.0, 0.0
        
        start_dpo_btn.click(
            fn=start_dpo_training,
            inputs=[
                rlhf_base_checkpoint, rlhf_tokenizer, alignment_method,
                preference_data, preference_file,
                dpo_beta, dpo_lr, dpo_batch_size, dpo_epochs, dpo_session_name
            ],
            outputs=[dpo_output, dpo_accuracy, dpo_loss, dpo_reward_margin]
        )
