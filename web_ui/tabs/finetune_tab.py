"""
Fine-tuning tab for GPT-2 models.
"""
import gradio as gr

from ..utils.helpers import get_available_checkpoints, get_available_tokenizers


def create_finetune_tab():
    """Create the GPT-2 fine-tuning tab."""
    
    with gr.Tab("🎯 Fine-Tune GPT-2"):
        gr.Markdown()
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Model Selection")

                finetune_checkpoint = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="GPT-2 checkpoint select",
                    value=None
                )
                refresh_finetune_checkpoint = gr.Button("🔄 Refresh", size="sm")
                
                finetune_tokenizer = gr.Dropdown(
                    choices=get_available_tokenizers(),
                    label="tokenizer select",
                    value=None
                )
                refresh_finetune_tokenizer = gr.Button("🔄 Refresh", size="sm")

                gr.Markdown("#### Training Data")

                finetune_corpus_file = gr.File(
                    label="Corpus File (TXT)",
                    file_types=[".txt"],
                    type="filepath"
                )
                
                finetune_corpus_text = gr.Textbox(
                    label="Or Corpus Text",
                    lines=10,
                    placeholder="Each line should be a training example..."
                )

                gr.Markdown("### Training Parameters")

                finetune_epochs = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Epochs",
                    info="For GPT-2, 1-5 epochs are usually sufficient"
                )
                
                finetune_batch_size = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=8,
                    step=1,
                    label="batch Size",
                    info="Recommended 1-8 for GTX 1050"
                )
                
                finetune_learning_rate = gr.Slider(
                    minimum=1e-6,
                    maximum=1e-3,
                    value=5e-5,
                    step=1e-6,
                    label="Learning Rate",
                    info="Use low LR for fine-tuning (5e-5 recommended)"
                )
                
                start_finetune_btn = gr.Button(
                    "🚀 Start Fine-Tuning",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                finetune_status = gr.Textbox(
                    label="📊 Fine-Tuning Status",
                    lines=20,
                    interactive=False,
                    value="💡 Select a GPT-2 checkpoint and tokenizer, then load your training data.\n\n"
                          "**Note:** Fine-tuning is much faster than training from scratch.\n"
                          "Since GPT-2 checkpoint is already pre-trained, only a few epochs are needed."
                )
        
        def refresh_finetune_checkpoints():
            return gr.update(choices=get_available_checkpoints())
        
        def refresh_finetune_tokenizers():
            return gr.update(choices=get_available_tokenizers())
        
        def start_finetune(checkpoint, tokenizer, corpus_file, corpus_text, epochs, batch_size, lr):
            return "⚠️ Fine-tuning functionality is being implemented. Please use the Training tab for now."
        
        refresh_finetune_checkpoint.click(
            fn=refresh_finetune_checkpoints,
            outputs=finetune_checkpoint
        )
        
        refresh_finetune_tokenizer.click(
            fn=refresh_finetune_tokenizers,
            outputs=finetune_tokenizer
        )
        
        start_finetune_btn.click(
            fn=start_finetune,
            inputs=[
                finetune_checkpoint,
                finetune_tokenizer,
                finetune_corpus_file,
                finetune_corpus_text,
                finetune_epochs,
                finetune_batch_size,
                finetune_learning_rate
            ],
            outputs=finetune_status
        )
