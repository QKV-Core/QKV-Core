"""
Tokenizer tab for the QKV Core Web Interface.
"""
import time
from pathlib import Path
import gradio as gr

from qkv_core.tokenization.bpe import BPETokenizer
from ..state.app_state import state


def create_tokenizer_tab():
    """Create the tokenizer training tab."""
    
    with gr.Tab("🔤 Tokenizer"):
        gr.Markdown("### BPE Tokenizer Training")
        gr.Markdown("Load corpus and train BPE tokenizer")
        
        with gr.Row():
            with gr.Column():
                corpus_file = gr.File(label="📄 Load Corpus File (.txt)", file_types=[".txt"])
                corpus_text = gr.Textbox(
                    label="Or Enter Text Directly",
                    placeholder="Enter your text here (one example per line)...",
                    lines=10
                )
                
                with gr.Row():
                    vocab_size = gr.Slider(
                        minimum=1000,
                        maximum=50000,
                        value=10000,
                        step=1000,
                        label="Vocabulary Size"
                    )
                    min_freq = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Minimum Frequency"
                    )
                
                tokenizer_name = gr.Textbox(
                    label="tokenizer Name",
                    value=f"tokenizer_{int(time.time())}",
                    placeholder="tokenizer_name.pkl"
                )
                
                train_tokenizer_btn = gr.Button("🎓 Train tokenizer", variant="primary", size="lg")
            
            with gr.Column():
                tokenizer_output = gr.Textbox(
                    label="📊 Training Output",
                    lines=15,
                    interactive=False
                )

                gr.Markdown("### Test Tokenizer")
                test_text = gr.Textbox(
                    label="Test Text",
                    placeholder="Enter text to test..."
                )
                test_btn = gr.Button("Test")
                test_output = gr.Textbox(label="Test Result", lines=5)
        
        def train_tokenizer_fn(corpus_file_obj, corpus_text_input, vocab_size_val, min_freq_val, tokenizer_name_val):
            try:
                corpus = []
                
                if corpus_file_obj is not None:
                    with open(corpus_file_obj.name, 'r', encoding='utf-8') as f:
                        corpus = [line.strip() for line in f if line.strip()]
                elif corpus_text_input:
                    corpus = [line.strip() for line in corpus_text_input.split('\n') if line.strip()]
                else:
                    return "❌ Error: Provide corpus file or text!"
                
                if len(corpus) < 10:
                    return "❌ Error: At least 10 lines of corpus required!"
                
                output = f"📊 Corpus loaded: {len(corpus)} lines\n\n"
                
                output += "🔧 Creating tokenizer...\n"
                tokenizer = BPETokenizer(
                    vocab_size=int(vocab_size_val),
                    min_frequency=int(min_freq_val)
                )
                
                output += "🎓 Training started...\n\n"
                tokenizer.train(corpus, verbose=False)
                
                if not tokenizer_name_val.endswith('.pkl'):
                    tokenizer_name_val += '.pkl'
                
                save_path = f"tokenizer/{tokenizer_name_val}"
                Path("tokenizer").mkdir(exist_ok=True)
                tokenizer.save(save_path)
                
                output += f"✅ tokenizer trained!\n"
                output += f"📁 Saved: {save_path}\n"
                output += f"📊 Vocabulary size: {tokenizer.get_vocab_size()}\n"
                output += f"🔢 Number of merges: {len(tokenizer.merges)}\n\n"
                
                state.current_tokenizer = tokenizer
                
                output += "✨ tokenizer is active and ready!"
                
                return output
                
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        def test_tokenizer_fn(test_text_input):
            if state.current_tokenizer is None:
                return "❌ Train or load a tokenizer first!"
            
            try:
                encoded = state.current_tokenizer.encode(test_text_input, add_special_tokens=True)
                decoded = state.current_tokenizer.decode(encoded, skip_special_tokens=False)
                
                output = f"📝 Original: {test_text_input}\n\n"
                output += f"🔢 Encoded ({len(encoded)} tokens):\n{encoded[:50]}{'...' if len(encoded) > 50 else ''}\n\n"
                output += f"📝 Decoded: {decoded}"
                
                return output
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        train_tokenizer_btn.click(
            fn=train_tokenizer_fn,
            inputs=[corpus_file, corpus_text, vocab_size, min_freq, tokenizer_name],
            outputs=tokenizer_output
        )
        
        test_btn.click(
            fn=test_tokenizer_fn,
            inputs=test_text,
            outputs=test_output
        )
