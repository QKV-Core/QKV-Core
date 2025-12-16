import time
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from qkv_core.tokenization.bpe import BPETokenizer
from qkv_core.core.transformer import TransformerModel
from qkv_core.inference.inference import InferenceEngine

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, torch {torch.__version__}")

    corpus_path = Path('data/sample_corpus.txt')
    corpus = []
    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [l.strip() for l in f if l.strip()]

    tokenizer = BPETokenizer(vocab_size=5000, min_frequency=1)
    t0 = time.time()
    tokenizer.train(corpus, verbose=False)
    t1 = time.time()
    print(f"Tokenizer trained: vocab_size={tokenizer.get_vocab_size()} train_time={t1-t0:.3f}s")

    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_seq_length=128,
        dropout=0.1
    )
    model.to(device)
    model.eval()

    prompt = "Artificial intelligence is transforming"
    t0 = time.time()
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    t1 = time.time()
    print(f"Tokenize: {len(input_ids)} tokens, time={t1-t0:.4f}s")

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        t0 = time.time()
        src_mask = model.create_padding_mask(input_tensor, tokenizer.pad_token_id)
        encoder_output = model.encode(input_tensor, src_mask)
        t1 = time.time()
    print(f"Encoder forward: time={t1-t0:.4f}s, encoder_output shape={encoder_output.shape}")

    with torch.no_grad():
        seq_len = input_tensor.size(1)
        tgt_mask = model.create_causal_mask(seq_len, device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
        t0 = time.time()
        dec_res = model.decode(input_tensor, encoder_output, src_mask, tgt_mask, use_cache=False)
        t1 = time.time()
        if isinstance(dec_res, tuple):
            dec_out = dec_res[0]
        else:
            dec_out = dec_res
    print(f"Decoder forward (full decode): time={t1-t0:.4f}s, dec_out shape={dec_out.shape}")

    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=device)
    t0 = time.time()
    out = engine.generate(prompt, max_length=20, method='greedy', log_to_db=False)
    t1 = time.time()
    print(f"InferenceEngine.generate (greedy) time={t1-t0:.4f}s")
    print("Generated:", out)

if __name__ == '__main__':
    main()