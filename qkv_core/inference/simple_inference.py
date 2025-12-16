import torch
import time

class SimpleInference:
    
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_simple(self, prompt: str, max_tokens: int = 20) -> str:
        
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
            if not input_ids:
                input_ids = [1]
        else:
            return ""
        
        current_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_text = ""
        
        with torch.no_grad():
            for i in range(max_tokens):
                outputs = self.model(current_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[:, -1, :]
                
                if i < 3:
                    next_token_logits[:, 50256] = float('-inf')
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                token_val = next_token.item()
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                if hasattr(self.tokenizer, 'decode'):
                    token_str = self.tokenizer.decode([token_val], skip_special_tokens=False)
                    if token_str not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<|endoftext|>", "</s>", ""]:
                        generated_text += token_str
                
                if token_val == 50256:
                    break
        
        return generated_text.strip()
    
    def generate_stream(self, prompt: str, max_tokens: int = 20):
        
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
            if not input_ids:
                input_ids = [1]
        else:
            return
        
        current_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        accumulated_text = ""
        stream_batch_size = 5
        
        with torch.no_grad():
            for i in range(max_tokens):
                outputs = self.model(current_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[:, -1, :]
                
                if i < 3:
                    next_token_logits[:, 50256] = float('-inf')
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                token_val = next_token.item()
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                if hasattr(self.tokenizer, 'decode'):
                    token_str = self.tokenizer.decode([token_val], skip_special_tokens=False)
                    if token_str not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<|endoftext|>", "</s>", ""]:
                        accumulated_text += token_str
                        
                        if (i + 1) % stream_batch_size == 0:
                            yield accumulated_text
                
                if token_val == 50256:
                    break
        
        yield accumulated_text