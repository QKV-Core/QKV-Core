import torch

def generate(model, tokenizer, prompt: str, device, max_tokens: int = 20) -> str:
    
    if isinstance(device, torch.device):
        device_str = str(device).replace("device(", "").replace(")", "").replace("'", "")
    else:
        device_str = device
    
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [1]
    else:
        return ""
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device_str)
    current_ids = input_tensor
    
    generated_text = ""
    
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(current_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            next_token_logits = logits[:, -1, :]
            
            if i < 3:
                next_token_logits[:, 50256] = float('-inf')
            
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            token_val = next_token.item()
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if hasattr(tokenizer, 'decode'):
                token_str = tokenizer.decode([token_val], skip_special_tokens=False)
                
                if token_str not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<|endoftext|>", "</s>", ""]:
                    generated_text += token_str
            
            if token_val == 50256:
                break
    
    return generated_text.strip()

def generate_stream(model, tokenizer, prompt: str, device, max_tokens: int = 20):
    
    if isinstance(device, torch.device):
        device_str = str(device).replace("device(", "").replace(")", "").replace("'", "")
    else:
        device_str = device
    
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [1]
    else:
        return
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device_str)
    current_ids = input_tensor
    
    accumulated_text = ""
    stream_batch_size = 5
    
    with torch.no_grad():
        for i in range(max_tokens):
            outputs = model(current_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            next_token_logits = logits[:, -1, :]
            
            if i < 3:
                next_token_logits[:, 50256] = float('-inf')
            
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            token_val = next_token.item()
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if hasattr(tokenizer, 'decode'):
                token_str = tokenizer.decode([token_val], skip_special_tokens=False)
                
                if token_str not in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<|endoftext|>", "</s>", ""]:
                    accumulated_text += token_str
                    
                    if (i + 1) % stream_batch_size == 0:
                        yield accumulated_text
            
            if token_val == 50256:
                break
    
    yield accumulated_text