import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in usage in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GPT2Attention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x, past_key_value=None):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=-2)
            v = torch.cat([prev_v, v], dim=-2)
            
        current_key_value = (k, v)

        # Causal Self-Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        seq_len = k.size(-2)
        q_len = q.size(-2) 
        att = att.masked_fill(self.bias[:,:, seq_len-q_len:seq_len, :seq_len] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v 
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y), current_key_value

class GPT2MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.act = NewGELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, max_len=1024):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(d_model, num_heads, max_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, d_ff, dropout)

    def forward(self, x, past_key_value=None):
        ln_1_x = self.ln_1(x)
        attn_out, current_cache = self.attn(ln_1_x, past_key_value)
        x = x + attn_out
        
        ln_2_x = self.ln_2(x)
        mlp_out = self.mlp(ln_2_x)
        x = x + mlp_out
        return x, current_cache

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.get('vocab_size', 50257)
        self.d_model = config.get('d_model', 768)
        self.max_len = config.get('max_seq_length', 1024)
        self.num_layers = config.get('num_layers', 12)
        
        self.wte = nn.Embedding(self.vocab_size, self.d_model)
        self.wpe = nn.Embedding(self.max_len, self.d_model)
        self.drop = nn.Dropout(config.get('dropout', 0.1))
        
        self.h = nn.ModuleList([
            GPT2Block(
                self.d_model, 
                config.get('num_heads', 12), 
                config.get('d_ff', 4 * self.d_model),
                config.get('dropout', 0.1),
                self.max_len
            ) for _ in range(self.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(self.d_model)
        
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)
        # Weight tying (optional but common in GPT/Transformer)
        self.output_projection.weight = self.wte.weight

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, past_key_values=None):
        device = input_ids.device
        b, t = input_ids.size()
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
            
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        
        if t + past_length > self.max_len:
             pos = torch.clamp(pos, max=self.max_len-1)

        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        new_past_key_values = []
        for i, block in enumerate(self.h):
            x, cache = block(x, past_key_values[i])
            new_past_key_values.append(cache)
            
        x = self.ln_f(x)
        
        # Calculate logits using the output projection (weights tied to embedding)
        logits = self.output_projection(x)
        
        return logits, new_past_key_values