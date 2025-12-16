import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import random

class TextDataset(Dataset):
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        mode: str = 'pretrain',
        lazy_loading: bool = True
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.lazy_loading = lazy_loading
        
        if lazy_loading:
            self.encoded_texts = None
        else:
            self.encoded_texts = []
            
            for text in texts:
                encoded = tokenizer.encode(text, add_special_tokens=True)
                
                if len(encoded) > max_length:
                    encoded = encoded[:max_length]
                
                self.encoded_texts.append(encoded)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.lazy_loading:
            text = self.texts[idx]
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
        else:
            encoded = self.encoded_texts[idx]
        
        if self.mode == 'pretrain':
            
            if not encoded:
                encoded = [self.tokenizer.unk_token_id]

            src = [self.tokenizer.bos_token_id] + encoded + [self.tokenizer.eos_token_id]
            
            if len(src) > self.max_length:
                src = src[:self.max_length]
            
            tgt_input = [self.tokenizer.bos_token_id] + encoded
            if len(tgt_input) > self.max_length:
                tgt_input = tgt_input[:self.max_length]
            
            tgt_output = encoded + [self.tokenizer.eos_token_id]
            if len(tgt_output) > self.max_length:
                tgt_output = tgt_output[:self.max_length]
        
        else:
            # Fallback / other modes
            src = [self.tokenizer.bos_token_id] + encoded + [self.tokenizer.eos_token_id]
            if len(src) > self.max_length:
                src = src[:self.max_length]

            tgt_input = [self.tokenizer.bos_token_id] + encoded
            if len(tgt_input) > self.max_length:
                tgt_input = tgt_input[:self.max_length]

            tgt_output = encoded + [self.tokenizer.eos_token_id]
            if len(tgt_output) > self.max_length:
                tgt_output = tgt_output[:self.max_length]
        
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long)
        )
    
    @staticmethod
    def collate_fn(batch, pad_token_id: int = 0):
        
        src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)
        
        # Determine lengths
        src_lengths = [len(seq) for seq in src_batch]
        max_src_len = max(src_lengths)
        
        # Pad Source
        src_padded = torch.full((len(batch), max_src_len), pad_token_id, dtype=torch.long)
        for i, src in enumerate(src_batch):
            src_padded[i, :len(src)] = src
        
        # Pad Target Input
        tgt_input_lengths = [len(t) for t in tgt_input_batch]
        max_tgt_input_len = max(tgt_input_lengths)
        
        tgt_input_padded = torch.full((len(batch), max_tgt_input_len), pad_token_id, dtype=torch.long)
        for i, tgt in enumerate(tgt_input_batch):
            tgt_input_padded[i, :len(tgt)] = tgt
        
        # Pad Target Output
        tgt_output_lengths = [len(t) for t in tgt_output_batch]
        max_tgt_output_len = max(tgt_output_lengths)
        
        tgt_output_padded = torch.full((len(batch), max_tgt_output_len), pad_token_id, dtype=torch.long)
        for i, tgt in enumerate(tgt_output_batch):
            tgt_output_padded[i, :len(tgt)] = tgt
        
        # Create Mask (1 for tokens, 0 for padding)
        src_mask = (src_padded != pad_token_id).unsqueeze(1).unsqueeze(2)
        
        return src_padded, tgt_input_padded, tgt_output_padded, src_mask

class IncrementalDataset(Dataset):
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        initial_texts: List[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_texts = []
        
        if initial_texts:
            self.add_texts(initial_texts)
    
    def add_texts(self, texts: List[str]):
        print(f"Adding {len(texts)} new texts to incremental dataset...")
        
        for text in texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            
            self.encoded_texts.append(encoded)
        
        print(f"Total samples: {len(self.encoded_texts)}")
    
    def __len__(self) -> int:
        return len(self.encoded_texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        encoded = self.encoded_texts[idx]
        
        # Ensure minimum length
        if len(encoded) < 2:
            encoded = encoded + [self.tokenizer.pad_token_id] * (2 - len(encoded))
        
        src = encoded
        tgt_input = [self.tokenizer.bos_token_id] + encoded[:-1]
        tgt_output = encoded
        
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long)
        )