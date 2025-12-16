import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from qkv_core.core.transformer import TransformerModel
from qkv_core.inference.inference import InferenceEngine

class BatchInferenceEngine(InferenceEngine):
    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 100,
        method: str = 'greedy',
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
        return_times: bool = False
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        batch_size = len(prompts)
        
        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        else:
            pad_token_id = self.tokenizer.pad_token_id

        input_ids_list = []
        max_input_len = 0
        
        for prompt in prompts:
            try:
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                input_ids_list.append(input_ids)
                max_input_len = max(max_input_len, len(input_ids))
            except Exception as e:
                print(f"Error encoding prompt '{prompt[:50]}...': {e}")
                input_ids_list.append([pad_token_id])
        
        padded_ids = []
        attention_masks = []
        
        for ids in input_ids_list:
            pad_len = max_input_len - len(ids)
            padded = ids + [pad_token_id] * pad_len
            
            mask = [1] * len(ids) + [0] * pad_len
            
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        input_tensor = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
        
        if method == 'greedy':
            output_ids_list = self._generate_greedy_batch(
                input_tensor, attention_mask, max_length, repetition_penalty, pad_token_id
            )
        elif method == 'sample':
            output_ids_list = self._generate_sample_batch(
                input_tensor, attention_mask, max_length, temperature, top_k, top_p, repetition_penalty, pad_token_id
            )
        elif method == 'beam':
            # Fallback to greedy for now or implement beam search
            output_ids_list = self._generate_greedy_batch(
                input_tensor, attention_mask, max_length, repetition_penalty, pad_token_id
            )
        else:
            raise ValueError(f"Unknown generation method: {method}")
        
        outputs = []
        times = []
        total_tokens = 0
        
        for i, output_ids in enumerate(output_ids_list):
            try:
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                outputs.append(generated_text)
                
                gen_len = len(output_ids)
                total_tokens += gen_len
                times.append(gen_len / max(time.time() - start_time, 0.001))
            except Exception as e:
                outputs.append(f"Error decoding: {str(e)}")
                times.append(0)
        
        elapsed = time.time() - start_time
        throughput = total_tokens / max(elapsed, 0.001)
        
        result = {
            'outputs': outputs,
            'total_time': elapsed,
            'throughput': throughput,
            'batch_size': batch_size,
            'avg_tokens': total_tokens / batch_size if batch_size > 0 else 0
        }
        
        if return_times:
            result['times'] = times
        
        return result
    
    def _generate_greedy_batch(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0
    ) -> List[List[int]]:
        
        batch_size, seq_len = input_tensor.shape
        
        encoder_output = None
        if hasattr(self.model, 'encode') and not getattr(self.model, 'is_decoder_only', False):
             try:
                 with torch.no_grad():
                    encoder_output = self.model.encode(input_tensor, attention_mask)
             except:
                 encoder_output = None

        current_ids = input_tensor
        generated_ids = [[] for _ in range(batch_size)]
        
        past_key_values = None
        
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(max_length):
            with torch.no_grad():
                outputs = self.model.decode(
                    current_ids,
                    encoder_output=encoder_output,
                    src_mask=attention_mask if encoder_output is not None else None,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    past_key_values = outputs[1]
                else:
                    logits = outputs
                    past_key_values = None

            next_token_logits = logits[:, -1, :]
            
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in generated_ids[i]:
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty
            
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            eos_id = getattr(self.tokenizer, 'eos_token_id', 50256)
            is_eos = (next_tokens == eos_id)
            
            is_eos = is_eos | (next_tokens == 50256)
            
            if pad_token_id is None: pad_token_id = 0
            
            next_tokens = next_tokens.masked_fill(finished_sequences, pad_token_id)
            
            finished_sequences = finished_sequences | is_eos
            
            next_tokens = next_tokens.masked_fill(finished_sequences, pad_token_id)
            
            # Double check against tokenizer eos if different
            if hasattr(self.tokenizer, 'eos_token_id'):
                is_eos = (next_tokens == self.tokenizer.eos_token_id)
                finished_sequences = finished_sequences | is_eos
            
            current_ids = next_tokens.unsqueeze(-1)
            
            for i in range(batch_size):
                if not finished_sequences[i].item() or is_eos[i].item():
                     generated_ids[i].append(next_tokens[i].item())
            
            if finished_sequences.all():
                break
                
        return generated_ids
    
    def _generate_sample_batch(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0
    ) -> List[List[int]]:
        
        batch_size, seq_len = input_tensor.shape
        
        encoder_output = None
        if hasattr(self.model, 'encode') and not getattr(self.model, 'is_decoder_only', False):
             try:
                 with torch.no_grad():
                    encoder_output = self.model.encode(input_tensor, attention_mask)
             except:
                 encoder_output = None

        current_ids = input_tensor
        generated_ids = [[] for _ in range(batch_size)]
        past_key_values = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_length):
            with torch.no_grad():
                outputs = self.model.decode(
                    current_ids,
                    encoder_output=encoder_output,
                    src_mask=attention_mask if encoder_output is not None else None,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    past_key_values = outputs[1]
                else:
                    logits = outputs

            next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in generated_ids[i]:
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty

            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            if temperature < 1e-5:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            else:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            next_tokens = next_tokens.masked_fill(finished_sequences, pad_token_id)
            
            # Check for EOS
            is_eos = (next_tokens == 50256)
            if hasattr(self.tokenizer, 'eos_token_id'):
                is_eos = is_eos | (next_tokens == self.tokenizer.eos_token_id)
            
            finished_sequences = finished_sequences | is_eos
            
            current_ids = next_tokens.unsqueeze(-1)
            
            for i in range(batch_size):
                if not finished_sequences[i].item() or is_eos[i].item():
                    generated_ids[i].append(next_tokens[i].item())
            
            if finished_sequences.all():
                break
        
        return generated_ids