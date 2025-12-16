import torch
import time
import re
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

# Check for GGUF support
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    Llama = None

class InferenceEngine:
    def __init__(self, model, tokenizer=None, db_manager=None, device=None):
        self.model = model
        
        # Check if this is a GGUF model (llama-cpp-python Llama instance)
        self.is_gguf = GGUF_AVAILABLE and isinstance(model, Llama)
        
        # For GGUF models, get tokenizer from model if not provided
        if self.is_gguf:
            if hasattr(model, '_gguf_tokenizer'):
                self.tokenizer = model._gguf_tokenizer
            elif tokenizer is None:
                # Create tokenizer wrapper if not provided
                try:
                    from qkv_core.formats.gguf_loader import GGUFTokenizerWrapper
                    self.tokenizer = GGUFTokenizerWrapper(model)
                except:
                    self.tokenizer = None
            else:
                self.tokenizer = tokenizer
        else:
            self.tokenizer = tokenizer
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if self.model and not self.is_gguf:
            self.model.to(self.device)
            self.model.eval()
        
        self.eos_token_id = self._get_eos_token_id()

    def _get_eos_token_id(self):
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, 'special_tokens'):
            special = self.tokenizer.special_tokens
            if isinstance(special, dict) and 'eos' in special:
                eos_token = special['eos']
                if hasattr(self.tokenizer, 'vocabulary') and eos_token in self.tokenizer.vocabulary:
                    return self.tokenizer.vocabulary[eos_token]
        return 50256

    def _apply_repetition_penalty(self, logits, generated_tokens, repetition_penalty):
        if repetition_penalty == 1.0 or not generated_tokens:
            return logits
        unique_tokens = list(set(generated_tokens))
        for token_id in unique_tokens:
            if token_id < logits.size(-1):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        return logits

    def _clean_text(self, text):
        """Gelişmiş metin temizleme ve encoding düzeltme"""
        if not text: return ""
        
        # 1. Byte-level artifact temizliği
        text = text.replace('Ġ', ' ')
        text = text.replace('Ċ', '\n')
        text = text.replace('<|endoftext|>', '')
        
        # 2. Bozuk karakterleri () temizle veya düzelt
        # Bu karakter genellikle "Replacement Character" (\ufffd) olarak gelir
        text = text.replace('\ufffd', '') 
        
        # 3. Yaygın Encoding Hatalarını Manuel Düzelt (Smart Quotes vb.)
        # GPT-2 bazen bunları byte olarak basar
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€”': '-',
            'Ã©': 'é',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        return text

    def generate_stream(
        self,
        prompt: str,
        max_length: int = 100,
        method: str = 'sample',
        temperature: float = 0.7, # 0.6-0.7 base model için idealdir
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        use_kv_cache: bool = True,
        min_length: int = 0
    ):
        # Use GGUF native generation if model is GGUF
        if self.is_gguf:
            yield from self._generate_stream_gguf(
                prompt, max_length, temperature, top_k, top_p, repetition_penalty
            )
            return
        try:
            if hasattr(self.tokenizer, 'encode'):
                input_ids = self.tokenizer.encode(prompt)
                if not input_ids: input_ids = [50256]
            else:
                input_ids = [50256]
        except:
            input_ids = [50256]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        current_ids = input_tensor
        generated_tokens_list = []
        past_key_values = None
        
        previous_decoded_text = ""
        
        # Prompt'un kendisini stream output'a dahil etmemek için uzunluğunu al
        prompt_len = len(prompt) 
        # Ancak decode ederken BPE yüzünden karakter sayısı tutmayabilir,
        # bu yüzden metin tabanlı fark (diff) alacağız.

        with torch.no_grad():
            for _ in range(max_length):
                if use_kv_cache and past_key_values is not None:
                    model_input = current_ids[:, -1:]
                else:
                    model_input = current_ids

                outputs = self.model(model_input, past_key_values=past_key_values)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    if use_kv_cache: past_key_values = outputs[1]
                else:
                    logits = outputs
                    past_key_values = None

                next_token_logits = logits[:, -1, :]
                
                # Repetition Penalty (Tüm context'e bak)
                all_context_tokens = input_ids + generated_tokens_list
                if repetition_penalty > 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, all_context_tokens, repetition_penalty
                    )

                # Sampling
                if method == 'sample':
                    if temperature > 0: next_token_logits /= temperature
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    if torch.isnan(probs).any() or torch.sum(probs) == 0:
                        probs = torch.ones_like(probs) / probs.size(-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                token_id = next_token.item()
                if token_id == self.eos_token_id or token_id == 50256:
                    break
                
                generated_tokens_list.append(token_id)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # --- GÜÇLENDİRİLMİŞ STREAM DECODING ---
                try:
                    if hasattr(self.tokenizer, 'decode'):
                        # Sadece yeni üretilenleri decode et
                        # errors='ignore' veya 'replace' kullanarak Python'un çökmesini engelle
                        current_text_chunk = self.tokenizer.decode(generated_tokens_list)
                        current_text_chunk = self._clean_text(current_text_chunk)
                        
                        # Farkı bul (Differential)
                        if len(current_text_chunk) > len(previous_decoded_text):
                            new_content = current_text_chunk[len(previous_decoded_text):]
                            previous_decoded_text = current_text_chunk
                            
                            # Ekrana basarken başta oluşan gereksiz boşlukları (eğer cümlenin başıysa) temizle
                            # Ama kelime ortasındaysa dokunma
                            yield new_content
                except:
                    pass
    
    def _generate_stream_gguf(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ):
        """
        Generate text using GGUF model's native streaming API.
        This is optimized for llama-cpp-python models.
        """
        try:
            # Qwen2.5-Instruct uses a specific chat template
            # Check if prompt already has the template format
            if "<|im_start|>" not in prompt and "User:" in prompt:
                # Convert simple User: prompt to Qwen format
                user_message = prompt.replace("User:", "").replace("AI:", "").strip()
                formatted_prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = prompt
            
            # Ensure minimum max_tokens for reasonable responses
            # max_length is in tokens, ensure at least 64 for basic responses
            max_tokens = max(max_length, 64)
            
            # Less aggressive stop tokens - only stop on actual end tokens
            # Removed "\n\n\n" as it's too aggressive and cuts off code/technical responses
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            
            # Try streaming first, fallback to non-streaming if it fails
            try:
                # Use the GGUF model's native streaming generation
                stream = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
                    stream=True,
                    stop=stop_tokens,
                )
                
                # Track cumulative text to extract only new chunks (incremental)
                cumulative_text = ""
                
                # llama-cpp-python 0.2.90 returns CompletionChunk objects
                # Each chunk has a 'choices' list with one choice
                for output in stream:
                    text_chunk = None
                    
                    # Format 1: CompletionChunk with choices[0].text (llama-cpp-python 0.2.x)
                    if hasattr(output, 'choices') and len(output.choices) > 0:
                        choice = output.choices[0]
                        
                        # Check finish_reason - if stopped, we're done
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            # If finished, yield any remaining text and break
                            break
                        
                        # Direct text attribute on choice (0.2.90 format)
                        if hasattr(choice, 'text') and choice.text:
                            text_chunk = choice.text
                        # Or check delta
                        elif hasattr(choice, 'delta'):
                            delta = choice.delta
                            if hasattr(delta, 'text') and delta.text:
                                text_chunk = delta.text
                            elif hasattr(delta, 'content') and delta.content:
                                text_chunk = delta.content
                    
                    # Format 2: Dict format
                    elif isinstance(output, dict):
                        if 'choices' in output and len(output['choices']) > 0:
                            choice = output['choices'][0]
                            # Check finish_reason in dict format
                            if isinstance(choice, dict) and choice.get('finish_reason'):
                                break
                            
                            if isinstance(choice, dict):
                                text_chunk = choice.get('text') or choice.get('delta', {}).get('text') or choice.get('delta', {}).get('content', '')
                            elif hasattr(choice, 'text'):
                                text_chunk = choice.text
                        elif 'text' in output:
                            text_chunk = output['text']
                    
                    # Format 3: Direct attributes
                    elif hasattr(output, 'text') and output.text:
                        text_chunk = output.text
                    elif isinstance(output, str):
                        text_chunk = output
                    
                    # Process the text chunk - handle cumulative vs incremental
                    if text_chunk:
                        # Some llama-cpp-python versions return cumulative text, others return incremental
                        # Check if this is cumulative (contains previous text) or incremental (new only)
                        if text_chunk.startswith(cumulative_text):
                            # Cumulative format - extract only new part
                            new_text = text_chunk[len(cumulative_text):]
                            cumulative_text = text_chunk
                        else:
                            # Incremental format - use as is
                            new_text = text_chunk
                            cumulative_text += new_text
                        
                        # Yield only if there's new content
                        if new_text:
                            yield new_text
                        
            except Exception as stream_error:
                # Fallback to non-streaming if streaming fails
                try:
                    result = self.model(
                        formatted_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repeat_penalty=repetition_penalty,
                        stream=False,
                        stop=stop_tokens,
                    )
                    
                    # Extract text from non-streaming result
                    text = ""
                    if hasattr(result, 'choices') and len(result.choices) > 0:
                        choice = result.choices[0]
                        if hasattr(choice, 'text'):
                            text = choice.text
                        elif hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            text = choice.message.content
                    elif isinstance(result, dict):
                        if 'choices' in result and len(result['choices']) > 0:
                            choice = result['choices'][0]
                            text = choice.get('text', '') if isinstance(choice, dict) else getattr(choice, 'text', '')
                    
                    if text:
                        # Yield as chunks for consistency
                        chunk_size = 20
                        for i in range(0, len(text), chunk_size):
                            yield text[i:i+chunk_size]
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield f"\n[Error during GGUF generation: {str(e)}]"
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n[Error during GGUF generation: {str(e)}]"
    
    # generate fonksiyonu aynen kalabilir, stream bizim için önemli olan.
    def generate(self, *args, **kwargs):
        # (Eski kodundaki generate metodunu buraya koyabilirsin, değişiklik gerekmez)
        pass