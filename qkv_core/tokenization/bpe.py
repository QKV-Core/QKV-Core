import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import pickle

class BPETokenizer:
    
    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        if special_tokens is None:
            special_tokens = {
                "pad": "<PAD>",
                "unk": "<UNK>",
                "bos": "<BOS>",
                "eos": "<EOS>",
                "mask": "<MASK>"
            }
        
        self.special_tokens = special_tokens
        self.vocabulary = {}
        self.reverse_vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        
        self._is_trained = False
    
    def _get_words_from_corpus(self, corpus: List[str]) -> Dict[str, int]:
        
        word_freq = Counter()
        
        for text in corpus:
            # Regex adjusted: \is -> \s
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            word_freq.update(words)
        
        return dict(word_freq)
    
    def _get_pair_statistics(self, word_freq: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        
        pairs = defaultdict(int)
        
        for word, freq in word_freq.items():
            if len(word) < 2:
                continue
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        
        return dict(pairs)
    
    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_freq: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        
        new_word_freq = {}
        
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freq.items():
            word_str = ' '.join(word)
            new_word_str = word_str.replace(bigram, replacement)
            new_word = tuple(new_word_str.split())
            new_word_freq[new_word] = freq
        
        return new_word_freq
    
    def train(self, corpus: List[str], verbose: bool = True):
        
        if verbose:
            print(f"Training BPE tokenizer on {len(corpus)} documents...")
            print(f"Target vocabulary size: {self.vocab_size}")
        
        word_freq_raw = self._get_words_from_corpus(corpus)
        
        # Filter by minimum frequency
        word_freq_raw = {w: f for w, f in word_freq_raw.items() if f >= self.min_frequency}
        
        word_freq = {}
        for word, freq in word_freq_raw.items():
            char_tuple = tuple(list(word) + ['</w>'])
            word_freq[char_tuple] = freq
        
        vocabulary = set()
        for word in word_freq.keys():
            vocabulary.update(word)
        
        if verbose:
            print(f"Initial vocabulary size (characters): {len(vocabulary)}")
        
        num_merges = self.vocab_size - len(vocabulary) - len(self.special_tokens)
        
        for i in range(num_merges):
            pairs = self._get_pair_statistics(word_freq)
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            word_freq = self._merge_pair(best_pair, word_freq)
            
            merged_token = ''.join(best_pair)
            vocabulary.add(merged_token)
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {merged_token} (freq: {pairs[best_pair]})")
        
        self._build_vocabulary(vocabulary)
        
        self._is_trained = True
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocabulary)}")
            print(f"Number of merges: {len(self.merges)}")
    
    def _build_vocabulary(self, vocabulary: set):
        
        self.token_to_id = {
            self.special_tokens["pad"]: self.pad_token_id,
            self.special_tokens["unk"]: self.unk_token_id,
            self.special_tokens["bos"]: self.bos_token_id,
            self.special_tokens["eos"]: self.eos_token_id,
            self.special_tokens["mask"]: self.mask_token_id,
        }
        
        idx = len(self.token_to_id)
        for token in sorted(vocabulary):
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                idx += 1
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        self.vocabulary = self.token_to_id
        self.reverse_vocab = self.id_to_token
    
    def _apply_merges(self, word: str) -> List[str]:
        
        chars = list(word) + ['</w>']
        
        for pair in self.merges:
            i = 0
            while i < len(chars) - 1:
                if (chars[i], chars[i + 1]) == pair:
                    chars[i:i + 2] = [''.join(pair)]
                else:
                    i += 1
        
        return chars
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # Regex adjusted: \is -> \s
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for word in words:
            subwords = self._apply_merges(word)
            
            for subword in subwords:
                token_id = self.token_to_id.get(subword, self.unk_token_id)
                token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        tokens = []
        
        for token_id in token_ids:
            if skip_special_tokens:
                special_ids = []
                if self.pad_token_id is not None:
                    special_ids.append(self.pad_token_id)
                if self.bos_token_id is not None:
                    special_ids.append(self.bos_token_id)
                if self.eos_token_id is not None:
                    special_ids.append(self.eos_token_id)
                if self.mask_token_id is not None:
                    special_ids.append(self.mask_token_id)
                
                if token_id == 0 and (self.pad_token_id is None or self.pad_token_id != 0):
                    continue
                
                if token_id in special_ids:
                    continue
            
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens:
                    special_token_strings = [
                        self.special_tokens.get("pad", "<PAD>"),
                        self.special_tokens.get("unk", "<UNK>"),
                        self.special_tokens.get("bos", "<BOS>"),
                        self.special_tokens.get("eos", "<EOS>"),
                        self.special_tokens.get("mask", "<MASK>"),
                        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>",
                        "<|endoftext|>",
                    ]
                    if token in special_token_strings:
                        continue
            else:
                if skip_special_tokens:
                    continue
                else:
                    token = self.special_tokens.get("unk", "<UNK>")
            tokens.append(token)
        
        text = ''.join(tokens)
        
        # Post-processing
        text = text.replace('</w>', ' ')
        
        # Clean up artifacts (common in BPE decoding)
        text = text.replace('Ġ', ' ')
        text = text.replace('´', "'")
        text = text.replace('Â', '')
        text = text.replace('’', "'")
        text = text.replace('“', '"')
        text = text.replace('”', '"')
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        
        import re
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def save(self, filepath: str):
        
        tokenizer_state = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'merges': self.merges,
            'is_trained': self._is_trained
        }
        
        # Changed 'if' to 'f'
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_state, f)
        
        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print(f"📦 Adapter tokenizer detected and loaded.")
            return data
        
        tokenizer_state = data
        tokenizer = cls(
            vocab_size=tokenizer_state['vocab_size'],
            min_frequency=tokenizer_state['min_frequency'],
            special_tokens=tokenizer_state['special_tokens']
        )
        
        tokenizer.token_to_id = tokenizer_state['token_to_id']
        tokenizer.id_to_token = tokenizer_state['id_to_token']
        tokenizer.merges = tokenizer_state['merges']
        tokenizer._is_trained = tokenizer_state['is_trained']
        
        tokenizer.vocabulary = tokenizer.token_to_id
        tokenizer.reverse_vocab = tokenizer.id_to_token
        
        if tokenizer_state.get('from_huggingface', False):
            special_ids = tokenizer_state.get('special_token_ids', {})
            if 'eos' in special_ids: tokenizer.eos_token_id = special_ids['eos']
        
        return tokenizer

    def get_vocab_size(self) -> int:
        return len(self.vocabulary)
    
    def is_trained(self) -> bool:
        return self._is_trained

class HuggingFaceTokenizerAdapter:
    
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.vocabulary = hf_tokenizer.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocabulary.items()}
        
        self.eos_token_id = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else 50256
        self.bos_token_id = hf_tokenizer.bos_token_id if hf_tokenizer.bos_token_id is not None else 50256
        self.unk_token_id = hf_tokenizer.unk_token_id if hf_tokenizer.unk_token_id is not None else 50256
        self.pad_token_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 50256
        
        self._is_trained = True
        self.merges = []

    def encode(self, text, add_special_tokens=True):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids, skip_special_tokens=True):
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self):
        return self._tokenizer.vocab_size
        
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)