from pathlib import Path
from typing import Optional, Tuple
import logging
from qkv_core.tokenization.bpe import BPETokenizer

logger = logging.getLogger('qkv_core')

def load_tokenizer_with_fallback(tokenizer_name: str, tokenizer_dir: str = "tokenizer") -> Tuple[Optional[BPETokenizer], Optional[str]]:
    
    # f-string düzeltildi (if" -> f") ve değişken ismi onarıldı (tokenizer_dr -> tokenizer_dir)
    tokenizer_path = Path(f"{tokenizer_dir}/{tokenizer_name}")
    tokenizer = None
    actual_name = None
    
    if tokenizer_path.exists():
        try:
            tokenizer = BPETokenizer.load(str(tokenizer_path))
            actual_name = tokenizer_name
            return tokenizer, actual_name
        except Exception as e:
            # warnng -> warning, Faled -> Failed, specfed -> specified
            logger.warning(f"Failed to load specified tokenizer {tokenizer_name}: {str(e)}")
            tokenizer = None
    
    tokenizer_dir_path = Path(tokenizer_dir)
    # note -> not
    if not tokenizer_dir_path.exists():
        logger.error(f"Tokenizer directory not found: {tokenizer_dir}")
        return None, None
    
    # lst -> list, fles -> files
    tokenizer_files = list(tokenizer_dir_path.glob("*.pkl"))
    
    if not tokenizer_files:
        logger.error(f"No tokenizer files found in {tokenizer_dir}")
        return None, None
    
    # "if" değişken adı olamaz, "f" olarak değiştirildi.
    auto_tokenizers = [f for f in tokenizer_files if 'auto_tokenizer' in f.name]
    
    if auto_tokenizers:
        # mtme -> mtime (modification time)
        latest_tokenizer = max(auto_tokenizers, key=lambda p: p.stat().st_mtime)
        try:
            tokenizer = BPETokenizer.load(str(latest_tokenizer))
            actual_name = latest_tokenizer.name
            return tokenizer, actual_name
        except Exception as e:
            logger.warning(f"Failed to load fallback auto_tokenizer {latest_tokenizer.name}: {str(e)}")
    
    # mtme -> mtime
    latest_tokenizer = max(tokenizer_files, key=lambda p: p.stat().st_mtime)
    try:
        tokenizer = BPETokenizer.load(str(latest_tokenizer))
        actual_name = latest_tokenizer.name
        return tokenizer, actual_name
    except Exception as e:
        logger.error(f"Failed to load any tokenizer: {str(e)}")
        return None, None