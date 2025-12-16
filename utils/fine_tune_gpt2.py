import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

from qkv_core.core.transformer import TransformerModel
from qkv_core.formats.huggingface_converter import HuggingFaceConverter
from qkv_core.utils.logger import get_logger

logger = get_logger()

def prepare_gpt2_for_finetuning(
    checkpoint_path: str,
    tokenizer_path: str,
    output_name: Optional[str] = None,
    target_config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
    else:
        raise ValueError("Checkpoint does not contain 'config' key")
    
    if target_config:
        final_config = {**saved_config, **target_config}
    else:
        final_config = saved_config
    
    if output_name is None:
        # Fixed: f-string syntax and time module
        output_name = f"gpt2_finetune_{int(time.time())}"
    
    output_checkpoint_path = Path("model_weights") / f"{output_name}.pt"
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fixed: Checkpoint structure restored
    fine_tune_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'config': final_config,
        'epoch': 0,
        'global_step': 0,
        'best_loss': float('inf'),  # Fixed: 'nf' -> 'inf' (infinity)
        'source': 'gpt2_finetune',
        'original_checkpoint': str(checkpoint_path),
        'prepared_at': time.time(),
        'ready_for_training': True
    }
    
    torch.save(fine_tune_checkpoint, output_checkpoint_path)
    
    return {
        'checkpoint_path': str(output_checkpoint_path),
        'tokenizer_path': str(tokenizer_path),
        'config': final_config
    }

def create_finetune_dataset_from_text(
    text_file: str,
    output_file: str,
    max_lines: Optional[int] = None
) -> int:
    
    text_path = Path(text_file)
    output_path = Path(output_file)
    
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines_processed = 0
    # Fixed: f_n -> f_in, wrte -> write, strp -> strip errors
    with open(text_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                # Fixed: \in -> \n (Newline)
                f_out.write(line + '\n')
                lines_processed += 1
                if max_lines and lines_processed >= max_lines:
                    break
                # Fixed: Empty if block filled
                if lines_processed % 10000 == 0:
                    logger.info(f"Processed {lines_processed} lines...")
    
    return lines_processed