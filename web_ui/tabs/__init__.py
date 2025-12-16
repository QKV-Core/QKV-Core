"""
Tabs module for QKV Core Web Interface.

This module contains all tab components for the Gradio interface.
"""
import sys
from pathlib import Path

# Add parent directories to path for imports
# This ensures that imports like 'from utils.logger' and 'from state.app_state' work correctly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Tabs module
__all__ = [
    'create_home_tab',
    'create_tokenizer_tab',
    'create_training_tab',
    'create_lora_tab',
    'create_finetune_tab',
    'create_rlhf_tab',
    'create_download_model_tab',
    'create_chat_tab',
    'create_statistics_tab',
]
