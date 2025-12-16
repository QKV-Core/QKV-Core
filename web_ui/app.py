"""
QKV Core Web Interface - Main Application

QKV Core (Query-Key-Value Core) - The Core of Transformer Intelligence

This is the main entry point for the Gradio web interface.
All tab components have been refactored into separate modules in the tabs/ directory.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path
# This must be done before any imports from parent directories
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
_web_ui_root = _current_file.parent

# Ensure project root is in path FIRST (before any imports)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_web_ui_root) not in sys.path:
    sys.path.insert(0, str(_web_ui_root))

# Import logger first (before config import attempts)
from qkv_core.utils.logger import get_logger, setup_logging

# Setup logging early
setup_logging("logs")
logger = get_logger()

# Verify config module is importable
# Ensure project root is in path before importing config
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import config
    import config.model_config
    logger.debug("Config module imported successfully")
except ImportError as e:
    # Try alternative import method
    try:
        import importlib.util
        config_path = _project_root / "config" / "model_config.py"
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("config.model_config", config_path)
            if spec and spec.loader:
                config_model_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_model_config)
                # Make it available as config.model_config
                if not hasattr(config, 'model_config'):
                    import types
                    config.model_config = types.ModuleType('model_config')
                    for attr in dir(config_model_config):
                        if not attr.startswith('_'):
                            setattr(config.model_config, attr, getattr(config_model_config, attr))
                logger.debug("Config module imported via importlib")
            else:
                raise ImportError(f"Could not create spec for {config_path}")
        else:
            raise ImportError(f"Config file not found: {config_path}")
    except Exception as e2:
        logger.warning(f"Could not import config module: {e}")
        logger.warning(f"Alternative import also failed: {e2}")
        logger.warning(f"Project root: {_project_root}")
        logger.warning(f"Config path: {_project_root / 'config'}")
        logger.warning(f"Config exists: {(_project_root / 'config').exists()}")
        # Continue anyway, some modules have fallbacks

import gradio as gr

# Import state and configuration
from .state.app_state import state

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Log system info
logger.log_system_info()

# Import all tab modules
# These imports may trigger config.model_config imports, so path must be set correctly
from .tabs.home_tab import create_home_tab
from .tabs.tokenizer_tab import create_tokenizer_tab
from .tabs.training_tab import create_training_tab
from .tabs.lora_tab import create_lora_tab
from .tabs.finetune_tab import create_finetune_tab
from .tabs.rlhf_tab import create_rlhf_tab
from .tabs.download_model_tab import create_download_model_tab
from .tabs.chat_tab import create_chat_tab
from .tabs.statistics_tab import create_statistics_tab

# Import external UI components
try:
    from .research_pipeline_ui import ResearchPipelineUI
    RESEARCH_PIPELINE_AVAILABLE = True
except ImportError:
    RESEARCH_PIPELINE_AVAILABLE = False
    ResearchPipelineUI = None

try:
    from model_registry.registry_browser import create_registry_browser_tab
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    create_registry_browser_tab = None


def create_app():
    """
    Create and configure the Gradio application.
    
    This function sets up all tabs and returns the configured Gradio Blocks app.
    """
    research_pipeline_ui = None
    if RESEARCH_PIPELINE_AVAILABLE:
        research_pipeline_ui = ResearchPipelineUI()

    with gr.Blocks(title="QKV Core - Web Interface") as app:
        
        gr.Markdown()
        
        with gr.Tabs():
            # Core tabs
            create_home_tab()
            create_download_model_tab()
            create_tokenizer_tab()
            create_training_tab()
            create_finetune_tab()
            create_lora_tab()
            create_rlhf_tab()
            create_chat_tab()
            create_statistics_tab()
            
            # Optional tabs
            if research_pipeline_ui:
                with gr.Tab("🔬 Research Pipeline"):
                    research_pipeline_ui.create_research_tab()
            
            if create_registry_browser_tab:
                with gr.Tab("📊 Model Registry"):
                    create_registry_browser_tab()
        
        gr.Markdown()
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, server_name="0.0.0.0")
