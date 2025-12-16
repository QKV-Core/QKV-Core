# QKV Core - Project Structure

## Overview

This project follows a standard Python package structure where:
- **`qkv_core/`** - The main Python package (installable package)
- **Root directory** - Project root with application files, scripts, and configuration

## Directory Structure

```
QKV-Core-Project/
├── qkv_core/              # Main Python package (installable)
│   ├── __init__.py
│   ├── core/              # Transformer architecture
│   ├── formats/           # Model format handlers (GGUF, PyTorch)
│   ├── inference/         # Inference engines
│   ├── training/          # Training implementations
│   ├── tokenization/      # Tokenizer implementations
│   ├── storage/           # Database implementations
│   └── utils/             # Utility functions
│
├── web_ui/                # Gradio web interface
├── cli/                   # Command-line interface
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── utils/                 # Additional utilities (not part of package)
├── data/                  # Sample data
├── docs/                  # Documentation
├── model_weights/         # Model checkpoints
├── logs/                  # Log files
│
├── debug_chat.py          # Debug chat interface
├── launch_web_ui.py       # Web UI launcher
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Package vs Project Files

### `qkv_core/` Package
- Contains reusable, installable Python modules
- Can be imported: `from qkv_core import TransformerModel`
- Follows Python package best practices
- Modules: core, formats, inference, training, tokenization, storage, utils

### Root Directory Files
- Application entry points (`debug_chat.py`, `launch_web_ui.py`)
- Web interface (`web_ui/`)
- CLI tools (`cli/`)
- Configuration (`config/`)
- Scripts and utilities (`scripts/`, `utils/`)
- Data and resources (`data/`, `model_weights/`, `logs/`)

## Usage

### As a Package
```python
from qkv_core import TransformerModel, SmartLoader, InferenceEngine
from qkv_core.formats import GGUFModelLoader
from qkv_core.training import Trainer
```

### As a Project
```bash
# Run debug chat
python debug_chat.py

# Launch web UI
python launch_web_ui.py

# Use CLI
python cli/run.py train --data data/sample_corpus.txt
```

## Why This Structure?

1. **Separation of Concerns**: Package code is separate from application code
2. **Reusability**: `qkv_core` can be installed and used in other projects
3. **Maintainability**: Clear organization makes the codebase easier to navigate
4. **Standard Practice**: Follows Python packaging conventions

