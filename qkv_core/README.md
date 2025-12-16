# QKV Core Package

This is the main Python package for QKV Core - Query-Key-Value Core.

## Package Structure

```
qkv_core/
├── __init__.py           # Package initialization and exports
├── core/                 # Transformer architecture implementations
├── formats/             # Model format handlers (GGUF, PyTorch)
├── inference/            # Inference engines for text generation
├── training/             # Training implementations
├── tokenization/         # Tokenizer implementations
├── storage/              # Database implementations
└── utils/                # Utility functions
```

## Installation

This package is part of the QKV Core project. To use it:

```python
from qkv_core import TransformerModel, SmartLoader, InferenceEngine
from qkv_core.formats import GGUFModelLoader
from qkv_core.training import Trainer
```

## Package vs Project

- **This package (`qkv_core/`)**: Contains reusable, installable Python modules
- **Project root**: Contains application files (web_ui, cli, scripts, etc.)

This separation follows Python packaging best practices:
- Package code is reusable and can be installed separately
- Application code uses the package but is not part of it
- Clear separation of concerns

## Usage

```python
# Import core components
from qkv_core import TransformerModel
from qkv_core.formats.smart_loader import SmartLoader
from qkv_core.inference.inference import InferenceEngine

# Use the components
model, config = SmartLoader.load_model("model_weights/model.pt")
engine = InferenceEngine(model, tokenizer)
```

