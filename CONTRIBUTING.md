# Contributing to QKV Core

Thank you for your interest in contributing to QKV Core (Query-Key-Value Core)! This document provides comprehensive guidelines for contributing to the project. We welcome contributions of all kinds, from bug reports and feature requests to code contributions and documentation improvements.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Report Bugs](#how-to-report-bugs)
3. [Setting up Development Environment](#setting-up-development-environment)
4. [Submitting Pull Requests](#submitting-pull-requests)
5. [Getting Started](#getting-started)
6. [Project Overview](#project-overview)
7. [Ways to Contribute](#ways-to-contribute)
8. [Project Structure](#project-structure)
9. [Coding Standards](#coding-standards)
10. [Testing Guidelines](#testing-guidelines)
11. [Priority Contribution Areas](#priority-contribution-areas)
12. [Communication](#communication)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. By participating, you agree to maintain a respectful and inclusive environment for everyone.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

---

## How to Report Bugs

Found a bug? Help us fix it! Follow these guidelines to report issues effectively.

**Before reporting:**
- Check existing issues to see if it's already reported
- Test with the latest version from the main branch
- Gather relevant information (error messages, system info, steps to reproduce)

**Creating a bug report:**

1. Go to the Issues tab on GitHub
2. Click "New Issue" → "Bug Report"
3. Fill out the template with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Python version, PyTorch version, GPU model)
   - Error messages and stack traces
   - Screenshots (if applicable)

**Example Bug Report:**
```markdown
**Bug Description:**
Training crashes with OOM error when using batch size 128 on RTX 3060.

**Steps to Reproduce:**
1. Run: `python cli/run.py train --data data/sample_corpus.txt --tokenizer tokenizer/distilgpt2.pkl --batch-size 128`
2. Training starts successfully
3. After ~100 steps, CUDA out of memory error occurs

**Expected Behavior:**
Training should complete or gracefully handle memory constraints.

**Actual Behavior:**
RuntimeError: CUDA out of memory. Tried to allocate 2.5GB...

**System Information:**
- OS: Windows 11
- Python: 3.10.11
- PyTorch: 2.7.1+cu118
- GPU: NVIDIA RTX 3060 (6GB VRAM)
- CUDA: 11.8

**Error Log:**
[Full stack trace here]
```

---

## Setting up Development Environment

Follow these steps to set up your development environment for contributing to QKV Core.

### Prerequisites

Before contributing, ensure you have:
- **Python 3.10+** (3.10, 3.11, or 3.12 recommended)
- **Git** for version control
- **CUDA Toolkit** (optional, for GPU acceleration)
- **Visual Studio Build Tools** (Windows only, for compiling some dependencies)

### Step-by-Step Setup

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/LLM-Core-Project.git
cd LLM-Core-Project
```

#### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install GGUF support (optional, for GGUF model support)
# See GGUF_INSTALL.md for platform-specific instructions
# Windows (Python 3.10):
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-win_amd64.whl

# Linux/Mac:
pip install llama-cpp-python
```

#### 4. Verify Installation

```bash
# Test core imports
python -c "import torch; import gradio; print('✅ Core dependencies installed')"

# Test project structure
python -c "from qkv_core.core.transformer import TransformerModel; print('✅ Project structure OK')"
```

#### 5. Set Up Pre-commit Hooks (Optional)

```bash
# Install pre-commit (if available)
pip install pre-commit
pre-commit install
```

#### 6. Create Required Directories

The project will create these automatically, but you can create them manually:

```bash
mkdir -p tokenizer model_weights storage logs data
```

---

## Submitting Pull Requests

Follow this process to submit your contributions via Pull Requests.

### 1. Create a Branch

```bash
# Feature branch
git checkout -b feature/add-model-pruning

# Bug fix branch
git checkout -b bugfix/fix-memory-leak

# Documentation branch
git checkout -b docs/improve-readme
```

**Branch Naming Convention:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test improvements
- `chore/` - Maintenance tasks

### 2. Make Your Changes

- Write clean, well-documented code
- Follow coding standards (see [Coding Standards](#coding-standards))
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

**Commit Message Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Good Commit Messages:**
```bash
feat: Add GGUF model support for optimized inference

- Implement GGUFModelLoader for loading GGUF format models
- Add GGUFTokenizerWrapper for built-in tokenizer support
- Update InferenceEngine with GGUF stream generation
- Add GGUF_INSTALL.md with installation instructions

Closes #42
```

### 4. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Then create PR on GitHub
```

**Pull Request Template:**
```markdown
## Description
Brief description of changes and motivation

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Changes Made
- Added X feature
- Fixed Y bug
- Updated Z documentation
- Refactored A module

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All existing tests pass
- [ ] Test coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines (PEP 8)
- [ ] Self-review performed
- [ ] Comments added for complex logic
- [ ] Documentation updated (docstrings, README, etc.)
- [ ] No new warnings or errors
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Logging added where appropriate

## Related Issues
Closes #123
Related to #456
```

### 5. Code Review

- Address all review comments
- Make requested changes
- Respond to feedback professionally
- Update PR if needed
- Once approved, maintainers will merge

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10+** (3.10, 3.11, or 3.12 recommended)
- **Git** for version control
- **CUDA Toolkit** (optional, for GPU acceleration)
- **Visual Studio Build Tools** (Windows only, for compiling some dependencies)

### Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/QKV-Core.git
   cd QKV-Core
   ```
3. **Set up development environment** (see [Development Environment Setup](#development-environment-setup))
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and test them
6. **Submit a Pull Request**

---

## Project Overview

QKV Core (Query-Key-Value Core) is a comprehensive framework for training, fine-tuning, and deploying Large Language Models (LLMs). Built on the fundamental Query-Key-Value attention mechanism that powers transformer architectures, QKV Core provides both a command-line interface and a web-based UI built with Gradio.

### Key Features

- **Transformer Architecture**: Full implementation of GPT-style transformer models
- **Training & Fine-tuning**: Support for full training, incremental training, and fine-tuning
- **Parameter-Efficient Methods**: LoRA and QLoRA for efficient fine-tuning
- **RLHF & DPO**: Reinforcement Learning from Human Feedback and Direct Preference Optimization
- **Model Formats**: Support for PyTorch (.pt) and GGUF formats
- **Hugging Face Integration**: Download and convert models from Hugging Face Hub
- **Web UI**: Comprehensive Gradio-based interface for all operations
- **CLI Interface**: Command-line tools for training and inference
- **Research Features**: Implementation of cutting-edge techniques (FlashAttention, Mamba SSM, etc.)

### Technology Stack

- **Core**: PyTorch, NumPy
- **Web UI**: Gradio 6.1+
- **Tokenization**: Custom BPE, Hugging Face tokenizers, SentencePiece
- **Storage**: SQLite (default), PostgreSQL (optional)
- **Model Formats**: PyTorch, GGUF (via llama-cpp-python)

---

## Ways to Contribute

### 1. 🐛 Bug Reports

*For detailed bug reporting guidelines, see the [How to Report Bugs](#how-to-report-bugs) section above.*

### 2. ✨ Feature Requests

Have an idea for a new feature? We'd love to hear it!

**Before requesting:**
- Check if a similar feature already exists
- Search discussions to see if it's been discussed
- Consider the use case and potential implementation

**Creating a feature request:**

1. Go to GitHub Discussions
2. Create a new discussion with the "Feature Request" category
3. Include:
   - Clear description of the feature
   - Motivation and use cases
   - Proposed implementation (if you have ideas)
   - Alternatives considered
   - Potential impact

**Example Feature Request:**
```markdown
**Feature:** Add model pruning support

**Motivation:**
Enable deployment of large models on resource-constrained devices by reducing model size without significant performance loss.

**Use Cases:**
- Deploy 1.5B models on edge devices with limited memory
- Reduce inference latency for production deployments
- Enable running multiple models simultaneously

**Proposed Solution:**
Implement magnitude-based pruning:
- Add pruning utilities in `utils/pruning.py`
- Integrate with training pipeline
- Support structured and unstructured pruning
- Add pruning ratio configuration

**Alternatives Considered:**
- Knowledge distillation (more complex, requires teacher model)
- Quantization (already supported, can be combined)
- Model compression (different approach)

**Impact:**
- Low: Minimal changes to existing code
- Medium: New utility module
- High: Integration with training pipeline
```

### 3. 📚 Documentation

Documentation improvements are highly valued! This includes:

- Fixing typos and grammatical errors
- Improving clarity and explanations
- Adding code examples and tutorials
- Translating documentation to other languages
- Creating video tutorials or guides
- Improving inline code documentation

**Documentation Files:**
- `CONTRIBUTING.md` (this file)
- `README.md` (if exists)
- `docs/RESEARCH_IMPLEMENTATIONS.md`
- `GGUF_INSTALL.md`
- Inline code docstrings
- Web UI help text

### 4. 💻 Code Contributions

Code contributions are the heart of open source! Follow these steps:

1. **Find or create an issue** describing what you want to work on
2. **Fork the repository** and create a feature branch
3. **Write your code** following our [coding standards](#coding-standards)
4. **Write tests** for your changes
5. **Test thoroughly** before submitting
6. **Submit a Pull Request** with a clear description

---

*Note: For detailed development environment setup instructions, see the [Setting up Development Environment](#setting-up-development-environment) section above.*

---

## Project Structure

Understanding the project structure helps you navigate and contribute effectively:

```
LLM-Core-Project/
├── cli/                    # Command-line interface
│   ├── run.py             # Main CLI entry point
│   └── research_cli.py    # Research pipeline CLI
│
├── config/                 # Configuration management
│   ├── model_config.py    # Model architecture configurations
│   └── database_config.py # Database connection settings
│
├── core/                   # Core model implementations
│   └── transformer.py     # Transformer architecture (GPT-style)
│
├── data/                   # Sample data and test corpora
│   ├── sample_corpus.txt
│   └── quick_test.txt
│
├── docs/                   # Documentation
│   └── RESEARCH_IMPLEMENTATIONS.md  # Research paper implementations
│
├── model_registry/         # Model registry system
│   └── registry_browser.py
│
├── model_weights/          # Trained model checkpoints (.pt, .gguf)
│
├── models/                 # Inference engines
│   ├── inference.py       # Main inference engine (supports GGUF)
│   ├── fast_inference.py  # Optimized inference
│   ├── batch_inference.py # Batch processing
│   └── simple_inference.py
│
├── scripts/                # Utility scripts
│   ├── setup_postgresql.py
│   ├── create_test_corpus.py
│   └── ...
│
├── storage/                 # Database implementations
│   ├── db.py              # SQLite database manager
│   └── postgresql_db.py  # PostgreSQL support
│
├── tokenizer/              # Tokenizer implementations
│   ├── bpe.py             # BPE tokenizer
│   └── *.pkl              # Saved tokenizers
│
├── training/               # Training implementations
│   ├── trainer.py         # Main trainer
│   ├── incremental_trainer.py
│   ├── rlhf.py            # RLHF implementation
│   ├── scaling_optimizer.py
│   └── dataset.py
│
├── utils/                  # Utility modules
│   ├── model_loader.py    # Model loading utilities
│   ├── smart_loader.py    # Intelligent model loader (supports GGUF)
│   ├── gguf_loader.py     # GGUF model support
│   ├── huggingface_converter.py  # HF model conversion
│   ├── logger.py          # Logging utilities
│   └── ...
│
├── web_ui/                 # Gradio web interface
│   ├── app.py             # Main application
│   ├── tabs/              # UI tab components
│   │   ├── home_tab.py
│   │   ├── chat_tab.py
│   │   ├── training_tab.py
│   │   ├── download_model_tab.py
│   │   └── ...
│   ├── config/            # UI configuration
│   └── state/             # Application state management
│
├── requirements.txt       # Python dependencies
├── launch_web_ui.py       # Web UI launcher
├── debug_chat.py          # Debug chat interface
└── CONTRIBUTING.md        # This file
```

### Key Modules

- **`core/transformer.py`**: Core transformer model implementation
- **`models/inference.py`**: Main inference engine with GGUF support
- **`training/trainer.py`**: Training loop and optimization
- **`utils/smart_loader.py`**: Universal model loader (PyTorch + GGUF)
- **`web_ui/app.py`**: Gradio application entry point

---

## Coding Standards

### 1. Code Style

**Python Style Guide:**
- **Follow PEP 8** - All code must adhere to PEP 8 style guidelines
- Maximum line length: **100 characters** (soft limit)
- Use **4 spaces** for indentation (no tabs)
- Use **snake_case** for functions and variables
- Use **PascalCase** for classes
- Use **UPPER_CASE** for constants

**PEP 8 Compliance:**
- Use a code formatter like `black` or `autopep8` to ensure PEP 8 compliance
- Run `flake8` or `pylint` to check for style violations
- All code must pass PEP 8 checks before merging

**Example:**
```python
# Good
def train_model(
    model: TransformerModel,
    dataset: TextDataset,
    epochs: int = 10
) -> Dict[str, float]:
    """Train the model on the given dataset."""
    pass

# Bad
def TrainModel(m, d, e=10):
    pass
```

### 2. Type Hints

Always use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple, Union

def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[TransformerModel, Dict[str, Any]]:
    """Load model from checkpoint."""
    pass
```

### 3. Docstrings

Use **Google-style docstrings**:

```python
def generate_text(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7
) -> str:
    """
    Generate text from a given prompt using the loaded model.
    
    Args:
        prompt: Input text prompt to generate from
        max_length: Maximum number of tokens to generate (default: 100)
        temperature: Sampling temperature. Higher values increase randomness.
            Range: 0.0 (deterministic) to 2.0 (very random). Default: 0.7
    
    Returns:
        Generated text string
    
    Raises:
        ValueError: If prompt is empty or max_length is invalid
        RuntimeError: If model is not loaded
    
    Example:
        >>> model = load_model("model_weights/best_model.pt")
        >>> text = generate_text("Hello, how are", max_length=50)
        >>> print(text)
        "Hello, how are you today? I'm doing well..."
    """
    pass
```

### 4. Comments

Write clear, meaningful comments:

```python
# Good: Explains WHY, not WHAT
# Apply temperature scaling to increase sampling diversity
# Lower temperature = more focused, higher = more creative
logits = logits / temperature

# Bad: States the obvious
# Divide logits by temperature
logits = logits / temperature
```

### 5. Error Handling

Always handle errors gracefully:

```python
def load_model_safe(model_path: str) -> Optional[TransformerModel]:
    """Load model with proper error handling."""
    try:
        model, config = SmartLoader.load_model(model_path)
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return None
```

### 6. Logging

Use the project's logging system:

```python
from utils.logger import get_logger

logger = get_logger()

def train_model():
    logger.info("Starting training...")
    try:
        # Training code
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_transformer.py

# Run with coverage
pytest --cov=core --cov=models --cov-report=html

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_transformer.py::test_multi_head_attention
```

### Writing Tests

**Test Structure:**
```python
import pytest
import torch
from core.transformer import MultiHeadAttention

class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention module."""
    
    def test_output_shape(self):
    """Test that MHA returns correct output shape."""
    batch_size = 2
    seq_len = 10
    d_model = 512
        num_heads = 8
    
        attention = MultiHeadAttention(d_model, num_heads)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attention(query, key, value)
    
    assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1.0."""
        # Test implementation
        pass
    
    @pytest.mark.parametrize("d_model,num_heads", [
        (512, 8),
        (768, 12),
        (1024, 16)
    ])
    def test_different_configurations(self, d_model, num_heads):
        """Test attention with different model configurations."""
        # Test implementation
        pass
```

### Test Coverage Goals

- **Core modules**: >90% coverage
- **Utility modules**: >80% coverage
- **Web UI**: >70% coverage (UI testing is harder)
- **Overall project**: >80% coverage

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test module interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Benchmark critical paths

---

*For detailed pull request guidelines, see the [Submitting Pull Requests](#submitting-pull-requests) section above.*

---

## Priority Contribution Areas

### High Priority 🔴

1. **Test Coverage**
   - Increase unit test coverage to >80%
   - Add integration tests for training pipeline
   - Add end-to-end tests for web UI

2. **GGUF Model Support**
   - Improve GGUF stream generation reliability
   - Add support for more GGUF quantization formats
   - Optimize GGUF model loading performance

3. **Documentation**
   - Complete API documentation
   - Add more code examples
   - Create video tutorials
   - Improve inline documentation

4. **Performance Optimization**
   - Optimize training loop
   - Improve inference speed
   - Reduce memory footprint
   - Add batch processing optimizations

### Medium Priority 🟡

1. **Multi-GPU Training**
   - Distributed training support
   - DataParallel and DistributedDataParallel
   - Multi-node training

2. **Model Quantization**
   - INT8 quantization
   - INT4 quantization
   - Dynamic quantization

3. **Additional Tokenizers**
   - WordPiece tokenizer
   - SentencePiece improvements
   - Custom tokenizer support

4. **Web UI Enhancements**
   - Better error messages
   - Progress indicators
   - Model comparison tools
   - Training visualization

### Low Priority 🟢

1. **ONNX Export**
   - Model export to ONNX format
   - ONNX Runtime integration

2. **Mobile Deployment**
   - Model optimization for mobile
   - Mobile app integration

3. **Advanced Features**
   - Beam search improvements
   - Nucleus sampling enhancements
   - Custom generation strategies

---

## Communication

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussions
- **Pull Requests**: For code-related questions during development

### Reporting Security Issues

**DO NOT** open a public issue for security vulnerabilities. Instead, please email the maintainers directly or use GitHub's private security advisory feature.

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

---

## Recognition

Contributors will be:
- Listed in the project's contributors section
- Credited in release notes
- Acknowledged in relevant documentation

Every contribution, no matter how small, is valuable and appreciated!

---

## Additional Resources

- **Research Implementations**: See `docs/RESEARCH_IMPLEMENTATIONS.md`
- **GGUF Installation**: See `GGUF_INSTALL.md`
- **Project License**: See `LICENSE`

---

Thank you for contributing to QKV Core! 🚀

**Happy Coding!**
