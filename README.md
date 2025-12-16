# QKV Core

**Query-Key-Value Core - The Core of Transformer Intelligence**

A comprehensive framework for training, fine-tuning, and deploying Large Language Models (LLMs) built on the fundamental Query-Key-Value attention mechanism that powers modern transformer architectures.

## 🚀 Features

- **Transformer Architecture**: Full implementation of GPT-style transformer models
- **Training & Fine-tuning**: Support for full training, incremental training, and fine-tuning
- **Parameter-Efficient Methods**: LoRA and QLoRA for efficient fine-tuning
- **RLHF & DPO**: Reinforcement Learning from Human Feedback and Direct Preference Optimization
- **Model Formats**: Support for PyTorch (.pt) and GGUF formats
- **Hugging Face Integration**: Download and convert models from Hugging Face Hub
- **Web UI**: Comprehensive Gradio-based interface for all operations
- **CLI Interface**: Command-line tools for training and inference
- **Research Features**: Implementation of cutting-edge techniques (FlashAttention, Mamba SSM, etc.)

## 📦 Installation

### Prerequisites

- Python 3.10+ (3.10, 3.11, or 3.12 recommended)
- PyTorch 2.0+
- CUDA Toolkit (optional, for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/QKV-Core/QKV-Core.git
cd QKV-Core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install GGUF support (optional, for GGUF models)
# See GGUF_INSTALL.md for platform-specific instructions
```

## 🎯 Quick Start

### Web Interface

```bash
python launch_web_ui.py
```

Then open your browser to `http://localhost:7861`

### Command Line Interface

```bash
# Train a tokenizer
python cli/run.py train-tokenizer --corpus data/sample_corpus.txt --output tokenizer/my_tokenizer.pkl

# Train a model
python cli/run.py train --data data/sample_corpus.txt --tokenizer tokenizer/my_tokenizer.pkl

# Chat with a model
python debug_chat.py
```

## 📚 Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Comprehensive contribution guidelines
- **[GGUF_INSTALL.md](GGUF_INSTALL.md)**: GGUF model installation guide
- **[docs/RESEARCH_IMPLEMENTATIONS.md](docs/RESEARCH_IMPLEMENTATIONS.md)**: Research paper implementations

## 🏗️ Project Structure

```
QKV-Core/
├── core/              # Core transformer implementation
├── models/            # Inference engines
├── training/          # Training implementations
├── web_ui/            # Gradio web interface
├── cli/               # Command-line interface
├── utils/             # Utility modules
└── docs/              # Documentation
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the fundamental Query-Key-Value attention mechanism that powers transformer architectures. QKV Core brings production-grade AI capabilities to your fingertips.

---

**QKV Core - Where Query, Key, and Value Create Intelligence** 🚀


<br>
<hr>
<div align="right">
  <sub>Built with ❤️ for the Open Source AI Community by <a href="https://github.com/broxytr">Hüseyin Kama</a></sub>
</div>
