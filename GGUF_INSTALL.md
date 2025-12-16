# GGUF Model Support - Installation Guide

## Windows Installation

`llama-cpp-python` package requires a C++ compiler on Windows. You have several options:

### Option 1: Pre-built Wheel from GitHub Releases (Recommended - Easiest)

**Direct download from GitHub releases (Works reliably):**

For Python 3.10 (CPU):
```bash
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp310-cp310-win_amd64.whl
```

For Python 3.11 (CPU):
```bash
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp311-cp311-win_amd64.whl
```

For Python 3.12 (CPU):
```bash
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp312-cp312-win_amd64.whl
```

**Note:** Check the [GitHub releases page](https://github.com/abetlen/llama-cpp-python/releases) for the latest version and your specific Python version.

### Option 2: Alternative Pre-built Wheel Sources

**Using extra-index-url (May not always work):**

CPU:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

CUDA support (CUDA 12.1):
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

CUDA 11.8:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

### Option 3: Build from Source (Requires Visual Studio Build Tools)

1. Download and install Visual Studio Build Tools:
   - https://visualstudio.microsoft.com/downloads/
   - Download "Build Tools for Visual Studio"
   - During installation, check "Desktop development with C++"

2. After installation:
```bash
pip install llama-cpp-python
```

## Linux/Mac Installation

On Linux and Mac, you can usually install directly:

```bash
pip install llama-cpp-python
```

If you encounter errors, install required system libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
pip install llama-cpp-python
```

**Mac:**
```bash
brew install cmake
pip install llama-cpp-python
```

## Verification

To verify the installation was successful:

```python
try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python successfully installed!")
except ImportError:
    print("❌ llama-cpp-python is not installed!")
```

## Notes

- `llama-cpp-python` is optional for GGUF models - if you don't install it, only GGUF models won't work, other models will continue to work normally.
- Using pre-built wheels on Windows is the easiest method.
- For CUDA support, you need an NVIDIA GPU and CUDA toolkit.
- Check the [llama-cpp-python GitHub releases](https://github.com/abetlen/llama-cpp-python/releases) for the latest pre-built wheels.
