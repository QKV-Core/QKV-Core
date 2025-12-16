# QKV Core - API Reference

## Package Overview

```python
from qkv_core import TransformerModel, SmartLoader, InferenceEngine
from qkv_core.quantization import AdaptiveCompressor
from qkv_core.formats import QKVReader, QKVWriter
from qkv_core.kernels import compress_chunk, decompress_chunk
```

## Core Modules

### TransformerModel

```python
from qkv_core.core.transformer import TransformerModel

model = TransformerModel(config)
```

### SmartLoader

```python
from qkv_core.formats.smart_loader import SmartLoader

model, config = SmartLoader.load_model("model.pt")
```

### InferenceEngine

```python
from qkv_core.inference.inference import InferenceEngine

engine = InferenceEngine(model, tokenizer)
for token in engine.generate_stream(prompt):
    print(token, end="")
```

## Quantization

### AdaptiveCompressor

```python
from qkv_core.quantization.adaptive import AdaptiveCompressor

compressor = AdaptiveCompressor()
method, codebook, compressed, size = compressor.analyze_and_compress(data)
```

## Formats

### QKV Format

```python
from qkv_core.formats.qkv_handler import QKVReader, QKVWriter

# Reading
with QKVReader("model.qkv") as reader:
    tensor = reader.read_tensor()

# Writing
with QKVWriter("model.qkv") as writer:
    writer.write_tensor("layer.weight", data)
```

## Kernels

### Numba Functions

```python
from qkv_core.kernels.numba_engine import compress_chunk, decompress_chunk

compressed_len = compress_chunk(input_data, codebook, output_buffer)
decompressed = decompress_chunk(compressed_buffer, codebook, output_size)
```

