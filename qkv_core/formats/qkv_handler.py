"""
QKV Format Handler - Custom Format Reader/Writer

This module handles reading and writing QKV Core's custom .qkv format files.
The QKV format uses adaptive compression to achieve optimal storage efficiency.

Why QKV format?
- Native format optimized for QKV Core's compression algorithms
- Supports adaptive compression (raw vs dictionary per tensor)
- Fast decompression with Numba-optimized kernels
- Smaller file sizes than raw formats while maintaining speed

Note: This format is currently deprecated in favor of GGUF for better
compatibility with llama.cpp and the broader ecosystem.
"""

import struct
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
import numpy as np

from qkv_core.quantization.adaptive import CompressionMethod, AdaptiveCompressor
from qkv_core.kernels.numba_engine import decompress_chunk
from qkv_core.utils.logger import get_logger

logger = get_logger()


class QKVReader:
    """
    Reads QKV format files, decompressing tensors as needed.
    
    The QKV format structure:
    - Magic number: 'QKV3' (4 bytes)
    - For each tensor:
      - Name length (4 bytes, uint32)
      - Name (variable length, UTF-8)
      - Compression method (1 byte)
      - Original size (8 bytes, uint64)
      - Data length (8 bytes, uint64)
      - Data (variable length)
      - If ADAPTIVE: Codebook length (4 bytes) + Codebook data
    
    Time Complexity: O(N) for reading N tensors
    Space Complexity: O(M) where M is the size of the largest tensor
    """
    MAGIC_NUMBER = b'QKV3'  # QKV v3 uses adaptive compression with bitmap flagging
    
    def __init__(self, file_path: str):
        """
        Initialize QKV reader.
        
        Args:
            file_path: Path to the input QKV file
        
        Time Complexity: O(1)
        """
        self.file_path = Path(file_path)
        self.file_handle: Optional[BinaryIO] = None
    
    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            Self for use in 'with' statements
        
        Time Complexity: O(1)
        """
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Ensures file handle is closed even if an exception occurs.
        
        Time Complexity: O(1)
        """
        self.close()
    
    def open(self):
        """
        Open QKV file for reading and validate magic number.
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not a valid QKV v3 format
        
        Time Complexity: O(1)
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"QKV file not found: {self.file_path}")
        
        self.file_handle = open(self.file_path, "rb")
        magic = self.file_handle.read(len(self.MAGIC_NUMBER))
        if magic != self.MAGIC_NUMBER:
            raise ValueError(f"Invalid QKV file format. Expected: {self.MAGIC_NUMBER.decode()}, Got: {magic.decode()}")
    
    def close(self):
        """
        Close the QKV file handle.
        
        Time Complexity: O(1)
        """
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def read_tensor(self) -> Optional[Dict[str, Any]]:
        """
        Read the next tensor from the QKV file, decompressing if necessary.
        
        The method reads tensor metadata and data, then decompresses using
        Numba-optimized kernels if the tensor was compressed with ADAPTIVE method.
        
        Returns:
            Dictionary containing:
            - 'name': Tensor name (str)
            - 'method': CompressionMethod used
            - 'data': Decompressed tensor data as numpy array (np.ndarray)
            - 'original_size': Original size in uint16 elements (int)
            Returns None if end of file is reached.
        
        Time Complexity: O(M) where M is the size of the tensor in elements
                        (decompression is O(M) with Numba JIT)
        Space Complexity: O(M) for the decompressed tensor data
        
        Raises:
            RuntimeError: If file is not open
            struct.error: If file format is corrupted
        """
        if not self.file_handle:
            raise RuntimeError("File not open. Call .open() first or use a context manager.")
        
        try:
            # Read tensor name length and name
            name_len_bytes = self.file_handle.read(4)
            if not name_len_bytes:
                return None  # End of file
            
            name_len = struct.unpack('I', name_len_bytes)[0]
            name = self.file_handle.read(name_len).decode('utf-8')
            
            # Read compression method
            method_byte = self.file_handle.read(1)
            method = CompressionMethod(method_byte[0])
            
            # Read original size
            original_size = struct.unpack('Q', self.file_handle.read(8))[0]
            
            # Read data length and data
            data_len = struct.unpack('Q', self.file_handle.read(8))[0]
            data_bytes = self.file_handle.read(data_len)
            
            # Decompress if needed
            if method == CompressionMethod.ADAPTIVE:
                # Read codebook
                cb_len = struct.unpack('I', self.file_handle.read(4))[0]
                codebook = None
                if cb_len > 0:
                    cb_bytes = self.file_handle.read(cb_len * 2)
                    codebook = np.frombuffer(cb_bytes, dtype=np.uint16)
                
                # Decompress using Numba-optimized kernel
                compressed_buffer = np.frombuffer(data_bytes, dtype=np.uint8)
                data = decompress_chunk(compressed_buffer, codebook, original_size)
            else:
                # Raw data
                data = np.frombuffer(data_bytes, dtype=np.uint16)
            
            return {
                'name': name,
                'method': method,
                'data': data,
                'original_size': original_size
            }
        except struct.error:
            return None  # End of file or corrupted


class QKVWriter:
    """
    Writes tensors to QKV format files, applying adaptive compression.
    
    The writer uses AdaptiveCompressor to decide whether to compress each tensor
    or store it as raw data, based on compression effectiveness.
    
    Time Complexity: O(N*M) for writing N tensors of average size M
                     (compression analysis is O(M) per tensor)
    Space Complexity: O(M) for compression buffers
    """
    MAGIC_NUMBER = b'QKV3'
    
    def __init__(self, file_path: str, compressor: Optional[AdaptiveCompressor] = None):
        """
        Initialize QKV writer.
        
        Args:
            file_path: Output file path
            compressor: AdaptiveCompressor instance (creates default if None)
                       Why optional? Allows custom compression strategies per file
        
        Time Complexity: O(1)
        """
        self.file_path = Path(file_path)
        self.file_handle: Optional[BinaryIO] = None
        
        if compressor is None:
            self.compressor = AdaptiveCompressor()
        else:
            self.compressor = compressor
    
    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            Self for use in 'with' statements
        
        Time Complexity: O(1)
        """
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Ensures file handle is closed even if an exception occurs.
        
        Time Complexity: O(1)
        """
        self.close()
    
    def open(self):
        """
        Open QKV file for writing. Writes magic number.
        
        Time Complexity: O(1)
        """
        self.file_handle = open(self.file_path, "wb")
        self.file_handle.write(self.MAGIC_NUMBER)
    
    def close(self):
        """
        Close the QKV file handle.
        
        Time Complexity: O(1)
        """
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def write_tensor(self, name: str, data: np.ndarray):
        """
        Write a tensor to the QKV file, applying adaptive compression if beneficial.
        
        The method:
        1. Analyzes the tensor using AdaptiveCompressor
        2. Decides whether to compress or store raw
        3. Writes metadata and data to file
        
        Args:
            name: Tensor name (will be encoded as UTF-8)
            data: Tensor data as numpy array (expected uint16 for QKV v3)
        
        Time Complexity: O(M) where M is the number of elements in data
                        (compression analysis is O(M log M) for frequency analysis)
        Space Complexity: O(M) for compression buffers
        
        Raises:
            RuntimeError: If file is not open
        """
        if not self.file_handle:
            raise RuntimeError("File not open. Call .open() first or use a context manager.")
        
        # Apply adaptive compression
        method, codebook, compressed_data, original_size = self.compressor.analyze_and_compress(data)
        
        # Write tensor name
        name_bytes = name.encode('utf-8')
        self.file_handle.write(struct.pack('I', len(name_bytes)))
        self.file_handle.write(name_bytes)
        
        # Write compression method
        self.file_handle.write(struct.pack('B', method.value))
        
        # Write original size
        self.file_handle.write(struct.pack('Q', original_size))
        
        # Write data length and data
        self.file_handle.write(struct.pack('Q', len(compressed_data)))
        self.file_handle.write(compressed_data)
        
        # Write codebook if ADAPTIVE
        if method == CompressionMethod.ADAPTIVE and codebook is not None:
            self.file_handle.write(struct.pack('I', len(codebook)))
            self.file_handle.write(codebook.tobytes())
