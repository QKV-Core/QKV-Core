"""
Tests for Quantization Module

Unit tests for adaptive compression and quantization functionality.
"""

import unittest
import numpy as np
from qkv_core.quantization.adaptive import AdaptiveCompressor, CompressionMethod


class TestAdaptiveCompression(unittest.TestCase):
    """Test cases for adaptive compression."""
    
    def test_compressor_initialization(self):
        """Test that compressor initializes correctly."""
        compressor = AdaptiveCompressor()
        self.assertIsNotNone(compressor)
        self.assertEqual(compressor.method, CompressionMethod.ADAPTIVE)
        self.assertEqual(compressor.codebook_size, 256)
    
    def test_analyze_and_compress(self):
        """Test compression analysis."""
        compressor = AdaptiveCompressor()
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        
        method, codebook, compressed, original_size = compressor.analyze_and_compress(data)
        
        self.assertIsInstance(method, CompressionMethod)
        self.assertEqual(original_size, len(data))


if __name__ == "__main__":
    unittest.main()

