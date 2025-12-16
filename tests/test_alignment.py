"""
Tests for Data Alignment Utilities

Unit tests for padding, trimming, and alignment functions.
Tests the surgical alignment process that ensures GGUF block size compliance.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from typing import Tuple

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a simple test runner if pytest is not available
    class pytest:
        @staticmethod
        def fixture(func):
            return func


def align_data(raw_data: bytes, block_size: int = 110) -> bytes:
    """
    Align data to block size boundary by trimming excess bytes.
    
    This function implements the surgical alignment process described in the architecture:
    - Calculates the largest multiple of block_size that is <= current size
    - Trims excess bytes to make data divisible by block_size
    - Ensures llama.cpp compatibility
    
    Args:
        raw_data: Input byte data that may not be aligned
        block_size: Target block size for alignment (default: 110 for GGUF)
    
    Returns:
        Aligned byte data (divisible by block_size)
    
    Time Complexity: O(1) - only calculates and trims
    Space Complexity: O(N) - creates new byte array
    
    Example:
        align_data(152064 bytes) -> 152020 bytes (1382 * 110)
    """
    current_size = len(raw_data)
    
    # Calculate largest multiple of block_size <= current_size
    aligned_size = (current_size // block_size) * block_size
    
    if aligned_size == current_size:
        # Already aligned
        return raw_data
    
    # Trim to aligned size
    return raw_data[:aligned_size]


class TestAlignment:
    """Test cases for data alignment utilities."""
    
    def test_alignment_152064_to_152020(self):
        """
        Test that 152064 bytes are correctly trimmed to 152020 (divisible by 110).
        
        This is the specific case mentioned in the architecture documentation.
        """
        # Create byte array of size 152064
        data = np.random.randint(0, 255, size=152064, dtype=np.uint8)
        raw_bytes = data.tobytes()
        
        assert len(raw_bytes) == 152064, "Initial size should be 152064"
        
        # Align to block size 110
        aligned = align_data(raw_bytes, block_size=110)
        
        # Verify alignment
        # 152064 / 110 = 1382.4..., so floor is 1382
        # 1382 * 110 = 152020
        expected_size = (152064 // 110) * 110
        assert expected_size == 152020, f"Expected size should be 152020, got {expected_size}"
        
        assert len(aligned) == 152020, f"Aligned size should be 152020, got {len(aligned)}"
        assert len(aligned) % 110 == 0, "Aligned data should be divisible by 110"
        
        # Verify we trimmed 44 bytes (152064 - 152020 = 44)
        assert len(raw_bytes) - len(aligned) == 44, \
            f"Should trim 44 bytes, trimmed {len(raw_bytes) - len(aligned)}"
    
    def test_alignment_already_aligned(self):
        """Test that already aligned data is not modified."""
        # Create data that is already divisible by 110
        size = 110 * 100  # 11000 bytes
        data = np.random.randint(0, 255, size=size, dtype=np.uint8)
        raw_bytes = data.tobytes()
        
        aligned = align_data(raw_bytes, block_size=110)
        
        assert len(aligned) == len(raw_bytes), "Aligned data should be same size if already aligned"
        assert len(aligned) % 110 == 0, "Should still be divisible by 110"
        assert aligned == raw_bytes, "Data should be unchanged if already aligned"
    
    def test_alignment_various_sizes(self):
        """Test alignment with various data sizes."""
        test_cases = [
            (100, 110),      # 100 -> 0 (trim to 0, but that's edge case)
            (200, 110),      # 200 -> 110 (trim 90)
            (500, 110),      # 500 -> 440 (trim 60)
            (1000, 110),     # 1000 -> 990 (trim 10)
            (152064, 110),   # 152064 -> 152020 (trim 44)
        ]
        
        for original_size, block_size in test_cases:
            data = np.random.randint(0, 255, size=original_size, dtype=np.uint8)
            raw_bytes = data.tobytes()
            
            aligned = align_data(raw_bytes, block_size=block_size)
            
            # Verify alignment
            assert len(aligned) % block_size == 0, \
                f"Size {len(aligned)} should be divisible by {block_size}"
            
            # Verify we didn't grow the data
            assert len(aligned) <= len(raw_bytes), \
                f"Aligned size {len(aligned)} should be <= original {len(raw_bytes)}"
            
            # Verify we trimmed to the largest multiple <= original
            expected_size = (original_size // block_size) * block_size
            assert len(aligned) == expected_size, \
                f"Expected {expected_size}, got {len(aligned)}"
    
    def test_alignment_gguf_block_size(self):
        """Test that aligned data returns valid GGUF block sizes."""
        # Test with the specific case from architecture docs
        data = np.random.randint(0, 255, size=152064, dtype=np.uint8)
        raw_bytes = data.tobytes()
        
        aligned = align_data(raw_bytes, block_size=110)
        
        # Verify it's a valid GGUF block size
        assert len(aligned) % 110 == 0, "Must be divisible by GGUF block size (110)"
        
        # Verify it can be divided into complete blocks
        num_blocks = len(aligned) // 110
        assert num_blocks * 110 == len(aligned), "Should form complete blocks"
        assert num_blocks > 0, "Should have at least one block"
        
        # For 152020 bytes: 152020 / 110 = 1382 blocks
        if len(raw_bytes) == 152064:
            assert len(aligned) == 152020, "Should align to 152020 for 152064 input"
            assert len(aligned) // 110 == 1382, "Should form 1382 blocks"
    
    def test_alignment_preserves_data_integrity(self):
        """Test that alignment preserves the beginning of the data."""
        # Create data with known pattern
        size = 200
        data = np.arange(size, dtype=np.uint8)
        raw_bytes = data.tobytes()
        
        aligned = align_data(raw_bytes, block_size=110)
        
        # First part of data should be preserved
        preserved_size = len(aligned)
        assert raw_bytes[:preserved_size] == aligned, \
            "Beginning of data should be preserved"
        
        # Verify the preserved data is correct
        preserved_data = np.frombuffer(aligned, dtype=np.uint8)
        expected_data = np.arange(preserved_size, dtype=np.uint8)
        np.testing.assert_array_equal(preserved_data, expected_data)


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        # Simple test runner without pytest
        import unittest
        unittest.main()
