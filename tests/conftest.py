"""
Pytest Configuration and Fixtures

This module provides shared fixtures for all test modules.
Fixtures are used to create reusable test data without needing actual model weights.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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

import numpy as np
from typing import Tuple


@pytest.fixture
def dummy_tensor_small() -> np.ndarray:
    """
    Create a small dummy tensor for testing.
    
    Returns:
        Small numpy array of uint16 values
    """
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)


@pytest.fixture
def dummy_tensor_medium() -> np.ndarray:
    """
    Create a medium-sized dummy tensor for testing.
    
    Returns:
        Medium numpy array of uint16 values (1000 elements)
    """
    return np.random.randint(0, 65535, size=1000, dtype=np.uint16)


@pytest.fixture
def dummy_tensor_large() -> np.ndarray:
    """
    Create a large dummy tensor for testing.
    
    Returns:
        Large numpy array of uint16 values (10000 elements)
    """
    return np.random.randint(0, 65535, size=10000, dtype=np.uint16)


@pytest.fixture
def repetitive_tensor() -> np.ndarray:
    """
    Create a tensor with highly repetitive values for compression testing.
    
    Returns:
        Numpy array with only 10 unique values repeated many times
    """
    # Create array with only 10 unique values
    unique_values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=np.uint16)
    # Repeat each value 100 times
    repetitive = np.tile(unique_values, 100)
    return repetitive


@pytest.fixture
def random_tensor() -> np.ndarray:
    """
    Create a tensor with random values (should not compress well).
    
    Returns:
        Numpy array with random uint16 values
    """
    return np.random.randint(0, 65535, size=1000, dtype=np.uint16)


@pytest.fixture
def misaligned_byte_array() -> bytes:
    """
    Create a byte array of size 152064 (not divisible by 110).
    
    Returns:
        Bytes object of size 152064
    """
    # 152064 bytes = 76032 uint16 values
    data = np.random.randint(0, 65535, size=76032, dtype=np.uint16)
    return data.tobytes()


@pytest.fixture
def aligned_byte_array() -> bytes:
    """
    Create a byte array of size 152020 (divisible by 110).
    
    Returns:
        Bytes object of size 152020
    """
    # 152020 bytes = 76010 uint16 values
    data = np.random.randint(0, 65535, size=76010, dtype=np.uint16)
    return data.tobytes()

