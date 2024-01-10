import pytest
import random
import numpy as np
import hashlib
import sys
import os

def pytest_configure():
    # Get the directory path of the current file (conftest.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Add the directory to sys.path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

@pytest.fixture(autouse=True)
def reset_rng_state():
    random.seed(142)
    np.random.seed(142)
