import pytest
import random
import numpy as np


@pytest.fixture(autouse=True)
def reset_rng_state():
    random.seed(142)
    np.random.seed(142)
