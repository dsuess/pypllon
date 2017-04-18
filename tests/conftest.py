import numpy as np
import pytest as pt


@pt.fixture(scope="module")
def rgen():
    return np.random.RandomState(seed=3476583865)
