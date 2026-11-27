import random

import pytest

from pagedserve.config import ModelConfig
from pagedserve.model import init_random


@pytest.fixture(scope="session")
def tiny_model():
    return init_random(ModelConfig.tiny(), seed=0)


@pytest.fixture(scope="session")
def prompts():
    rng = random.Random(0)
    lengths = [1, 3, 7, 16, 17, 30, 45, 5]
    return [[rng.randrange(256) for _ in range(n)] for n in lengths]
