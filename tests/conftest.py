import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402

from signsight.synthetic import make_signs  # noqa: E402


@pytest.fixture(scope="session")
def signs():
    return make_signs(tracks_per_class=8, frames=6, size=32, seed=3)
