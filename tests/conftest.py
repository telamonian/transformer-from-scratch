"""Shared pytest configuration and fixtures.

The package lives under ``pysrc/`` (see ``[tool.uv.build-backend]`` in
pyproject.toml).  ``[tool.pytest.ini_options].pythonpath`` already adds that
directory to ``sys.path`` when pytest runs, but we also insert it here so the
suite works even when invoked without the project config (e.g. ``pytest tests``
from an odd cwd, or via an IDE runner).
"""

import os
import sys

import pytest
import torch as th

PYSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pysrc")
if PYSRC not in sys.path:
    sys.path.insert(0, PYSRC)


@pytest.fixture(autouse=True)
def _seed():
    """Make every test deterministic.

    ``autouse`` so individual tests don't have to remember to seed before
    creating randomly-initialised modules / tensors.
    """
    th.manual_seed(0)
    yield


# ---- small, fast model dimensions reused across the suite ----
# d_model must be even (PositionalEncoder splits the last dim into sin/cos
# halves) and divisible by n_heads (MultiHeadAttention splits d_model into
# n_heads * d_k).
D_MODEL = 8
N_HEADS = 2
D_K = D_MODEL // N_HEADS


@pytest.fixture
def dims():
    return {"d_model": D_MODEL, "n_heads": N_HEADS, "d_k": D_K}
