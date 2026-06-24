"""Shared fixtures and small, fast model dimensions for the test suite.

Everything is deliberately tiny so the whole suite runs in well under a second, and a fixed
seed is applied before every test so results are deterministic and comparable run-to-run.
"""
import pytest
import torch as th

# Small dims keep the tests fast while still exercising every code path. d_model must stay
# divisible by n_heads (d_k = d_model // n_heads).
D_MODEL = 8
N_HEADS = 2
D_K = D_MODEL // N_HEADS
BATCH = 2
SEQ = 5
N_VOCAB = 11
PAD = 0


@pytest.fixture(autouse=True)
def _seed():
    """Re-seed before every test so randomly initialised modules / inputs are reproducible."""
    th.manual_seed(0)


@pytest.fixture
def dims():
    """Bundle of the small dimensions, handed to tests that want to read them by name."""
    return {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "d_k": D_K,
        "batch": BATCH,
        "seq": SEQ,
        "n_vocab": N_VOCAB,
        "pad": PAD,
    }


@pytest.fixture
def tokens():
    """A plain (unpadded) batch of token ids, shape [BATCH, SEQ]."""
    # low=1 so we never accidentally emit the PAD id (0) in the "no padding" batch.
    return th.randint(low=1, high=N_VOCAB, size=(BATCH, SEQ))


@pytest.fixture
def tokens_padded():
    """A batch of token ids with trailing PAD, including one fully-padded row.

    Row 0 has real tokens in the first two positions then PAD; row 1 is entirely PAD. The
    all-PAD row is the stress case for the softmax NaN fix (a fully-masked attention row).
    """
    t = th.randint(low=1, high=N_VOCAB, size=(BATCH, SEQ))
    t[0, 2:] = PAD
    t[1, :] = PAD
    return t
