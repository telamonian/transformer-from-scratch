"""Tests for transformer_from_scratch.embedding."""
from math import sqrt

import pytest
import torch as th
from torch import nn

from transformer_from_scratch.embedding import Embedding, PositionalEncoder, xavierInit

from conftest import D_MODEL, N_VOCAB


class TestXavierInit:
    def test_shape(self):
        assert xavierInit(N_VOCAB, D_MODEL).shape == (N_VOCAB, D_MODEL)

    def test_values_within_bounds(self):
        w = xavierInit(N_VOCAB, D_MODEL)
        # fan_in = size[1], fan_out = size[0] for a 2-D weight (field size == 1)
        a = sqrt(6 / (D_MODEL + N_VOCAB))
        assert w.min() >= -a
        assert w.max() <= a

    def test_raises_for_less_than_two_dims(self):
        with pytest.raises(ValueError):
            xavierInit(5)

    def test_field_size_for_higher_rank(self):
        # For size (o, i, k1, k2) the field size is k1*k2, so bounds use that factor.
        o, i, k1, k2 = 3, 4, 2, 2
        w = xavierInit(o, i, k1, k2)
        assert w.shape == (o, i, k1, k2)
        field = k1 * k2
        a = sqrt(6 / (i * field + o * field))
        assert w.min() >= -a
        assert w.max() <= a


class TestEmbedding:
    def test_weight_is_parameter_with_shape(self):
        emb = Embedding(n_vocab=N_VOCAB, d_model=D_MODEL)
        assert isinstance(emb.weight, nn.Parameter)
        assert emb.weight.shape == (N_VOCAB, D_MODEL)

    def test_forward_shape(self):
        emb = Embedding(n_vocab=N_VOCAB, d_model=D_MODEL)
        idx = th.randint(0, N_VOCAB, (2, 5))
        out = emb(idx)
        assert out.shape == (2, 5, D_MODEL)

    def test_forward_is_gather(self):
        emb = Embedding(n_vocab=N_VOCAB, d_model=D_MODEL)
        idx = th.randint(0, N_VOCAB, (2, 5))
        # no sqrt(d_model) scaling at this layer; it's just an index into weight
        assert th.allclose(emb(idx), emb.weight[idx])


class TestPositionalEncoder:
    def test_pe_is_buffer_not_parameter(self):
        pe = PositionalEncoder(d_model=D_MODEL)
        buffer_names = {name for name, _ in pe.named_buffers()}
        assert "pe" in buffer_names
        assert len(list(pe.parameters())) == 0

    def test_default_seq_len_max(self):
        pe = PositionalEncoder(d_model=D_MODEL)
        assert pe.seq_len_max == 5000
        assert pe.pe.shape == (1, 5000, D_MODEL)

    def test_custom_seq_len_max(self):
        pe = PositionalEncoder(d_model=D_MODEL, seq_len_max=16)
        assert pe.seq_len_max == 16
        assert pe.pe.shape == (1, 16, D_MODEL)

    def test_sin_cos_values(self):
        pe = PositionalEncoder(d_model=D_MODEL, seq_len_max=16)
        # position 0: sin(0)=0 in even columns, cos(0)=1 in odd columns
        row0 = pe.pe[0, 0]
        assert th.allclose(row0[0::2], th.zeros(D_MODEL // 2), atol=1e-6)
        assert th.allclose(row0[1::2], th.ones(D_MODEL // 2), atol=1e-6)
        # spot-check a positive position against the formula
        pos = 3
        arg = pos / (10000 ** (th.arange(0, D_MODEL, 2) / D_MODEL))
        assert th.allclose(pe.pe[0, pos, 0::2], th.sin(arg), atol=1e-6)
        assert th.allclose(pe.pe[0, pos, 1::2], th.cos(arg), atol=1e-6)

    def test_forward_adds_correct_slice(self):
        pe = PositionalEncoder(d_model=D_MODEL, seq_len_max=16)
        x = th.randn(2, 5, D_MODEL)
        out = pe(x)
        assert out.shape == x.shape
        # pe broadcasts across the batch: out == x + pe[:, :seq_len]
        assert th.allclose(out, x + pe.pe[:, :5], atol=1e-6)
