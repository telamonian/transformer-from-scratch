"""Tests for transformer_from_scratch.embedding."""

import math

import pytest
import torch as th

from transformer_from_scratch.embedding import Embedding, PositionalEncoder, xavierInit


# --------------------------------------------------------------------------- #
# xavierInit
# --------------------------------------------------------------------------- #
class TestXavierInit:
    def test_shape_2d(self):
        t = xavierInit(4, 8)
        assert t.shape == (4, 8)

    def test_values_within_bound_2d(self):
        n_out, n_in = 4, 8
        t = xavierInit(n_out, n_in)
        a = math.sqrt(6 / (n_in + n_out))  # fan_in=n_in, fan_out=n_out (fieldSize=1)
        assert t.abs().max().item() <= a

    def test_higher_rank_uses_field_size(self):
        # for >2 dims, fan_in/out fold in the trailing "field" dims
        t = xavierInit(2, 3, 4)
        assert t.shape == (2, 3, 4)
        field = 4
        a = math.sqrt(6 / (3 * field + 2 * field))
        assert t.abs().max().item() <= a

    def test_raises_for_fewer_than_two_dims(self):
        with pytest.raises(ValueError):
            xavierInit(5)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class TestEmbedding:
    def test_weight_shape_and_is_parameter(self, dims):
        emb = Embedding(n_vocab=10, d_model=dims["d_model"])
        assert emb.weight.shape == (10, dims["d_model"])
        # registered as a trainable parameter
        assert any(p is emb.weight for p in emb.parameters())

    def test_forward_indexes_rows(self, dims):
        emb = Embedding(n_vocab=10, d_model=dims["d_model"])
        x = th.tensor([[1, 2, 3], [4, 5, 6]])
        out = emb(x)
        assert out.shape == (2, 3, dims["d_model"])
        # forward is a plain lookup into the weight table
        assert th.allclose(out, emb.weight[x])


# --------------------------------------------------------------------------- #
# PositionalEncoder
# --------------------------------------------------------------------------- #
class TestPositionalEncoder:
    def test_pe_buffer_shape_and_default_len(self, dims):
        pe = PositionalEncoder(d_model=dims["d_model"])
        assert pe.seq_len_max == 5000
        assert pe.pe.shape == (1, 5000, dims["d_model"])

    def test_pe_registered_as_buffer_not_parameter(self, dims):
        pe = PositionalEncoder(d_model=dims["d_model"])
        assert "pe" in dict(pe.named_buffers())
        assert "pe" not in dict(pe.named_parameters())

    def test_seq_len_max_override(self, dims):
        pe = PositionalEncoder(d_model=dims["d_model"], seq_len_max=16)
        assert pe.pe.shape == (1, 16, dims["d_model"])

    def test_first_position_sin_cos_values(self, dims):
        pe = PositionalEncoder(d_model=dims["d_model"])
        row0 = pe.pe[0, 0]
        # even indices use sin(0) == 0, odd indices use cos(0) == 1
        assert th.allclose(row0[0::2], th.zeros_like(row0[0::2]), atol=1e-6)
        assert th.allclose(row0[1::2], th.ones_like(row0[1::2]), atol=1e-6)

    def test_forward_adds_leading_slice(self, dims):
        pe = PositionalEncoder(d_model=dims["d_model"], seq_len_max=16)
        x = th.zeros(2, 5, dims["d_model"])
        out = pe(x)
        assert out.shape == x.shape
        # adding to zeros returns exactly the positional slice (broadcast over batch)
        assert th.allclose(out, pe.pe[:, :5].expand_as(out))
