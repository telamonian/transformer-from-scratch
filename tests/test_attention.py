"""Tests for transformer_from_scratch.attention."""

import math

import pytest
import torch as th

from transformer_from_scratch.attention import (
    MultiHeadAttention,
    attentionSDP,
    attentionSDP_oneline,
    softmax,
)


# --------------------------------------------------------------------------- #
# softmax
# --------------------------------------------------------------------------- #
class TestSoftmax:
    def test_matches_torch_softmax(self):
        x = th.randn(2, 3, 5)
        out = softmax(x, dim=-1)
        assert th.allclose(out, th.softmax(x, dim=-1), atol=1e-6)

    def test_rows_sum_to_one(self):
        x = th.randn(4, 6)
        out = softmax(x, dim=-1)
        assert th.allclose(out.sum(dim=-1), th.ones(4), atol=1e-6)

    def test_respects_dim_argument(self):
        x = th.randn(3, 4)
        out = softmax(x, dim=0)
        # normalising along dim 0 => each column sums to 1
        assert th.allclose(out.sum(dim=0), th.ones(4), atol=1e-6)

    def test_dtype_argument_is_honoured(self):
        # exercise the ``else dtype`` branch on line 6
        x = th.randn(2, 3, dtype=th.float32)
        out = softmax(x, dim=-1, dtype=th.float32)
        assert out.dtype == th.float32

    def test_fully_masked_row_is_uniform_not_nan(self):
        # the per-row-max fix: a row of all -1e9 must collapse to a uniform
        # distribution instead of producing 0/0 = NaN.
        x = th.full((1, 4), -1e9)
        out = softmax(x, dim=-1)
        assert not th.isnan(out).any()
        assert th.allclose(out, th.full((1, 4), 0.25), atol=1e-6)

    def test_numerically_stable_for_large_logits(self):
        x = th.tensor([[1000.0, 1001.0, 1002.0]])
        out = softmax(x, dim=-1)
        assert not th.isnan(out).any()
        assert th.allclose(out.sum(dim=-1), th.ones(1), atol=1e-6)


# --------------------------------------------------------------------------- #
# attentionSDP / attentionSDP_oneline
# --------------------------------------------------------------------------- #
class TestAttentionSDP:
    def _qkv(self, batch=2, heads=2, n=3, d_k=4):
        q = th.randn(batch, heads, n, d_k)
        k = th.randn(batch, heads, n, d_k)
        v = th.randn(batch, heads, n, d_k)
        return q, k, v

    def test_output_and_weight_shapes(self):
        q, k, v = self._qkv()
        out, weights = attentionSDP(q, k, v)
        assert out.shape == v.shape
        assert weights.shape == (2, 2, 3, 3)  # [batch, heads, n_q, n_k]

    def test_weights_are_pre_softmax_scores(self):
        # attentionSDP returns the raw (scaled, possibly masked) scores, not the
        # softmaxed weights.
        q, k, v = self._qkv()
        _, weights = attentionSDP(q, k, v)
        d_k = k.shape[-1]
        expected = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)
        assert th.allclose(weights, expected, atol=1e-6)

    def test_mask_forces_attention_onto_visible_key(self):
        # only key index 0 is visible -> every query should read value[...,0,:]
        q, k, v = self._qkv()
        mask = th.zeros(1, 1, 1, 3, dtype=th.bool)
        mask[..., 0] = True
        out, _ = attentionSDP(q, k, v, mask=mask)
        expected = v[:, :, 0:1, :].expand_as(out)
        assert th.allclose(out, expected, atol=1e-4)

    def test_oneline_matches_unmasked_attentionSDP(self):
        q, k, v = self._qkv()
        out_full, _ = attentionSDP(q, k, v)
        out_oneline = attentionSDP_oneline(q, k, v)
        assert th.allclose(out_full, out_oneline, atol=1e-6)


# --------------------------------------------------------------------------- #
# MultiHeadAttention
# --------------------------------------------------------------------------- #
class TestMultiHeadAttention:
    def test_init_computes_d_k(self, dims):
        mha = MultiHeadAttention(d_model=dims["d_model"], n_heads=dims["n_heads"])
        assert mha.d_k == dims["d_k"]
        # four projection matrices, each d_model x d_model
        for layer in (mha.q_linear, mha.k_linear, mha.v_linear, mha.out_linear):
            assert layer.in_features == dims["d_model"]
            assert layer.out_features == dims["d_model"]

    def test_forward_preserves_shape(self, dims):
        mha = MultiHeadAttention(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 5, dims["d_model"])
        out = mha(x, x, x)
        assert out.shape == x.shape

    def test_forward_runs_with_mask(self, dims):
        mha = MultiHeadAttention(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 4, dims["d_model"])
        mask = th.tril(th.ones(4, 4, dtype=th.bool)).reshape(1, 1, 4, 4)
        out = mha(x, x, x, mask=mask)
        assert out.shape == x.shape
        assert not th.isnan(out).any()

    def test_unfold_then_fold_is_identity(self, dims):
        mha = MultiHeadAttention(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 6, dims["d_model"])
        recovered = mha.fold(mha.unfold(x))
        assert recovered.shape == x.shape
        assert th.allclose(recovered, x, atol=1e-6)

    def test_unfold_produces_head_split_shape(self, dims):
        mha = MultiHeadAttention(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 6, dims["d_model"])
        unfolded = mha.unfold(x)
        assert unfolded.shape == (2, dims["n_heads"], 6, dims["d_k"])
