"""Tests for transformer_from_scratch.attention."""
from math import sqrt

import pytest
import torch as th
from torch import nn, Tensor

from transformer_from_scratch.attention import (
    MultiHeadAttention,
    attentionSDP,
    attentionSDP_oneline,
    softmax,
)

from conftest import BATCH, D_K, D_MODEL, N_HEADS, SEQ


class TestSoftmax:
    def test_sums_to_one(self):
        x = th.randn(4, 6)
        out = softmax(x, dim=-1)
        assert th.allclose(out.sum(dim=-1), th.ones(4), atol=1e-6)

    def test_matches_torch_softmax(self):
        x = th.randn(3, 5, 7)
        assert th.allclose(softmax(x, dim=-1), th.softmax(x, dim=-1), atol=1e-6)

    def test_non_default_dim(self):
        x = th.randn(3, 5)
        out = softmax(x, dim=0)
        # normalisation is along dim 0, so each column should sum to one
        assert th.allclose(out.sum(dim=0), th.ones(5), atol=1e-6)
        assert th.allclose(out, th.softmax(x, dim=0), atol=1e-6)

    def test_dtype_argument_honored(self):
        x = th.randn(3, 4)
        out = softmax(x, dim=-1, dtype=th.float64)
        assert out.dtype == th.float64

    def test_fully_masked_row_is_nan_free(self):
        # A row that is entirely -1e9 mimics a fully-padded attention row. The per-row-max
        # trick must keep this NaN-free (uniform distribution), unlike a naive global-max softmax.
        x = th.tensor([[1.0, 2.0, 3.0], [-1e9, -1e9, -1e9]])
        out = softmax(x, dim=-1)
        assert not th.isnan(out).any()
        assert th.allclose(out.sum(dim=-1), th.ones(2), atol=1e-6)
        # the all-masked row collapses to a uniform distribution
        assert th.allclose(out[1], th.full((3,), 1 / 3), atol=1e-6)

    def test_gradient_flows(self):
        # softmax is fed grad-requiring scores during forward; the all-out-of-place math
        # must preserve autograd.
        x = th.randn(2, 4, requires_grad=True)
        softmax(x, dim=-1).sum().backward()
        assert x.grad is not None
        assert not th.isnan(x.grad).any()


class TestAttentionSDP:
    def _qkv(self) -> tuple[Tensor, Tensor, Tensor]:
        q = th.randn(BATCH, N_HEADS, SEQ, D_K)
        k = th.randn(BATCH, N_HEADS, SEQ, D_K)
        v = th.randn(BATCH, N_HEADS, SEQ, D_K)
        return q, k, v

    def test_returns_context_and_weights(self):
        q, k, v = self._qkv()
        context, weights = attentionSDP(q, k, v)
        assert context.shape == (BATCH, N_HEADS, SEQ, D_K)
        assert weights.shape == (BATCH, N_HEADS, SEQ, SEQ)

    def test_weights_match_scaled_scores(self):
        q, k, v = self._qkv()
        _, weights = attentionSDP(q, k, v)
        expected = q.matmul(k.transpose(-2, -1)) / sqrt(D_K)
        assert th.allclose(weights, expected, atol=1e-6)

    def test_context_matches_reference(self):
        q, k, v = self._qkv()
        context, _ = attentionSDP(q, k, v)
        scores = q.matmul(k.transpose(-2, -1)) / sqrt(D_K)
        expected = th.softmax(scores, dim=-1).matmul(v)
        assert th.allclose(context, expected, atol=1e-6)

    def test_mask_zeroes_attention_to_masked_keys(self):
        q, k, v = self._qkv()
        # mask out the last key position for every query (shape broadcasts over heads/queries)
        mask = th.ones(BATCH, 1, 1, SEQ, dtype=th.bool)
        mask[:, :, :, -1] = False
        context, _ = attentionSDP(q, k, v, mask=mask)
        # recompute attention weights the way attentionSDP does and confirm masked column ≈ 0
        scores = q.matmul(k.transpose(-2, -1)) / sqrt(D_K)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = th.softmax(scores, dim=-1)
        assert th.allclose(attn[:, :, :, -1], th.zeros(BATCH, N_HEADS, SEQ), atol=1e-6)
        # and the masked-attention context matches the reference built from those weights
        assert th.allclose(context, attn.matmul(v), atol=1e-6)


class TestAttentionSDPOneline:
    def test_returns_single_tensor(self):
        q = th.randn(BATCH, N_HEADS, SEQ, D_K)
        k = th.randn(BATCH, N_HEADS, SEQ, D_K)
        v = th.randn(BATCH, N_HEADS, SEQ, D_K)
        out = attentionSDP_oneline(q, k, v)
        assert isinstance(out, th.Tensor)
        assert out.shape == (BATCH, N_HEADS, SEQ, D_K)

    def test_equivalent_to_attentionSDP_without_mask(self):
        q = th.randn(BATCH, N_HEADS, SEQ, D_K)
        k = th.randn(BATCH, N_HEADS, SEQ, D_K)
        v = th.randn(BATCH, N_HEADS, SEQ, D_K)
        context, _ = attentionSDP(q, k, v)
        assert th.allclose(attentionSDP_oneline(q, k, v), context, atol=1e-6)


class TestMultiHeadAttention:
    def test_init_dims_and_layers(self):
        mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        assert mha.d_k == D_MODEL // N_HEADS
        for layer in (mha.q_linear, mha.k_linear, mha.v_linear, mha.out_linear):
            assert isinstance(layer, nn.Linear)
            assert layer.in_features == D_MODEL
            assert layer.out_features == D_MODEL

    def test_forward_preserves_shape(self):
        mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        x = th.randn(BATCH, SEQ, D_MODEL)
        out = mha(x, x, x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_forward_with_mask(self):
        mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        x = th.randn(BATCH, SEQ, D_MODEL)
        mask = th.ones(BATCH, 1, 1, SEQ, dtype=th.bool)
        mask[:, :, :, -1] = False
        out = mha(x, x, x, mask=mask)
        assert out.shape == (BATCH, SEQ, D_MODEL)
        assert not th.isnan(out).any()

    def test_unfold_shape(self):
        mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        t = th.randn(BATCH, SEQ, D_MODEL)
        assert mha.unfold(t).shape == (BATCH, N_HEADS, SEQ, D_MODEL // N_HEADS)

    def test_fold_shape(self):
        mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        t = th.randn(BATCH, N_HEADS, SEQ, D_MODEL // N_HEADS)
        assert mha.fold(t).shape == (BATCH, SEQ, D_MODEL)

    def test_fold_unfold_roundtrip(self):
        mha = MultiHeadAttention(d_model=D_MODEL, n_heads=N_HEADS)
        t = th.randn(BATCH, SEQ, D_MODEL)
        assert th.allclose(mha.fold(mha.unfold(t)), t, atol=1e-6)
