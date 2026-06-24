"""Tests for transformer_from_scratch.xcoder (LayerNorm, FeedForward, En/Decoder)."""
import pytest
import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from transformer_from_scratch.xcoder import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    LayerNorm,
)

from conftest import BATCH, D_MODEL, N_HEADS, SEQ


class TestLayerNorm:
    def test_affine_params_init(self):
        ln = LayerNorm(d_model=D_MODEL)
        assert isinstance(ln.β, nn.Parameter) and isinstance(ln.γ, nn.Parameter)
        assert ln.β.shape == (D_MODEL,) and ln.γ.shape == (D_MODEL,)
        assert th.allclose(ln.β, th.zeros(D_MODEL))
        assert th.allclose(ln.γ, th.ones(D_MODEL))

    def test_normalizes_to_zero_mean_unit_var(self):
        ln = LayerNorm(d_model=D_MODEL)
        x = th.randn(BATCH, SEQ, D_MODEL)
        out = ln(x)
        assert th.allclose(out.mean(dim=-1), th.zeros(BATCH, SEQ), atol=1e-5)
        # default γ=1, β=0 so output uses population variance (correction=0) ~ 1
        var = out.var(dim=-1, correction=0)
        assert th.allclose(var, th.ones(BATCH, SEQ), atol=1e-4)

    def test_matches_torch_layer_norm(self):
        ln = LayerNorm(d_model=D_MODEL)
        x = th.randn(BATCH, SEQ, D_MODEL)
        expected = F.layer_norm(x, (D_MODEL,), weight=ln.γ, bias=ln.β, eps=ln.ε)
        assert th.allclose(ln(x), expected, atol=1e-5)

    def test_custom_epsilon_used(self):
        ln = LayerNorm(d_model=D_MODEL, ε=1e-2)
        assert ln.ε == 1e-2
        x = th.randn(BATCH, SEQ, D_MODEL)
        expected = F.layer_norm(x, (D_MODEL,), weight=ln.γ, bias=ln.β, eps=1e-2)
        assert th.allclose(ln(x), expected, atol=1e-5)

    def test_affine_params_shift_and_scale(self):
        ln = LayerNorm(d_model=D_MODEL)
        with th.no_grad():
            ln.γ.fill_(2.0)
            ln.β.fill_(3.0)
        x = th.randn(BATCH, SEQ, D_MODEL)
        base = F.layer_norm(x, (D_MODEL,), eps=ln.ε)
        assert th.allclose(ln(x), 2.0 * base + 3.0, atol=1e-5)


class TestFeedForward:
    def test_default_d_ff(self):
        ff = FeedForward(d_model=D_MODEL)
        assert ff.d_ff == 4 * D_MODEL
        assert ff.linear_zero.out_features == 4 * D_MODEL
        assert ff.linear_one.in_features == 4 * D_MODEL

    def test_custom_d_ff(self):
        ff = FeedForward(d_model=D_MODEL, d_ff=16)
        assert ff.d_ff == 16
        assert ff.linear_zero.in_features == D_MODEL
        assert ff.linear_zero.out_features == 16
        assert ff.linear_one.in_features == 16
        assert ff.linear_one.out_features == D_MODEL

    def test_forward_shape(self):
        ff = FeedForward(d_model=D_MODEL)
        x = th.randn(BATCH, SEQ, D_MODEL)
        assert ff(x).shape == (BATCH, SEQ, D_MODEL)


def _pad_mask() -> Tensor:
    mask = th.ones(BATCH, 1, 1, SEQ, dtype=th.bool)
    mask[:, :, :, -1] = False
    return mask


class TestEncoderLayer:
    def test_forward_shape(self):
        layer = EncoderLayer(d_model=D_MODEL, n_heads=N_HEADS)
        x = th.randn(BATCH, SEQ, D_MODEL)
        assert layer(x).shape == (BATCH, SEQ, D_MODEL)

    def test_forward_with_mask(self):
        layer = EncoderLayer(d_model=D_MODEL, n_heads=N_HEADS)
        x = th.randn(BATCH, SEQ, D_MODEL)
        out = layer(x, mask=_pad_mask())
        assert out.shape == (BATCH, SEQ, D_MODEL)
        assert not th.isnan(out).any()


class TestDecoderLayer:
    def test_forward_shape_with_masks(self):
        layer = DecoderLayer(d_model=D_MODEL, n_heads=N_HEADS)
        x = th.randn(BATCH, SEQ, D_MODEL)
        out_enc = th.randn(BATCH, SEQ, D_MODEL)
        causal = th.tril(th.ones(SEQ, SEQ, dtype=th.bool)).reshape(1, 1, SEQ, SEQ)
        out = layer(x, out_enc, mask=causal, mask_enc=_pad_mask())
        assert out.shape == (BATCH, SEQ, D_MODEL)
        assert not th.isnan(out).any()


class TestEncoder:
    def test_default_layer_count(self):
        enc = Encoder(d_model=D_MODEL, n_heads=N_HEADS)
        assert isinstance(enc.layers, nn.ModuleList)
        assert len(enc.layers) == 6

    def test_custom_layer_count_and_params_registered(self):
        enc = Encoder(d_model=D_MODEL, n_heads=N_HEADS, n_layers=2)
        assert len(enc.layers) == 2
        # ModuleList wrapping means the stacked layers' params are discoverable
        assert len(list(enc.parameters())) > 0

    def test_forward_shape_with_mask(self):
        enc = Encoder(d_model=D_MODEL, n_heads=N_HEADS, n_layers=2)
        x = th.randn(BATCH, SEQ, D_MODEL)
        assert enc(x, mask=_pad_mask()).shape == (BATCH, SEQ, D_MODEL)


class TestDecoder:
    def test_default_layer_count(self):
        dec = Decoder(d_model=D_MODEL, n_heads=N_HEADS)
        assert isinstance(dec.layers, nn.ModuleList)
        assert len(dec.layers) == 6

    def test_custom_layer_count_and_params_registered(self):
        dec = Decoder(d_model=D_MODEL, n_heads=N_HEADS, n_layers=2)
        assert len(dec.layers) == 2
        assert len(list(dec.parameters())) > 0

    def test_forward_shape_with_masks(self):
        dec = Decoder(d_model=D_MODEL, n_heads=N_HEADS, n_layers=2)
        x = th.randn(BATCH, SEQ, D_MODEL)
        out_enc = th.randn(BATCH, SEQ, D_MODEL)
        causal = th.tril(th.ones(SEQ, SEQ, dtype=th.bool)).reshape(1, 1, SEQ, SEQ)
        out = dec(x, out_enc=out_enc, mask=causal, mask_enc=_pad_mask())
        assert out.shape == (BATCH, SEQ, D_MODEL)
