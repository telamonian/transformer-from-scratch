"""Tests for transformer_from_scratch.xcoder."""

import pytest
import torch as th

from transformer_from_scratch.xcoder import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    LayerNorm,
)


# --------------------------------------------------------------------------- #
# LayerNorm
# --------------------------------------------------------------------------- #
class TestLayerNorm:
    def test_params_initialised(self, dims):
        ln = LayerNorm(d_model=dims["d_model"])
        assert th.allclose(ln.β, th.zeros(dims["d_model"]))
        assert th.allclose(ln.γ, th.ones(dims["d_model"]))

    def test_normalises_last_dim(self, dims):
        ln = LayerNorm(d_model=dims["d_model"])
        x = th.randn(2, 3, dims["d_model"]) * 5 + 7  # shift & scale
        out = ln(x)
        assert out.shape == x.shape
        # with γ=1, β=0 the output is zero-mean, unit-variance along the last dim
        assert th.allclose(out.mean(dim=-1), th.zeros(2, 3), atol=1e-5)
        assert th.allclose(out.var(dim=-1, correction=0), th.ones(2, 3), atol=1e-3)

    def test_affine_params_are_applied(self, dims):
        ln = LayerNorm(d_model=dims["d_model"])
        with th.no_grad():
            ln.γ.fill_(2.0)
            ln.β.fill_(1.0)
        x = th.randn(4, dims["d_model"])
        out = ln(x)
        # mean shifts to β, std scales to γ
        assert th.allclose(out.mean(dim=-1), th.ones(4), atol=1e-5)


# --------------------------------------------------------------------------- #
# FeedForward
# --------------------------------------------------------------------------- #
class TestFeedForward:
    def test_default_d_ff_is_four_times_d_model(self, dims):
        ff = FeedForward(d_model=dims["d_model"])
        assert ff.d_ff == 4 * dims["d_model"]
        assert ff.linear_zero.out_features == 4 * dims["d_model"]
        assert ff.linear_one.out_features == dims["d_model"]

    def test_custom_d_ff(self, dims):
        ff = FeedForward(d_model=dims["d_model"], d_ff=16)
        assert ff.d_ff == 16
        assert ff.linear_zero.out_features == 16

    def test_forward_shape(self, dims):
        ff = FeedForward(d_model=dims["d_model"])
        x = th.randn(2, 3, dims["d_model"])
        assert ff(x).shape == x.shape


# --------------------------------------------------------------------------- #
# EncoderLayer / DecoderLayer
# --------------------------------------------------------------------------- #
class TestEncoderLayer:
    def test_forward_shape(self, dims):
        layer = EncoderLayer(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 4, dims["d_model"])
        out = layer(x)
        assert out.shape == x.shape
        assert not th.isnan(out).any()

    def test_forward_with_mask(self, dims):
        layer = EncoderLayer(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 4, dims["d_model"])
        mask = th.ones(2, 1, 1, 4, dtype=th.bool)
        out = layer(x, mask=mask)
        assert out.shape == x.shape


class TestDecoderLayer:
    def test_forward_shape_with_distinct_src_len(self, dims):
        layer = DecoderLayer(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 3, dims["d_model"])       # target
        out_enc = th.randn(2, 5, dims["d_model"])  # encoder memory, longer
        out = layer(x, out_enc)
        assert out.shape == x.shape
        assert not th.isnan(out).any()

    def test_forward_with_masks(self, dims):
        layer = DecoderLayer(d_model=dims["d_model"], n_heads=dims["n_heads"])
        x = th.randn(2, 3, dims["d_model"])
        out_enc = th.randn(2, 5, dims["d_model"])
        self_mask = th.tril(th.ones(3, 3, dtype=th.bool)).reshape(1, 1, 3, 3)
        cross_mask = th.ones(2, 1, 1, 5, dtype=th.bool)
        out = layer(x, out_enc, mask=self_mask, mask_enc=cross_mask)
        assert out.shape == x.shape


# --------------------------------------------------------------------------- #
# Encoder / Decoder (stacks)
# --------------------------------------------------------------------------- #
class TestEncoder:
    def test_layer_count_and_modulelist(self, dims):
        enc = Encoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=3)
        assert len(enc.layers) == 3
        assert isinstance(enc.layers, th.nn.ModuleList)

    def test_parameters_are_registered(self, dims):
        # the ModuleList fix: stacked layers' params must be discoverable
        enc = Encoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=2)
        assert len(list(enc.parameters())) > 0

    def test_forward_shape(self, dims):
        enc = Encoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=2)
        x = th.randn(2, 4, dims["d_model"])
        out = enc(x)
        assert out.shape == x.shape
        assert not th.isnan(out).any()

    def test_forward_with_mask(self, dims):
        enc = Encoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=2)
        x = th.randn(2, 4, dims["d_model"])
        mask = th.ones(2, 1, 1, 4, dtype=th.bool)
        assert enc(x, mask=mask).shape == x.shape


class TestDecoder:
    def test_layer_count(self, dims):
        dec = Decoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=3)
        assert len(dec.layers) == 3
        assert isinstance(dec.layers, th.nn.ModuleList)

    def test_forward_shape(self, dims):
        dec = Decoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=2)
        x = th.randn(2, 3, dims["d_model"])
        out_enc = th.randn(2, 5, dims["d_model"])
        out = dec(x, out_enc)
        assert out.shape == x.shape
        assert not th.isnan(out).any()

    def test_forward_with_masks(self, dims):
        dec = Decoder(d_model=dims["d_model"], n_heads=dims["n_heads"], n_layers=2)
        x = th.randn(2, 3, dims["d_model"])
        out_enc = th.randn(2, 5, dims["d_model"])
        self_mask = th.tril(th.ones(3, 3, dtype=th.bool)).reshape(1, 1, 3, 3)
        cross_mask = th.ones(2, 1, 1, 5, dtype=th.bool)
        out = dec(x, out_enc, mask=self_mask, mask_enc=cross_mask)
        assert out.shape == x.shape
