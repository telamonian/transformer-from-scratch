"""Tests for transformer_from_scratch.transformer (masks + the full Transformer)."""
from math import sqrt

import pytest
import torch as th
from torch import nn, Tensor

from transformer_from_scratch.transformer import (
    Transformer,
    makeMaskCausal,
    makeMaskPad,
)

from conftest import BATCH, D_MODEL, N_HEADS, N_VOCAB, PAD, SEQ


class TestMakeMaskPad:
    def test_shape_and_dtype(self):
        t = th.randint(1, N_VOCAB, (BATCH, SEQ))
        mask = makeMaskPad(t, pad_elem=PAD)
        assert mask.shape == (BATCH, 1, 1, SEQ)
        assert mask.dtype == th.bool

    def test_true_for_non_pad_false_for_pad(self):
        t = th.tensor([[1, 2, PAD, PAD, 3]])
        mask = makeMaskPad(t, pad_elem=PAD)
        expected = th.tensor([True, True, False, False, True])
        assert th.equal(mask.reshape(-1), expected)


class TestMakeMaskCausal:
    def test_shape_and_dtype(self):
        t = th.randint(1, N_VOCAB, (BATCH, SEQ))
        mask = makeMaskCausal(t)
        assert mask.shape == (1, 1, SEQ, SEQ)
        assert mask.dtype == th.bool

    def test_is_lower_triangular(self):
        t = th.randint(1, N_VOCAB, (BATCH, SEQ))
        mask = makeMaskCausal(t)
        expected = th.tril(th.ones(SEQ, SEQ, dtype=th.bool))
        assert th.equal(mask.reshape(SEQ, SEQ), expected)


def _model(**kwargs) -> Transformer:
    defaults = dict(d_model=D_MODEL, n_heads=N_HEADS, n_vocab=N_VOCAB)
    defaults.update(kwargs)
    return Transformer(**defaults)


class TestTransformerInit:
    def test_n_vocab_tgt_defaults_to_n_vocab(self):
        model = _model()
        assert model.n_vocab_tgt == N_VOCAB

    def test_n_vocab_tgt_explicit(self):
        model = _model(n_vocab_tgt=7)
        assert model.n_vocab_tgt == 7
        assert model.output.out_features == 7

    def test_d_model_sqrt(self):
        model = _model()
        assert model.d_model_sqrt == sqrt(D_MODEL)

    def test_output_is_bias_free_linear(self):
        model = _model()
        assert isinstance(model.output, nn.Linear)
        assert model.output.bias is None
        assert model.output.out_features == N_VOCAB

    def test_tie_embed_shares_module(self):
        tied = _model(tie_embed=True)
        assert tied.embedding_dec is tied.embedding_enc
        untied = _model(tie_embed=False)
        assert untied.embedding_dec is not untied.embedding_enc

    def test_tie_output_shares_weight(self):
        model = _model(tie_output=True)
        assert model.output.weight is model.embedding_dec.weight


class TestTransformerForward:
    def test_no_pad_branch_output_shape(self):
        # pad_elem=None -> src_mask is None and tgt uses a pure causal mask
        model = _model(pad_elem=None)
        src = th.randint(1, N_VOCAB, (BATCH, SEQ))
        tgt = th.randint(1, N_VOCAB, (BATCH, SEQ - 1))
        out = model(src, tgt)
        assert out.shape == (BATCH, SEQ - 1, N_VOCAB)
        assert not th.isnan(out).any()

    def test_pad_branch_handles_padding(self, tokens_padded: Tensor):
        # pad_elem set -> src padding mask + (pad & causal) tgt mask; includes an all-pad row
        model = _model(pad_elem=PAD)
        out = model(tokens_padded, tokens_padded)
        assert out.shape == (BATCH, SEQ, N_VOCAB)
        # the fully-padded row must not produce NaNs (softmax per-row-max fix end to end)
        assert not th.isnan(out).any()

    def test_distinct_tgt_vocab_in_output_dim(self):
        model = _model(n_vocab_tgt=7, pad_elem=None)
        src = th.randint(1, N_VOCAB, (BATCH, SEQ))
        tgt = th.randint(0, 7, (BATCH, SEQ))
        out = model(src, tgt)
        assert out.shape == (BATCH, SEQ, 7)

    def test_deterministic_under_seed(self):
        src = th.randint(1, N_VOCAB, (BATCH, SEQ))
        tgt = th.randint(1, N_VOCAB, (BATCH, SEQ))

        th.manual_seed(0)
        out1 = _model(pad_elem=None)(src, tgt)
        th.manual_seed(0)
        out2 = _model(pad_elem=None)(src, tgt)
        assert th.allclose(out1, out2, atol=1e-6)
