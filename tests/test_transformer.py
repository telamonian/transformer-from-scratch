"""Tests for transformer_from_scratch.transformer."""

import pytest
import torch as th

from transformer_from_scratch.transformer import Transformer, makeMaskCausal, makeMaskPad


# --------------------------------------------------------------------------- #
# mask builders
# --------------------------------------------------------------------------- #
class TestMaskBuilders:
    def test_pad_mask_shape_and_values(self):
        t = th.tensor([[5, 5, 0, 0]])
        mask = makeMaskPad(t, pad_elem=0)
        assert mask.shape == (1, 1, 1, 4)
        assert mask.dtype == th.bool
        assert mask.flatten().tolist() == [True, True, False, False]

    def test_pad_mask_batch(self):
        t = th.tensor([[1, 0], [2, 3]])
        mask = makeMaskPad(t, pad_elem=0)
        assert mask.shape == (2, 1, 1, 2)
        assert mask[0].flatten().tolist() == [True, False]
        assert mask[1].flatten().tolist() == [True, True]

    def test_causal_mask_is_lower_triangular_bool(self):
        t = th.zeros(1, 3)
        mask = makeMaskCausal(t)
        assert mask.shape == (1, 1, 3, 3)
        assert mask.dtype == th.bool
        assert th.equal(mask[0, 0], th.tril(th.ones(3, 3, dtype=th.bool)))

    def test_combined_target_mask_blocks_future_and_pad(self):
        # mirrors Transformer.forward: pad AND causal
        tgt = th.tensor([[1, 2, 0]])  # last token is padding
        combined = makeMaskPad(tgt, pad_elem=0) & makeMaskCausal(tgt)
        # query 0 may only see key 0
        assert combined[0, 0, 0].tolist() == [True, False, False]
        # query 1 sees keys 0,1
        assert combined[0, 0, 1].tolist() == [True, True, False]
        # padding key (col 2) is never visible to anyone
        assert combined[0, 0, :, 2].tolist() == [False, False, False]


# --------------------------------------------------------------------------- #
# Transformer construction
# --------------------------------------------------------------------------- #
class TestTransformerInit:
    def test_default_target_vocab_matches_source(self, dims):
        m = Transformer(d_model=dims["d_model"], n_heads=dims["n_heads"], n_vocab=10)
        assert m.n_vocab_tgt == 10
        assert m.output.out_features == 10
        assert m.output.in_features == dims["d_model"]

    def test_explicit_target_vocab(self, dims):
        m = Transformer(
            d_model=dims["d_model"], n_heads=dims["n_heads"], n_vocab=10, n_vocab_tgt=7
        )
        assert m.n_vocab_tgt == 7
        assert m.output.out_features == 7
        assert m.embedding_dec.weight.shape == (7, dims["d_model"])

    def test_tie_embed_shares_embedding_module(self, dims):
        m = Transformer(
            d_model=dims["d_model"], n_heads=dims["n_heads"], n_vocab=10, tie_embed=True
        )
        assert m.embedding_dec is m.embedding_enc

    def test_untied_embeddings_are_distinct(self, dims):
        m = Transformer(d_model=dims["d_model"], n_heads=dims["n_heads"], n_vocab=10)
        assert m.embedding_dec is not m.embedding_enc

    def test_tie_output_shares_weight_with_decoder_embedding(self, dims):
        m = Transformer(
            d_model=dims["d_model"], n_heads=dims["n_heads"], n_vocab=10, tie_output=True
        )
        assert m.output.weight is m.embedding_dec.weight


# --------------------------------------------------------------------------- #
# Transformer.forward
# --------------------------------------------------------------------------- #
class TestTransformerForward:
    def _model(self, dims, **kwargs):
        return Transformer(
            d_model=dims["d_model"], n_heads=dims["n_heads"], n_vocab=10, **kwargs
        )

    def test_forward_output_shape_no_padding(self, dims):
        m = self._model(dims)
        src = th.randint(0, 10, (2, 4))
        tgt = th.randint(0, 10, (2, 3))
        out = m(src, tgt)
        assert out.shape == (2, 3, 10)  # [batch, tgt_len, n_vocab_tgt]
        assert not th.isnan(out).any()

    def test_forward_with_padding_path(self, dims):
        # pad_elem set -> exercises makeMaskPad branch + src/cross masks
        m = self._model(dims, pad_elem=0)
        src = th.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
        tgt = th.tensor([[1, 2, 0], [8, 9, 0]])
        out = m(src, tgt)
        assert out.shape == (2, 3, 10)
        assert not th.isnan(out).any()

    def test_forward_explicit_target_vocab_shape(self, dims):
        m = self._model(dims, n_vocab_tgt=7)
        src = th.randint(0, 10, (2, 4))
        tgt = th.randint(0, 7, (2, 5))
        out = m(src, tgt)
        assert out.shape == (2, 5, 7)

    def test_forward_is_deterministic_under_seed(self, dims):
        src = th.randint(0, 10, (1, 4))
        tgt = th.randint(0, 10, (1, 3))

        th.manual_seed(123)
        m1 = self._model(dims)
        out1 = m1(src, tgt)

        th.manual_seed(123)
        m2 = self._model(dims)
        out2 = m2(src, tgt)

        assert th.allclose(out1, out2, atol=1e-6)
