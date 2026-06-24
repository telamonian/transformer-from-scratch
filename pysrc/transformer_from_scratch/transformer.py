from math import sqrt
import torch as th
import torch.nn as nn

from transformer_from_scratch.embedding import Embedding, PositionalEncoder
from transformer_from_scratch.xcoder import Encoder, Decoder

def makeMaskPad(t, pad_elem):
    """expects t shape to be [batch_size, n_samples]
    n_samples is equivalent to what is called seq_len in some discussions/implementations
    """
    batch_size, n_samples = t.shape
    return (t != pad_elem).reshape(batch_size, 1, 1, n_samples)

def makeMaskCausal(t):
    _, n_samples = t.shape
    return th.tril(th.ones(n_samples, n_samples, dtype=th.bool)).reshape(1, 1, n_samples, n_samples)

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_vocab, n_vocab_tgt=None, d_ff=None, pad_elem=None, tie_embed=False, tie_output=False):
        super().__init__()

        self.d_model = d_model
        # precalculate
        self.d_model_sqrt = sqrt(self.d_model)
        self.n_heads = n_heads
        self.n_vocab = n_vocab
        self.n_vocab_tgt = n_vocab if n_vocab_tgt is None else n_vocab_tgt
        self.d_ff = d_ff
        # element that represents padding in input
        self.pad_elem = pad_elem
        self.tie_embed = tie_embed
        self.tie_output = tie_output

        self.embedding_enc = Embedding(n_vocab=self.n_vocab, d_model=self.d_model)
        if self.tie_embed:
            self.embedding_dec = self.embedding_enc
        else:
            self.embedding_dec = Embedding(n_vocab=self.n_vocab_tgt, d_model=self.d_model)

        self.positional_enc = PositionalEncoder(n_vocab=self.n_vocab, d_model=self.d_model)

        self.encoder = Encoder(d_model=self.d_model, n_heads=self.n_heads, n_layers=6, d_ff=self.d_ff)
        self.decoder = Decoder(d_model=self.d_model, n_heads=self.n_heads, n_layers=6, d_ff=self.d_ff)

        self.output = nn.Linear(self.d_model, self.n_vocab_tgt, bias=False)

        if self.tie_output:
            # TODO does the right hand side need to be transposed?
            self.output.weight = self.embedding_dec.weight

    def forward(self, src, tgt):
        # build masks from the raw token ids before embedding.
        # src_mask is used for encoder self-attention and is reused for decoder cross-attention. It hides any padding, if present, in the src
        # tgt_mask is used for decoder self-attention. It always includes the causal mask that ensures positions cannot see the "future", and also hides any padding, if present, in the tgt
        if self.pad_elem is None:
            src_mask = None
            tgt_mask = makeMaskCausal(tgt)
        else:
            src_mask = makeMaskPad(src, pad_elem=self.pad_elem)
            tgt_mask = makeMaskPad(tgt, pad_elem=self.pad_elem) & makeMaskCausal(tgt)

        src = self.embedding_enc(src)*self.d_model_sqrt
        tgt = self.embedding_dec(tgt)*self.d_model_sqrt

        src = self.positional_enc(src)
        tgt = self.positional_enc(tgt)

        src = self.encoder(src, mask=src_mask)
        tgt = self.decoder(tgt, out_enc=src, mask=tgt_mask, mask_enc=src_mask)

        return self.output(tgt)
