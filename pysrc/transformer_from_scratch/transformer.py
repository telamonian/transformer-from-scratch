from math import sqrt
import torch as th
import torch.nn as nn

from transformer_from_scratch.embedding import Embedding, PositionalEncoder
from transformer_from_scratch.xcoder import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_vocab, n_vocab_tgt=None, d_ff=None, tie_embed=False, tie_output=False):
        super().__init__()

        self.d_model = d_model
        # precalculate
        self.d_model_sqrt = sqrt(self.d_model)
        self.n_heads = n_heads
        self.n_vocab = n_vocab
        self.n_vocab_tgt = n_vocab if n_vocab_tgt is None else n_vocab_tgt
        self.d_ff = d_ff
        self.tie_embed = tie_embed
        self.tie_output = tie_output

        self.embedding_enc = Embedding(n_vocab=self.n_vocab, d_model=self.d_model)
        if self.tie_embed:
            self.embedding_dec = self.embedding_enc
        else:
            self.embedding_dec = Embedding(n_vocab=self.n_vocab_tgt, d_model=self.d_model)

        self.positional_enc = PositionalEncoder(n_vocab=self.n_vocab, d_model=self.d_model)

        self.encoder = Encoder(d_model=self.d_model, n_heads=self.n_heads, n_layers=6, d_ff=self.d_ff, mask=None)
        self.decoder = Decoder(d_model=self.d_model, n_heads=self.n_heads, n_layers=6, d_ff=self.d_ff, mask=None, mask_enc=None)

        self.output = nn.Linear(self.d_model, self.n_vocab_tgt, bias=False)

        if tie_output:
            # TODO does the right hand side need to be transposed?
            self.output.weight = self.embedding_dec.weight

    def forward(self, src, tgt):
        # TODO figure out masking
        src = self.embedding_enc(src)*self.d_model_sqrt
        tgt = self.embedding_dec(tgt)*self.d_model_sqrt

        src = self.positional_enc(src)
        tgt = self.positional_enc(tgt)

        src = self.encoder(src)
        tgt = self.decoder(tgt, out_enc=src)

        return self.output(tgt)
