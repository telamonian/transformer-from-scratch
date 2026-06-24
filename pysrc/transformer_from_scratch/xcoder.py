import torch as th
import torch.nn as nn

from transformer_from_scratch.attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, d_model, ε=1e-5):
        super().__init__()

        self.d_model = d_model
        self.ε = ε

        self.β = nn.Parameter(th.zeros(self.d_model))
        self.γ = nn.Parameter(th.ones(self.d_model))

    def forward(self, x):
        μ = th.mean(x, -1, keepdim=True)

        # σσ is meant to be σ^2, ie the variance
        σσ = th.var(x, -1, keepdim=True, correction=0)

        return self.γ * (x - μ)/th.sqrt(σσ + self.ε) + self.β

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = 4*self.d_model if d_ff is None else d_ff

        self.linear_zero = nn.Linear(self.d_model, self.d_ff)
        self.activation_one = nn.ReLU()
        self.linear_one = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        x = self.linear_zero(x)
        x = self.activation_one(x)
        x = self.linear_one(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.mha_zero = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads)
        self.norm_zero = LayerNorm(d_model=self.d_model)

        self.ff_one = FeedForward(d_model=self.d_model, d_ff=self.d_ff)
        self.norm_one = LayerNorm(d_model=self.d_model)

    def forward(self, x, mask=None):
        """x: source sequence that has been passed through an embedding. shape: [batch_size, n_samples, d_model]
        """
        residual = x
        x = self.mha_zero(x, x, x, mask=mask)
        x += residual
        x = self.norm_zero(x)

        residual = x
        x = self.ff_one(x)
        x += residual
        x = self.norm_one(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.mha_zero = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads)
        self.norm_zero = LayerNorm(d_model=self.d_model)

        self.mha_one = MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads)
        self.norm_one = LayerNorm(d_model=self.d_model)

        self.ff_two = FeedForward(d_model=self.d_model, d_ff=self.d_ff)
        self.norm_two = LayerNorm(d_model=self.d_model)

    def forward(self, x, out_enc, mask=None, mask_enc=None):
        """x: target sequence that has been passed through an embedding. shape: [batch_size, n_samples, d_model]
        out_enc: the output from the encoder
        mask: mask for self-attention. Should account for both padding in the samples and causal masking
        mask_enc: mask for cross-attention. This will be the same mask used by the encoder, and accounts for padding
        """
        residual = x
        x = self.mha_zero(x, x, x, mask=mask)
        x += residual
        x = self.norm_zero(x)

        residual = x
        x = self.mha_one(x, out_enc, out_enc, mask=mask_enc)
        x += residual
        x = self.norm_one(x)

        residual = x
        x = self.ff_two(x)
        x += residual
        x = self.norm_two(x)

        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=6, d_ff=None):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        # plain list of layers needs to be wrapped in ModuleList for pytorch to treat them correctly as model layers
        self.layers = nn.ModuleList([EncoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
        ) for _ in range(self.n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=6, d_ff=None):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        # plain list of layers needs to be wrapped in ModuleList for pytorch to treat them correctly as model layers
        self.layers = nn.ModuleList([DecoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
        ) for _ in range(self.n_layers)])

    def forward(self, x, out_enc, mask=None, mask_enc=None):
        for layer in self.layers:
            x = layer(x, out_enc, mask=mask, mask_enc=mask_enc)

        return x
