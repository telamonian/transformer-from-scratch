from math import sqrt
import torch as th
import torch.nn as nn
from torch.nn.functional import softmax

def attentionSDP(queries, keys, values, mask=None):
    """scaled dot-product attention
    """

    weights = queries.matmul(keys.transpose(-2, -1))/sqrt(len(keys))

    if mask is not None:
        weights = weights.masked_fill(mask == 0, -1e9)

    return softmax(weights).matmul(values), weights

def attentionSDP_oneline(queries, keys, values):
    return softmax(queries.matmul(keys.transpose(-2, -1))/sqrt(len(keys))).matmul(values)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, queries, keys, values, mask=None):
        queries = self.unfold(self.q_linear(queries))
        keys = self.unfold(self.k_linear(keys))
        values = self.unfold(self.v_linear(values))

        values, _ = attentionSDP(queries, keys, values, mask=mask)

        return self.out_linear(self.fold(values))

    def unfold(self, t):
        """tensor t starts out with shape [batch_size, n_samples, d_model],
        ends up with [batch_size, n_heads, n_samples, d_k]
        """
        batch_size, n_samples, _ = t.shape
        return t.view(batch_size, n_samples, self.n_heads, self.d_k).transpose(1, 2)

    def fold(self, t):
        """tensor t starts out with shape [batch_size, n_heads, n_samples, d_k],
        ends up with [batch_size, n_samples, d_model]
        """
        batch_size, _, n_samples, _ = t.shape
        # return t.transpose(2, 1).view(batch_size, n_samples, self.d_model)

        # TODO: investigate if we really need reshape instead of view
        return t.transpose(2, 1).reshape(batch_size, n_samples, self.d_model)

class LayerNorm(nn.Module):
    def __init__(self, d_model, ε=1e-5):
        super().__init__()

        self.d_model = d_model
        self.ε = ε

        self.β = nn.Parameter(th.zeros(self.d_model))
        self.γ = nn.Parameter(th.ones(self.d_model))

    def forward(self, x):
        μ = th.mean(x, -1, keepdim=True)
        σσ = th.var(x, -1, keepdim=True, correction=0)
        return self.γ * (x - μ)/sqrt(σσ + self.ε) + self.β

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
    def __init__(self, d_model, d_heads, d_ff=None, mask=None):
        super().__init__()

        self.d_model = d_model
        self.d_heads = d_heads
        self.d_ff = d_ff
        self.mask = mask

        self.mha_zero = MultiHeadAttention(d_model=self.d_model, d_heads=self.d_heads)
        self.norm_zero = LayerNorm(d_model=self.d_model)

        self.ff_one = FeedForward(d_model=self.d_model, d_ff=self.d_ff)
        self.norm_one = LayerNorm(d_model=self.d_model)

    def forward(self, x):
        residual = x
        x = self.mha_zero(x, x, x, mask=self.mask)
        x += residual
        x = self.norm_zero(x)

        residual = x
        x = self.ff_one(x)
        x += residual
        x = self.norm_one(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_heads, d_ff=None, mask=None, mask_enc=None):
        super().__init__()

        self.d_model = d_model
        self.d_heads = d_heads
        self.d_ff = d_ff
        self.mask = mask
        self.mask_enc = mask_enc

        self.mha_zero = MultiHeadAttention(d_model=self.d_model, d_heads=self.d_heads)
        self.norm_zero = LayerNorm(d_model=self.d_model)

        self.mha_one = MultiHeadAttention(d_model=self.d_model, d_heads=self.d_heads)
        self.norm_one = LayerNorm(d_model=self.d_model)

        self.ff_two = FeedForward(d_model=self.d_model, d_ff=self.d_ff)
        self.norm_two = LayerNorm(d_model=self.d_model)

    def forward(self, x, out_enc):
        residual = x
        x = self.mha_zero(x, x, x, mask=self.mask)
        x += residual
        x = self.norm_zero(x)

        residual = x
        x = self.mha_one(x, out_enc, out_enc, mask=self.mask_enc)
        x += residual
        x = self.norm_one(x)

        residual = x
        x = self.ff_two(x)
        x += residual
        x = self.norm_two(x)

        return x
