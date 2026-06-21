from math import sqrt
import torch as th
from torch import nn, Tensor

def softmax(x: Tensor, dim, dtype=None):
    out = th.empty(*x.shape, dtype=th.float if dtype is None else dtype)

    # x_max is a single element tensor
    x_max = th.max(x)

    # subtracting x_max balances out in the division step
    th.exp(x - x_max, out=out)

    out /= out.sum(dim=dim, keepdim=True)
    # th.div(out, out.sum(dim=dim, keepdim=True), out=out)

    return out

def attentionSDP(queries, keys, values, mask=None):
    """scaled dot-product attention.
    Expects q, k, v to be in the form [batch_size, n_heads, n_samples, d_k]
    """
    _, _, _, d_k = keys.shape

    weights = queries.matmul(keys.transpose(-2, -1))/sqrt(d_k)

    if mask is not None:
        weights = weights.masked_fill(mask == 0, -1e9)

    return softmax(weights, -1).matmul(values), weights

def attentionSDP_oneline(queries, keys, values):
    return softmax(queries.matmul(keys.transpose(-2, -1))/sqrt(keys.shape[3]), -1).matmul(values)

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
