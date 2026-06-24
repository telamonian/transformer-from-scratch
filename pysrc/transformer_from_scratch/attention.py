from math import sqrt
import torch as th
from torch import nn, Tensor

def softmax(x: Tensor, dim: int, dtype: th.dtype | None = None) -> Tensor:
    """NOTE: all math must be out-of-place. Using th.exp(..., out=out) / in-place /= breaks
    autograd, which matters because softmax is fed grad-requiring scores during a forward pass.
    """
    dtype = th.float if dtype is None else dtype

    # x_max.shape is the same as x.shape, except that x.max_shape[dim] == 1.
    # Each index of x_max holds the maximum value along the collapsed dim holding the other indices constant.
    # We use a per-row max, instead of a global max like in many descriptions of softmax.
    # This avoids a 0/0 == NaN that can arise one of the "rows" is all padding (-1e9)
    x_max = th.max(x, dim=dim, keepdim=True).values

    # we subtract x_max in order to keep the exponents small, which helps with numerical stability.
    # Subtracting x_max balances out in the division step, since you can rewrite:
    # {e^{x - x_m}}/{\sum_i{e^{x_i - x_m}}} -> {e^{x}e^{-x_m}}}/{e^{-x_m}\sum_i{e^{x_i}}}}
    out = th.exp((x - x_max).to(dtype))

    return out / out.sum(dim=dim, keepdim=True)

def attentionSDP(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """scaled dot-product attention.
    Expects q, k, v to be in the form [batch_size, n_heads, n_samples, d_k]
    """
    _, _, _, d_k = keys.shape

    weights = queries.matmul(keys.transpose(-2, -1))/sqrt(d_k)

    if mask is not None:
        weights = weights.masked_fill(mask == 0, -1e9)

    return softmax(weights, -1).matmul(values), weights

def attentionSDP_oneline(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    return softmax(queries.matmul(keys.transpose(-2, -1))/sqrt(keys.shape[-1]), -1).matmul(values)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_k: int = d_model // n_heads

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> Tensor:
        queries = self.unfold(self.q_linear(queries))
        keys = self.unfold(self.k_linear(keys))
        values = self.unfold(self.v_linear(values))

        values, _ = attentionSDP(queries, keys, values, mask=mask)

        return self.out_linear(self.fold(values))

    def unfold(self, t: Tensor) -> Tensor:
        """tensor t starts out with shape [batch_size, n_samples, d_model],
        ends up with [batch_size, n_heads, n_samples, d_k]
        """
        batch_size, n_samples, _ = t.shape
        return t.view(batch_size, n_samples, self.n_heads, self.d_k).transpose(1, 2)

    def fold(self, t: Tensor) -> Tensor:
        """tensor t starts out with shape [batch_size, n_heads, n_samples, d_k],
        ends up with [batch_size, n_samples, d_model]
        """
        batch_size, _, n_samples, _ = t.shape
        # return t.transpose(2, 1).view(batch_size, n_samples, self.d_model)

        # TODO: investigate if we really need reshape instead of view
        return t.transpose(2, 1).reshape(batch_size, n_samples, self.d_model)
