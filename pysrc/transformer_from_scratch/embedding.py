import math
from math import sqrt
import torch as th
import torch.nn as nn

def xavierInit(*size):
    """see github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/torch/nn/init.py#L479,
    Massive Exploration of Neural Machine Translation Architectures: arxiv.org/pdf/1703.03906
    """

    def getFanInFanOut(*size):
        if len(size) < 2:
            raise ValueError("Cannot calculate fan_in/_out for less then 2 dimensions")

        fieldSize = math.prod(size[2:])
        return size[1]*fieldSize, size[0]*fieldSize

    fanIn, fanOut = getFanInFanOut(*size)
    a = sqrt(6/(fanIn + fanOut))
    return th.empty(*size).uniform_(-a, a)

class Embedding(nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model

        self.weight = nn.Parameter(xavierInit(self.n_vocab, self.d_model))

    def forward(self, x):
        """From AIAYN: "In the embedding layers, we multiply those weights by sqrt(d_model)"
        """
        # when indexing a tensor with another tensor, each tensor element is treated as an individual index.
        # The results are then batched according to the dimensions of the indexing tensor
        # NOTE moved scalar multiplication to Transformer.forward
        return self.weight[x] #*self.d_model_sqrt

class PositionalEncoder(nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()

        self.n_vocab = n_vocab
        self.d_model = d_model

        pe = th.zeros(n_vocab, d_model)
        pos = th.arange(n_vocab).view(-1, 1)
        arg = pos/(10000 ** (th.arange(0, d_model, 2)/d_model))

        pe[:, ::2] = th.sin(arg)
        pe[:, 1::2] = th.cos(arg)

        # we register pe as a buffer since it is a fixed lookup table.
        # Registration implies that it is part of the model's runtime state but is not a trainable tensor.
        # In practical terms, registration allows pe to be part of the model's saved/loaded state,
        # and also tells pytorch to move pe to a guest device (eg gpu) along with the rest of the model when appropriate
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x is expected to have shape [batch_size, n_sample, d_model]
        a chunk of pe with shape [1, n_sample, d_model] is added to x
        """
        return x + self.pe[:, :x.shape[1]]
