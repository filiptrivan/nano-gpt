import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1111)

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # The nn.Linear layer is applied independently to each 32-dimensional token vector inside the tensor x
        # Because the k, q and v are all produced by x, it's called self attention (not cross)
        k = self.key(x) #  (4, 8, 16) — a 16-dimensional key vector for each token.
        q = self.query(x) # (4, 8, 16) — a 16-dimensional query vector for each token.
        v = self.value(x) # (4, 8, 16) — a 16-dimensional query vector for each token.

        wei = k @ q.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # FT: As we use bigger and bigger decay for softmax, lets say * 8, softmax will sharpen, and make one hot encoding.
        wei = F.softmax(wei, dim=-1)

        out = wei @ v

        return out