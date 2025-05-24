import torch;
import torch.nn as nn
from classes.Head import Head

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])

    def forward(self, idx):
        return torch.cat([h(idx) for h in self.heads], dim=-1)
    