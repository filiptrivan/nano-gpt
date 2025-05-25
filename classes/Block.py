import torch.nn as nn

from classes.FeedForward import FeedForward
from classes.MultiHeadAttention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, n_embd, block_size, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x