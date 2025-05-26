import torch.nn as nn

# Computation part of the transformer
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # If a value is positive, keep it. If it’s negative, make it zero.
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )
    
    def forward(self, idx):
        return self.net(idx)
