import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1111)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # FT: idx example (every single number will get it's corresponding logits): [
        #   [1, 2, 3, 4, 5, 1, 4, 10]
        #   [2, 4, 1, 3, 7, 6, 4, 21]
        #   [1, 12, 3, 14, 5, 1, 4, 11]
        #   [7, 2, 0, 3, 15, 51, 14, 4]
        # ]
        # FT: If we pass 0 (eg. 'A'), we will get logits (posibility scores for the next word) for the first row.
        logits = self.token_embedding_table(idx) # [B (batch=4), T (time=8), C (channels=vocab_size)]

        if targets is None:
            loss = None
        else:
            # FT: Because PyTorch expects [B, C, T]
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # FT: Cross entropy example:
            # logits = tensor([2.0, 1.0, 0.1])
            # target = tensor(0)
            # F.softmax(logits, dim=0)
            # Output: tensor([0.659, 0.242, 0.099])
            # loss = -log(0.659) ≈ 0.417
            loss = F.cross_entropy(logits, targets) # FT: Cross entropy = softmax + NLL

        return logits, loss # FT: Last layer before applying softmax (confidence scores, not probabilities yet)
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :] # last character's logits
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx