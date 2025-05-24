import torch
import torch.nn as nn
from torch.nn import functional as F
from classes.Head import Head
from classes.MultiHeadAttention import MultiHeadAttention
torch.manual_seed(1111)

n_embd = 32

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size) # Language modeling head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # FT: idx example (every single number will get it's corresponding logits): [
        #   [1, 2, 3, 4, 5, 1, 4, 10]
        #   [2, 4, 1, 3, 7, 6, 4, 21]
        #   [1, 12, 3, 14, 5, 1, 4, 11]
        #   [7, 2, 0, 3, 15, 51, 14, 4]
        # ]
        # FT: If we pass 0 (eg. 'A'), we will get logits (posibility scores for the next word) for the first row.

        tok_emb = self.token_embedding_table(idx) # [B (batch=4), T (time=8), C (channels)]
        pos_emb = self.position_embedding_table(torch.arange(T)) # [T, C]
        x = tok_emb + pos_emb
        x = self.sa_heads.forward(x) # Apply multiple heads of self-attention [B, T, C]
        logits = self.lm_head(x) # [B (batch=4), T (time=8), vocab_size]

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
            idx_cond = idx[:, -self.block_size:] # Pick the last block_size tokens to feed the model
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] # last character's logits
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) # randomly picks token indices based on the given probabilities (adds randomness to generation)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx