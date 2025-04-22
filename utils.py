#region Read data

with open('../nano-gpt-messages.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) } # dict comperhension eg. A:0
itos = { i:ch for i,ch in enumerate(chars) } # dict comperhension eg. 0:A

def encode(chars):
    return [stoi[c] for c in chars]

def decode(chars):
    return ''.join([itos[i] for i in chars])

#endregion

#region Split data

import torch

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * (len(data)))
train_data = data[:n]
val_data = data[n:]

#endregion

