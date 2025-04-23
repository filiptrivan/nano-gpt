import torch

#region Hyperparams

torch.manual_seed(1111)
batch_size = 32 # How many independent sequences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?
learning_rate = 1e-3
training_epochs = 10000
eval_epochs = 200

#endregion

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

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * (len(data)))
train_data = data[:n]
val_data = data[n:]

#endregion

#region context/target building

# FT: Context length, length of the chunk of the text for the training, if we would use all text at once, it would be computationaly expensive
# FT: It's important to do because we want to train our network for small texts also (eg. one char), not just because of perf

# Making batches for context blocks, storing multiple blocks inside one tensor, because perf, making GPUs busy
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,)) # 4 element array
    # [4, 5, 1, 10]
    x = torch.stack([data[i:i + block_size] for i in ix]) # [data[4:12], data[5:13]...]
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]) # [data[5:13], data[6,14]...]
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            X, Y = get_batch(split)
            _, loss = model.forward(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# xb, yb = get_batch('train')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f'When input is {context.tolist()} the target is {target}')

#endregion

#region BigramLM

# Simplest biagram language model, working only on the 2 chars at the time, looking at one and trying to predict next
# Bigram is the (char1, char2) tuple
from classes.BigramLanguageModel import BigramLanguageModel

model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    if epoch % 1000 == 0:
        losses = estimate_loss()
        print(f"Step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() # Uses gradients to update the model, something like: param = param - learning_rate * param.grad

#endregion

#region Intuition test

def intuition_test(input: str):
    prompt_ids = encode(input)
    context = torch.tensor(
        [prompt_ids],
        dtype=torch.long
    )
    prediction = model.generate(context, 100)
    print(decode(prediction[0].tolist()))

#endregion