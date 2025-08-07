import torch
import torch.nn as nn
from torch.nn import functional as F
import requests

#<-------------------- Parameters -------------------->
batch_size = 64
context_size = 256  # Number of characters used to predict at a time
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Run on GPU if available
eval_iters = 200
n_embd = 384  # Size of the embedding vector for each character
n_head = 6
n_layer = 6
dropout = 0.2  # Dropout rate for regularization

torch.manual_seed(1337)

#<-------------------- Data Preparation -------------------->
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map charatcters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #Encoder
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder

#<-------------------- Train/Validation Split -------------------->
data = torch.tensor(encode(response.text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% for training
train_data = data[:n]
val_data = data[n:]

#<-------------------- Data Loading -------------------->
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad() # For memory efficiency
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#<-------------------- Self-Attention -------------------->
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # Apply dropout to attention weights

        out = wei @ v  # (B, T, head_size)
        return out

#<-------------------- Multi-Head Self-Attention -------------------->
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # Projection layer to combine heads
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # Apply dropout and projection
        return out

#<-------------------- Feed Forward -------------------->   
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #expand for more complexity
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #compress back to original size
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)  # Apply the feed-forward network

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#<-------------------- Model Definition -------------------->
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Lookup table for tokens
        self.position_embedding_table = nn.Embedding(context_size, n_embd) # Lookup table for positions
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Linear layer for output logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Final layer normalization
        logits = self.lm_head(x)  # (B, T, vocab_size)
       
        if targets is None:
           loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
           
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:] # Helps ensure embedding table is not out of scope
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Eval loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initial context
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))