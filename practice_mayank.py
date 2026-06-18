import torch
import torch.nn as nn
import torch.optim as optim
import random
import tiktoken
import torch.nn.functional as F

with open("./data/tiny_corpus.txt", "r", encoding="utf8") as f:
    text = f.read()

print(text[:150])
print("Total characters:", len(text))

tokenizer = tiktoken.encoding_for_model("gpt-4o")

data = tokenizer.encode(text)

split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

print("Train tokens:", len(train_data))
print("Validation tokens:", len(val_data))


block_size = 4
def get_batch(data, batch_size):
    indices = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = [data[i:i + block_size] for i in indices]
    y = [data[i + 1:i + block_size + 1] for i in indices]

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    return x, y

x, y = get_batch(train_data, batch_size=4)
print("Input:", x)
print("Target:", y)

class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Final projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):

        b, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim)
        K = K.view(b, seq_len, self.num_heads, self.head_dim)
        V = V.view(b, seq_len, self.num_heads, self.head_dim)

        # Move heads before seq_len (b, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float('-inf'))

        # Attention probabilities
        attn_weights = torch.softmax(scores, dim=-1)

        # Context vectors
        context = attn_weights @ V

        # Combine heads
        # (b, num_heads, seq_len, head_dim)
        # ->
        # (b, seq_len, num_heads, head_dim)

        context = context.transpose(1, 2).contiguous()

        # Merge heads
        context = context.view(b, seq_len, self.d_model)

        # Final projection
        output = self.out_proj(context)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), # Linear layer
            nn.GELU(),  # Activation function Modern GPTs use GELU
            nn.Linear(hidden_dim, d_model), # Second linear layer
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        self.attention = MaskedMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.ffn = FeedForward(embed_dim, hidden_dim) 
        self.norm2 = nn.LayerNorm(embed_dim) 

    def forward(self, x):

        norm_x = self.norm1(x) # Layer normalization
        attention_output = self.attention(norm_x)  #  Multi-head causal self attention 
        x = x + attention_output # Residual connections
        norm_x = self.norm2(x) # Layer normalization
        ffn_output = self.ffn(norm_x) # Feedforward neural network 
        x = x + ffn_output # Residual connections
    
        return x
    
class MiniGPT(nn.Module):

    def __init__(self, vocab_size, block_size, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        
        b, seq_len = x.shape
        token_embeddings = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
    
torch.manual_seed(123)

model = MiniGPT(
    vocab_size=tokenizer.n_vocab,
    block_size=block_size,
    embed_dim=32,
    num_heads=4,
    hidden_dim=128,
    num_layers=2
)

print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

learning_rate = 3e-4
batch_size = 32
max_steps = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()

for step in range(max_steps):

    # Get batch
    x, y = get_batch(train_data, batch_size=batch_size)
    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    # Reshape for cross entropy
    B, T, C = logits.shape

    loss = F.cross_entropy(
        logits.view(B * T, C),
        y.view(B * T)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

def generate(model, tokenizer, prompt, max_new_tokens=50):

    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device)
    x = x.unsqueeze(0)

    with torch.no_grad():

        for _ in range(max_new_tokens):

            x_cond = x[:, -block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)

    output_tokens = x[0].tolist()
    generated_text = tokenizer.decode(output_tokens)

    return generated_text

prompt = "The two men"

generated = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=15
)

print(generated)