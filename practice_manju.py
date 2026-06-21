import torch
import torch.nn as nn
import torch.optim as optim
import random
import tiktoken
import torch.nn.functional as F

# 1. Tokenization Setup
with open("./data/tiny_corpus.txt", "r", encoding="utf8") as f:
    text = f.read()

tokenizer = tiktoken.encoding_for_model("gpt-4o")
data = tokenizer.encode(text)

split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

block_size = 16

def get_batch(data, batch_size):
    indices = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = [data[i:i + block_size] for i in indices]
    y = [data[i + 1:i + block_size + 1] for i in indices]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# 2. Rotary Position Embedding (RoPE) Helper Functions
def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    # head_dim must be even
    assert head_dim % 2 == 0
    dim_idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (dim_idx / head_dim))
    
    # Generate positions framework
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    
    # Cache cos and sin values
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    # x shape: (b, num_heads, seq_len, head_dim)
    # cos, sin shape: (seq_len, head_dim // 2) -> reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(1) # (1, 1, seq_len, head_dim // 2)
    sin = sin.unsqueeze(0).unsqueeze(1) # (1, 1, seq_len, head_dim // 2)
    
    # Split x into even and odd components
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # Apply rotation matrix transformation
    x_rot1 = x1 * cos - x2 * sin
    x_rot2 = x1 * sin + x2 * cos
    
    # Recombine components back together
    x_out = torch.stack([x_rot1, x_rot2], dim=-1).flatten(-2)
    return x_out


# 3. Masked Multi-Head Attention with RoPE
class RoPEMaskedMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Max upper bound for generation steps context tracking
        max_positions = 2048
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(max_positions, max_positions), diagonal=1).bool(),
            persistent=False
        )
            
    def forward(self, x, cos, sin):
        b, seq_len, _ = x.shape

        Q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE embeddings right after projections to Q and K
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask_slice = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask_slice, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        context = (attn_weights @ V).transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        
        return self.out_proj(context)


# 4. Block and Core Architecture Adjustments
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = RoPEMaskedMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim) 

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin) 
        x = x + self.ffn(self.norm2(x)) 
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.head_dim = embed_dim // num_heads
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, position_ids=None):
        b, seq_len = x.shape
        x = self.token_embedding(x)
        
        # If no explicit position ids are given (e.g. training), assume default sequential indexing
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            
        # Dynamically calculate RoPE sine/cosine frequencies based on explicit position indexes passed
        # Fetching max value from position_ids to know bounds
        max_pos = position_ids.max().item() + 1
        cos_cached, sin_cached = precompute_theta_pos_frequencies(self.head_dim, int(max_pos), x.device)
        
        # Index into frequencies using given absolute position IDs
        cos = cos_cached[position_ids].squeeze(0) # Drops batch dimension helper mapping 
        sin = sin_cached[position_ids].squeeze(0)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        return self.lm_head(self.final_norm(x))


# 5. Initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(123)

model = MiniGPT(
    vocab_size=tokenizer.n_vocab, embed_dim=32, num_heads=4, hidden_dim=128, num_layers=4
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# 6. Training Loop Execution
model.train()
for step in range(1000): # Shortened iterations for demonstration convenience
    x_batch, y_batch = get_batch(train_data, batch_size=32)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    logits = model(x_batch) # position_ids default handled inside forward automatically
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), y_batch.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 7. Updated Generator supporting Absolute Tracker Constraints
def generate_with_rope(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x_gen = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_seq_len = x_gen.shape[1]
            
            # Slicing the input to follow your block window constraints
            start_idx = max(0, curr_seq_len - block_size)
            x_cond = x_gen[:, start_idx:]
            
            # CRITICAL FOR ROPE: Build explicit tracking vectors mapping back 
            # to their true original structural placement positions.
            position_ids = torch.arange(start_idx, curr_seq_len, device=device).unsqueeze(0)
            
            logits = model(x_cond, position_ids=position_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            x_gen = torch.cat([x_gen, next_token], dim=1)

    return tokenizer.decode(x_gen[0].tolist())

# Test Generation with RoPE
print(generate_with_rope(model, tokenizer, "The two men", max_new_tokens=20))