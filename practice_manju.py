import tiktoken
import torch
import torch.nn as nn

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(b, seq_len, self.num_heads, self.head_dim)
        K = K.view(b, seq_len, self.num_heads, self.head_dim)
        V = V.view(b, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ V
        context = context.transpose(1, 2).contiguous()
        context = context.view(b, seq_len, self.d_model)
        output = self.out_proj(context)

        return output, attn_weights


# MAIN DATA PIPELINE EXECUTION

torch.manual_seed(123)

with open("./data/tiny_corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()
    print(corpus)
    print("Total characters in corpus :", len(corpus))
    chars = sorted(list(set(corpus)))
    print("Vocabulary Size:", len(chars))
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    token_ids = tokenizer.encode(corpus)
    print(token_ids)
    print(tokenizer.decode(token_ids))
    print("No of tokens:", len(token_ids))
 
    vocab_size = tokenizer.n_vocab 
    data = torch.tensor(token_ids, dtype=torch.long)
    print(f"Dataset Shape:", data.shape)
    
    # Token Embedding
    d_model = 256
    token_embedded = nn.Embedding(vocab_size, d_model)
    block_size = 64
    x_tokens = torch.tensor(token_ids[:block_size], dtype=torch.long)

    print(f"\nInput token IDs (Shape: {list(x_tokens.shape)}):")
    print(x_tokens)
    X_token_embedded = token_embedded(x_tokens)

    print(f"\nOutput embedded tensor shape: {list(X_token_embedded.shape)}")
    print("First token's vector snippet (first 5 elements):")
    print(X_token_embedded[0][:5].tolist())
    
    # Positional Embedding
    pos_embedding = nn.Embedding(block_size, d_model)
    positions = torch.arange(block_size)
    print(f"Position Indices: {positions.tolist()}")
    X_pos_embedded = pos_embedding(positions)
    print(f"Positional Embedding Shape: {list(X_pos_embedded.shape)}")
    X_final = X_token_embedded + X_pos_embedded
    print(f"Final Combined Embedding Shape: {list(X_final.shape)}")
    
  
    #Masked Multihead Attention

    num_heads = 4
    mha = MaskedMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    X_batch = X_final.unsqueeze(0)
    
    output, attention_weights = mha(X_batch)
    
    print("Attention Weights Tensor Shape:", list(attention_weights.shape))
    print("Output Tensor Shape:", list(output.shape))