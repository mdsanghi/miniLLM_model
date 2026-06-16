import tiktoken
import torch
import torch.nn as nn
with open("./data/tiny_corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()
# print(corpus)
tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(corpus)
print(f"Total tokens: {len(tokens)}")
# print(tokens)
# print(tokenizer.decode(tokens))
# print(len(tokens))



# Model hyperparameters
vocab_size = tokenizer.n_vocab
d_model = 256
block_size = 64
batch_size = 4

data = torch.tensor(tokens, dtype=torch.long)


def get_batch():
    # Pick random starting positions for multiple training examples.
    ix = torch.randint(0, len(data) - block_size, (batch_size,))

    # x contains input tokens; y contains the next token for each position.
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


# These embedding layers will be learned during training.
token_embedding = nn.Embedding(vocab_size, d_model)
pos_embedding = nn.Embedding(block_size, d_model)

# Create one sample batch.
x, y = get_batch()

# ----- Token embedding block -----
# Convert each token ID in the batch into a vector.
token_embedded = token_embedding(x)

# ----- Positional embedding block -----
# Create position vectors for positions 0 to block_size - 1.
positions = torch.arange(block_size)
pos_embedded = pos_embedding(positions)

# ----- Final input embedding block -----
# Add token and positional embeddings before sending into Transformer blocks.
x_final = token_embedded + pos_embedded

print(f"Input batch x shape: {list(x.shape)}")
print(f"Target batch y shape: {list(y.shape)}")
print(f"Token embedding shape: {list(token_embedded.shape)}")
print(f"Positional embedding shape: {list(pos_embedded.shape)}")
print(f"Final Transformer input shape: {list(x_final.shape)}")
