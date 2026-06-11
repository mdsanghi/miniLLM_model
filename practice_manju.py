import tiktoken
import torch
import torch.nn as nn
#Tokenization
with open("./data/tiny_corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()
    print(corpus)
    print("Total characters in corpus :",len(corpus))
    chars = sorted(list(set(corpus)))
    print("Vocabulary Size:",len(chars))
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    token_ids  = tokenizer.encode(corpus)
    print(token_ids)
    print(tokenizer.decode(token_ids))
    print("No of tokens:",len(token_ids))
 
    vocab_size = tokenizer.n_vocab 
    data = torch.tensor(token_ids, dtype=torch.long)
    print(f"Dataset Shape:",data.shape)
    
    
    # Token Embedding
    
    d_model=256
    token_embedded = nn.Embedding(vocab_size, d_model)
    block_size = 64
    x_tokens = torch.tensor(token_ids[:block_size], dtype=torch.long)

    print(f"\nInput token IDs (Shape: {list(x_tokens.shape)}):")
    print(x_tokens)
    X_token_embedded = token_embedded(x_tokens)

    print(f"\nOutput embedded tensor shape: {list(X_token_embedded.shape)}")
    print("First token's vector snippet (first 5 elements):")
    print(X_token_embedded[0][:5].tolist())
    
    #Positional Embedding
    pos_embedding = nn.Embedding(block_size, d_model)
    positions = torch.arange(block_size)
    print(f"Position Indices: {positions.tolist()}")
    X_pos_embedded = pos_embedding(positions)
    print(f"Positional Embedding Shape: {list(X_pos_embedded.shape)}")
    X_final = X_token_embedded + X_pos_embedded
    print(f"Final Combined Embedding Shape: {list(X_final.shape)}")