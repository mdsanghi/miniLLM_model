import tiktoken

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
    
    # Token Embedding
    
    