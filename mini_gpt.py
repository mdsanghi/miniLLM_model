import tiktoken


with open("./data/tiny_corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()
# print(corpus)

token = tiktoken.encoding_for_model("text-embedding-3-small")

tokens  = token.encode(corpus)
print(tokens)
# print(token.decode(tokens))