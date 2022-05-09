
import numpy as np
from bpemb import BPEmb

""" Subword Embeddings: https://nlp.h-its.org/bpemb """

class Text2Embed:
    def __init__(self):
        self.bpemb_en = BPEmb(lang="en", vs=100000, dim=300)

    def to_tokens(self, word):
        tokens = self.bpemb_en.encode(word)
        return tokens

    def to_embed(self, word, mean=True):
        embed = self.bpemb_en.embed(word)
        if mean == True and len(embed) > 1:
            embed = np.mean(embed, axis=0)
            embed = np.expand_dims(embed, axis=0)
        return embed

if __name__ == "__main__":
    # words = ["polyp", "instrument", "nuclei", "skin cancer", "neural structure"]
    words = ["small", "medium", "large"]
    embed = Text2Embed()

    embed_vec = []
    for word in words:
        tokens = embed.to_tokens(word)
        vec = embed.to_embed(word, mean=False)
        embed_vec.append(vec)

        print(f"Tokens: {tokens} - Vec: {vec.shape}")

    embed_vec = np.array(embed_vec)
    print(embed_vec.shape)
