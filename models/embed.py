import os
import torch
import numpy as np
import torch.nn as nn

from transformers.word2vec import Word2VecTransformer


class Embedding(nn.Module):
    def __init__(self, int2word):
        nn.Module.__init__(self)
        self.vector_path = 'embedding/embed.npy'

        self.int2word = int2word
        self.vocab_size = len(int2word)

        if os.path.isfile(self.vector_path):
            print('Loading saved embedding vectors')
            self.weights = np.load(self.vector_path, allow_pickle=True)
        else:
            self.weights = self.create_embed_tensor()
            np.save(self.vector_path, self.weights)

        self.embed_dim = self.weights.shape[1]
        self.embedding = nn.Embedding.from_pretrained(self.weights)

    def forward(self, captions):
        """
        :param captions: (B, S)
        :return: vectors: (B, S, Vector_dim)
        """
        vectors = self.embedding(captions)
        return vectors

    def create_embed_tensor(self):
        transformer = Word2VecTransformer()
        vectors = []
        for i in range(self.vocab_size):
            vec = transformer.transform(self.int2word[i])
            vectors.append(vec)
        embed = np.stack(vectors, axis=0)
        return torch.from_numpy(embed)
