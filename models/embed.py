import os
import torch
import numpy as np
import torch.nn as nn

from transformers.word2vec import Word2VecTransformer


class Embedding(nn.Module):
    def __init__(self, int2word, embed_name='glove'):
        nn.Module.__init__(self)
        self.vector_path = 'embedding/embed_{}.npy'.format(embed_name)
        self.embed_name = embed_name

        self.int2word = int2word
        self.vocab_size = len(int2word)

        if os.path.isfile(self.vector_path):
            print('Loading saved embedding vectors')
            self.weights = np.load(self.vector_path, allow_pickle=True)
            self.weights = torch.from_numpy(self.weights)
        else:
            print('Creating embed tensor')
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
        transformer = Word2VecTransformer(self.embed_name)
        vectors = []
        for i in range(self.vocab_size):
            print("\r{:.2f}%".format(i * 100 / self.vocab_size), flush=True, end='')
            vec = transformer.transform(self.int2word[i])
            vectors.append(vec)
        embed = np.stack(vectors, axis=0)
        return torch.from_numpy(embed)


if __name__ == '__main__':
    import pickle

    vocab = pickle.load(open('data/vocab.pkl', 'rb'))
    word2int, int2word = vocab
    Embedding(int2word)

