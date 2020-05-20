import os
import pickle as pkl
import numpy as np
from gensim.models import KeyedVectors

np.random.seed(42)


class Word2VecTransformer:
    def __init__(self, embed_name='glove'):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.embed_name = embed_name
        if embed_name == 'glove':
            self.vector_path = os.path.join(project_dir, 'embedding/glove_dict.pkl')
            self.model = pkl.load(open(self.vector_path, 'rb'))
            self.vector_size = len(next(iter(self.model.values())))

        elif embed_name == 'bert':
            from easybert import Bert
            self.vector_path = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
            self.vector_size = 768

            print('Downloading BERT Model')
            self.model = Bert(self.vector_path)

        elif embed_name == 'fasttext':
            self.vector_path = os.path.join(project_dir, 'embedding/fasttext.pkl')
            self.model_path = os.path.join(project_dir, 'embedding/fasttext.vec')

            if os.path.isfile(self.vector_path):
                model_file = open(self.model_path, 'rb')
                self.model = pkl.load(model_file)
            else:
                print('Loading Fasttext model')
                self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False, unicode_errors='replace')
                model_file = open(self.model_path, 'wb')
                pkl.dump(self.model, model_file)

        self.special_tokens = ['<start>', '<end>', '<pad>', '<unk>']
        self.special_tok_dict = {}
        for token in self.special_tokens:
            self.special_tok_dict[token] = np.random.rand(self.vector_size)

    def transform(self, x):
        """
        returns bert vector of input word

        :param str x: input word
        :return:
        """
        if x in self.special_tokens:
            r = self.special_tok_dict[x]
        else:
            if self.embed_name == 'bert':
                with self.model:
                    r = self.model.embed(x)
            elif self.embed_name == 'glove':
                if x in self.model:
                    r = self.model[x]
                else:
                    r = np.random.rand(self.vector_size)
            elif self.embed_name == 'fasttext':
                if x in self.model.wv:
                    r = self.model.wv[x]
                else:
                    r = np.random.rand(self.vector_size)
        return r
