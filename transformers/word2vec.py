import os
import pickle as pkl
import numpy as np

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
        return r
