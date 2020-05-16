import os
import pickle
import numpy as np

from easybert import Bert

np.random.seed(42)


class Word2VecTransformer:
    def __init__(self):
        self.model_path = 'embedding/bert.pkl'
        self.vector_path = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
        self.vector_size = 768

        if os.path.isfile(self.model_path):
            print('Loading saved BERT model from pickle')
            model_file = open(self.model_path, 'rb')
            self.bert = pickle.load(model_file)
        else:
            self.model = Bert(self.vector_path)
            model_file = open(self.model_path, 'wb')
            pickle.dump(self.model, model_file)

        self.start_vec = np.random.rand(self.vector_size)
        self.end_vec = np.random.rand(self.vector_size)

    def transform(self, x):
        """
        returns bert vector of input word

        :param str x: input word
        :return:
        """
        if x == '<start>':
            r = self.start_vec
        elif x == '<end>':
            r = self.end_vec
        else:
            with self.bert:
                r = self.bert.embed(x)
        return r
