import os
import random
import pandas as pd
import pickle as pkl

from collections import Counter
from transformers.preprocessing import Preprocess


class LoadData:
    def __init__(self, dataset_path, **data_params):
        self.dataset_path = os.path.join(dataset_path, 'dataset.pkl')
        self.vocab_path = os.path.join(dataset_path, 'vocab.pkl')
        self.raw_data_path = os.path.join(dataset_path, 'all-the-news')

        self.content_len = data_params.get('content_len', 200)
        self.title_len = data_params.get('title_len', 15)
        self.test_ratio = data_params.get('test_ratio', 0.1)
        self.val_ratio = data_params.get('val_ratio', 0.1)
        self.shuffle = data_params.get('shuffle', True)

        if not os.path.isfile(self.dataset_path) or \
                not os.path.isfile(self.vocab_path):
            print('Creating data from the raw path: {}'.format(self.raw_data_path))
            self.articles, self.titles, self.word2int, self.int2word = self.__create_data()
        else:
            print('Loading data from pickle ...')
            self.articles, self.titles, self.word2int, self.int2word = self.__load_pickle()

        # split test train validation
        self.data_dict, self.label_dict = self.__split_data()

    def __create_data(self):
        # read all articles 1, 2 and 3
        file_paths = os.listdir(self.raw_data_path)
        file_paths = [os.path.join(self.raw_data_path, file_name) for file_name in file_paths]

        df_list = []
        for file in file_paths:
            df = pd.read_csv(file, index_col=0)
            df_list.append(df)
        articles = pd.concat(df_list, ignore_index=True)

        contents = articles['content'].values
        titles = articles['title'].values

        # tokenize the title and content
        process_obj = Preprocess(content_len=self.content_len,
                                 title_len=self.title_len)
        X, y = process_obj.transform(X=contents, y=titles)

        # create_vocab
        all_words = []
        for c, t in zip(X, y):
            all_words += c + t

        vocab = Counter(all_words)
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

        word2int = {k: v for k, v in vocab}
        int2word = {v: k for k, v in word2int.items()}

        # convert data to numbers
        X_int, y_int = [], []
        for content, title in zip(X, y):
            X_int.append([word2int[word] for word in content])
            y_int.append([word2int[word] for word in title])

        # save data
        with open(self.dataset_path, 'wb') as f:
            pkl.dump([X_int, y_int], f)

        with open(self.vocab_path, 'wb') as f:
            pkl.dump([word2int, int2word], f)

        return X_int, y_int, word2int, int2word

    def __split_data(self):
        dataset_length = len(self.articles)

        if self.shuffle:
            zipped = list(zip(self.articles, self.titles))
            random.shuffle(zipped)
            self.articles, self.titles = zip(*zipped)
            self.articles = list(self.articles)
            self.titles = list(self.titles)

        test_count = int(dataset_length * self.test_ratio)
        val_count = int(dataset_length * self.val_ratio)

        data_dict = dict()
        data_dict['test'] = self.articles[:test_count]
        data_dict['validation'] = self.articles[test_count:test_count + val_count]
        data_dict['train'] = self.articles[test_count + val_count:]

        label_dict = dict()
        label_dict['test'] = self.titles[:test_count]
        label_dict['validation'] = self.titles[test_count:test_count + val_count]
        label_dict['train'] = self.titles[test_count + val_count:]

        return data_dict, label_dict

    def __load_pickle(self):
        with open(self.dataset_path, 'rb') as f:
            X, y = pkl.load(f)
        with open(self.vocab_path, 'rb') as f:
            word2int, int2word = pkl.load(f)

        return X, y, word2int, int2word


if __name__ == '__main__':
    data = LoadData(dataset_path='data')
    print()
