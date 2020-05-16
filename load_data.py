import os
import codecs
import random
import pandas as pd


class LoadData:
    def __init__(self, dataset_path, **data_params):
        self.word2int, self.int2word = self.__create_dicts(dataset_path)
        self.articles, self.titles = self.__load_dataset(dataset_path)

        self.test_ratio = data_params.get('test_ratio', 0.1)
        self.val_ratio = data_params.get('val_ratio', 0.1)
        self.shuffle = data_params.get('shuffle', True)
        self.data_dict, self.label_dict = self.__split_data()
        self.data_dict_num, self.label_dict_num = self.__split_data_num()

    def __split_data(self):
        dataset_length = len(self.articles)

        if self.shuffle:
            zipped = list(zip(self.articles, self.titles))
            random.shuffle(zipped)
            self.articles, self.titles = zip(*zipped)

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

    def __split_data_num(self):
        dataset_length = len(self.articles)

        if self.shuffle:
            zipped = list(zip(self.articles, self.titles))
            random.shuffle(zipped)
            self.articles, self.titles = zip(*zipped)

        test_count = int(dataset_length * self.test_ratio)
        val_count = int(dataset_length * self.val_ratio)

        data_dict = dict()
        data_dict['test'] = [self.word2int[i] for i in self.articles[:test_count]]
        data_dict['validation'] = [self.word2int[i] for i in self.articles[test_count:test_count + val_count]]
        data_dict['train'] = [self.word2int[i] for i in self.articles[test_count + val_count:]]

        label_dict = dict()
        label_dict['test'] = [self.word2int[i] for i in self.titles[:test_count]]
        label_dict['validation'] = [self.word2int[i] for i in self.titles[test_count:test_count + val_count]]
        label_dict['train'] = [self.word2int[i] for i in self.titles[test_count + val_count:]]

        return data_dict, label_dict


    @staticmethod
    def __load_dataset(dataset_path):
        """
        :param dataset_path: string
        :return: pd.DataFrame
        """
        dataset = pd.read_csv(os.path.join(dataset_path, 'dataset.csv'), sep='\t')

        return dataset['content'].tolist(), dataset['title'].tolist()

    @staticmethod
    def __create_dicts(dataset_path):
        with codecs.open(os.path.join(dataset_path, 'vocab.txt'), 'r', encoding='utf-8') as vocab_file:
            vocab = vocab_file.read().splitlines()

        word2int = {}
        for index in range(len(vocab)):
            word2int[vocab[index].split(' ', 2)[0]] = index+1

        word2int = {k: v for k, v in sorted(word2int.items(), key=lambda item: item[1])}
        int2word = {v: k for k, v in word2int.items()}

        return word2int, int2word


if __name__ == '__main__':
    data = LoadData(dataset_path='data')
    print()
