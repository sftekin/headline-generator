import os
import random
import pandas as pd
import pickle as pkl

from collections import Counter
from transformers.preprocessing import Preprocess
from transformers.summarizer import Summarizer
from transformers.summary_selector import SummarySelector


class LoadData:
    def __init__(self, dataset_path, **data_params):
        self.summary_path = os.path.join(dataset_path, 'summary_set.pkl')
        self.vocab_path = os.path.join(dataset_path, 'vocab.pkl')
        self.raw_data_path = os.path.join(dataset_path, 'all-the-news')

        self.content_len = data_params.get('content_len', 50)
        self.title_len = data_params.get('title_len', 15)
        self.num_samples = data_params.get('num_samples', -1)
        self.sentence_num = data_params.get('num_sentence', 3)
        self.unk_threshold = data_params.get('unk_threshold', 10)

        self.test_ratio = data_params.get('test_ratio', 0.1)
        self.val_ratio = data_params.get('val_ratio', 0.1)
        self.shuffle = data_params.get('shuffle', True)

        if not os.path.isfile(self.vocab_path) or not os.path.isfile(self.summary_path):
            contents, titles = self.__load_from_csv()
            print('\nCreating data ...')
            self.summaries, self.titles, self.word2int, self.int2word = self.__create_summary_set(contents, titles)

            all_sum_save_path = os.path.join(dataset_path, 'all_sum_set.pkl')
            with open(all_sum_save_path, 'wb') as f:
                pkl.dump([self.summaries, self.titles], f)

            # select one of the summaries from candidates
            print('\nSelecting summaries ...')
            selector = SummarySelector(self.word2int, self.int2word)
            self.summaries, self.titles = selector.transform(self.summaries, self.titles)

            with open(self.summary_path, 'wb') as f:
                pkl.dump([self.summaries, self.titles], f)

        else:
            print('\nLoading data from pickle ...')
            with open(self.vocab_path, 'rb') as f:
                self.word2int, self.int2word = pkl.load(f)
            with open(self.summary_path, 'rb') as f:
                self.summaries, self.titles = pkl.load(f)

        # split test train validation
        self.data_dict, self.label_dict = self.__split_data()

    def __load_from_csv(self):
        # read all articles 1, 2 and 3
        file_paths = os.listdir(self.raw_data_path)
        file_paths = [os.path.join(self.raw_data_path, file_name) for file_name in file_paths]

        df_list = []
        for file in file_paths:
            df = pd.read_csv(file, index_col=0)
            df_list.append(df)
        articles = pd.concat(df_list, ignore_index=True)

        contents = articles['content'].values[:self.num_samples]
        titles = articles['title'].values[:self.num_samples]

        return contents, titles

    def __create_summary_set(self, in_contents, in_titles):
        process_obj = Preprocess(content_len=self.content_len, title_len=self.title_len)

        print("Creating summaries ...")
        summarizer = Summarizer(num_sentence=self.sentence_num)
        sum_collection, sum_titles = summarizer.transform(X=in_contents, y=in_titles)

        summary_label = []
        summary_content = []
        for i in range(summarizer.num_summarizers):
            titles = []
            contents = []
            for sum_list, title in zip(sum_collection, sum_titles):
                contents.append(sum_list[i])
                titles.append(title)

            print('\nPreprocessing summary: {}'.format(i))
            contents, titles = process_obj.transform(X=contents, y=titles)

            summary_content.append(contents)
            summary_label.append(titles)

        # create vocab
        word2int, int2word = self.__create_vocab(summary_content, summary_label)

        sum_label_int = []
        sum_con_int = []
        for i in range(summarizer.num_summarizers):
            content_int = [[word2int[word] for word in content] for content in summary_content[i]]
            title_int = [[word2int[word] for word in title] for title in summary_label[i]]
            sum_con_int.append(content_int)
            sum_label_int.append(title_int)

        with open(self.vocab_path, 'wb') as f:
            pkl.dump([word2int, int2word], f)

        return sum_con_int, sum_label_int, word2int, int2word

    def __create_vocab(self, contents, titles):
        # create_vocab
        all_words = []
        for i in range(len(contents)):
            for c, t in zip(contents[i], titles[i]):
                all_words += c + t

        self.__rmv_less_frequent(all_words, contents, titles)

        vocab = Counter(all_words)
        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda x: x[1], reverse=True)}

        word2int = {k: v for v, k in enumerate(vocab.keys(), 1)}
        word2int['<unk>'] = 0
        int2word = {v: k for k, v in word2int.items()}

        return word2int, int2word

    def __rmv_less_frequent(self, all_words, contents, titles):
        vocab = Counter(all_words)
        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda x: x[1], reverse=True)}

        # replace the less frequent words with '<unk>'
        for i in range(len(contents)):
            for content, title in zip(contents[i], titles[i]):
                for k in range(len(content)):
                    word = content[k]
                    if vocab[word] <= self.unk_threshold:
                        content[k] = '<unk>'
                for k in range(len(title)):
                    word = title[k]
                    if vocab[word] <= self.unk_threshold:
                        title[k] = '<unk>'

    def __split_data(self):
        dataset_length = len(self.summaries)

        if self.shuffle:
            zipped = list(zip(self.summaries, self.titles))
            random.shuffle(zipped)
            self.summaries, self.titles = zip(*zipped)
            self.summaries = list(self.summaries)
            self.titles = list(self.titles)

        test_count = int(dataset_length * self.test_ratio)
        val_count = int(dataset_length * self.val_ratio)

        data_dict = dict()
        data_dict['test'] = self.summaries[:test_count]
        data_dict['validation'] = self.summaries[test_count:test_count + val_count]
        data_dict['train'] = self.summaries[test_count + val_count:]

        label_dict = dict()
        label_dict['test'] = self.titles[:test_count]
        label_dict['validation'] = self.titles[test_count:test_count + val_count]
        label_dict['train'] = self.titles[test_count + val_count:]

        return data_dict, label_dict


if __name__ == '__main__':
    data_params = {
        "content_len": 100,
        "title_len": 15,
        "num_samples": 100,
        "num_sentence": 3,
        "test_ratio": 0.1,
        "val_ratio": 0.1,
        "shuffle": True,
        "unk_threshold": 10
    }
    data = LoadData(dataset_path='data', **data_params)
    print()
