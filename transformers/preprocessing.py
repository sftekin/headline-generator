import re
import emoji
import string
import nltk

from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words('english'))


class Preprocess:
    def __init__(self, content_len=None, title_len=None):
        self.content_len = content_len
        self.title_len = title_len
        self.tokenizer = TweetTokenizer()

        # remove url, long-mention, mention, hash-tag, numbers
        self.re_list = [r'http\S+', r'\(@ ?[^\s].+\)', r'@ ?[^\s]+',
                        r'# ?[^\s]+', '[{}]'.format(re.escape(string.punctuation))]

        # replace '- The New York Times'
        self.re_title = r'\-[a-z A-Z]+'

        # before tokenizing, remove non textual emoji
        self.rmv_emoji = lambda x: emoji.get_emoji_regexp().sub(r'', x)

        # after tokenizing
        self.rmv_stop = lambda x: [w for w in x if w not in eng_stopwords]
        self.rmv_pun = lambda x: [w for w in x if w not in string.punctuation]
        self.rmv_short_long = lambda x: [w for w in x if 20 >= len(w) >= 3]

        self.rmv_list = [self.rmv_stop, self.rmv_pun, self.rmv_short_long]

    def transform(self, X, y):
        """
        X is the content
        y is the title

        :param X: [str, str, ..., str]
        :param y: [str, str, ..., str]
        :return:[[token , ..., token], ...], [[token , ..., token], ...]
        """
        labels = []
        clean_data = []
        for count, (content, title) in enumerate(zip(X, y)):
            print('\r{:.2f}%'.format(count * 100 / len(X)), flush=True, end='')
            token_content = self._preprocess(str(content).lower(), mode='content')
            title_content = self._preprocess(str(title).lower(), mode='title')
            clean_data.append(token_content)
            labels.append(title_content)

        return clean_data, labels

    def _preprocess(self, sentence, mode='content'):
        """
        :param sentence: string
        :return: [token, ..., token]
        """
        if mode == 'title':
            self.re_list.insert(0, self.re_title)

        # remove url, long-mention, mention, hash-tag, non-textual emoji, punctuation
        for re_op in self.re_list:
            sentence = re.sub(re_op, '', sentence)
        sentence = self.rmv_emoji(sentence)

        # tokenize the sentence
        tokens = self.tokenizer.tokenize(sentence)
        for rmv_op in self.rmv_list:
            tokens = rmv_op(tokens)

        # lower every word
        tokens = [token.lower() for token in tokens]

        # trim tokens
        if self.content_len and self.title_len:
            if mode == 'title':
                tokens = tokens[:self.title_len+1]
            else:
                tokens = tokens[:self.content_len+1]

        # add start end
        tokens = ['<start>'] + tokens + ['<end>']

        return tokens
