from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer


class Summarizer:
    def __init__(self, num_sentence, trim_len=5000):
        self.num_sentence = num_sentence
        self.trim_len = trim_len
        self.tokenizer = Tokenizer('english')

        self.summarizers = [LexRankSummarizer(), LsaSummarizer(), SumBasicSummarizer()]
        self.num_summarizers = len(self.summarizers)

    def transform(self, X, y):
        """
        Create 5 summaries for each content. If content has less than
        self.num_sentence it is not added to collection

        :param X: [str, str, ...]
        :param y: [str, str, ...]
        :return: X=[[sum1, sum2, ..., sum5], ...], y=[str, str, ...]
        """
        labels = []
        sum_collection = []
        for count, (content, title) in enumerate(zip(X, y)):
            print('\r{:.2f}%'.format(count * 100 / len(X)), flush=True, end='')

            trimed_content = content[:self.trim_len]

            parser = PlaintextParser.from_string(trimed_content, self.tokenizer)

            sum_list = []
            long_content = True
            for summarizer in self.summarizers:
                summary = summarizer(parser.document, self.num_sentence)
                if len(summary) >= self.num_sentence:
                    sum_txt = ' '.join(sentence._text for sentence in summary)
                    sum_list.append(sum_txt)
                else:
                    long_content = False

            if long_content:
                labels.append(y)
                sum_collection.append(sum_list)

        return sum_collection, labels
