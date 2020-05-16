import os, codecs, re, pickle
import pandas as pd
from collections import Counter

filepaths = ['./raw/articles1.csv', './raw/articles2.csv', './raw/articles3.csv']

punctuation_marks = ['!', '?', ',', ';', '.', ':', '…']

separators = ['-lrb-', '-rrb-', '\"', '\'', '’', '‘', '`', '“', '”', '^', '#', '+', '$', '%', '&', '/', '(', ')', '{',
              '}', '[', ']', '=', '*', '\\', '—', '-', '_', '|']


def clean(content):
    # Cast all characters to lowercase
    content = content.lower()

    # Remove punctuation marks
    for c in punctuation_marks:
        content = content.replace(c, '')

    # Replace separator characters with spaces
    for c in separators:
        content = content.replace(c, ' ')

    content = ' '.join(content.split())

    return content


def getvocab(content):
    # Cast all characters to lowercase
    content = clean(content)

    ## Create vocabulary
    # Count number of types
    vocab = Counter(content.split())
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    vocab_str = ''
    for pair in vocab:
        vocab_str = vocab_str + pair[0] + ' ' + str(pair[1]) + '\n'

    return vocab_str


def main():
    df_list = []

    for file in filepaths:
        df = pd.read_csv(file)
        df_list.append(df)

    articles = pd.concat(df_list, ignore_index=True)
    articles_to_save = pd.DataFrame()

    articles_to_save['content'] = [clean(str(i)) for i in articles['content']]
    articles_to_save['title'] = ['<start> ' + clean(str(i)) + ' <end>' for i in articles['title']]

    articles_to_save.to_csv('./dataset.csv', encoding='utf-8', sep='\t', index=False)

    content_for_vocab = clean(' '.join(articles_to_save['content']) + ' '.join(articles_to_save['title']))

    vocab_str = getvocab(content_for_vocab)
    with codecs.open('./vocab.txt', 'w', encoding='utf8') as vocab_out_file:
        vocab_out_file.write(vocab_str)


if __name__ == '__main__':
    main()
