import math


class SummarySelector:
    def __init__(self, word2int, int2word):
        self.word2int = word2int
        self.int2word = int2word

    def transform(self, summaries, y):
        idf = dict()

        total_word_count = 0
        total_document_count = len(summaries[0])

        candidate_corpus = []

        for num in range(len(summaries[0])):
            temp = summaries[0][num] + summaries[1][num] + summaries[2][num]
            candidate_corpus.append(temp)

        for word_int in list(self.int2word.keys()):
            belonging_doc = 0

            for doc in candidate_corpus:
                if word_int in doc:
                    belonging_doc += 1

            if belonging_doc == 0:
                continue

            idf[word_int] = math.log(total_document_count / belonging_doc)

        for candidate in candidate_corpus:
            total_word_count += len(candidate) - 6  # start end x3

        chosen_summaries = []
        for num in range(len(summaries[0])):
            scores = []
            candidates = []
            candidates.append(summaries[0][num][:])
            candidates.append(summaries[1][num][:])
            candidates.append(summaries[2][num][:])

            for candidate in candidates:
                score = 0
                for word_int in candidate:
                    if word_int == 1 or word_int == 2:
                        continue
                    score += (candidate_corpus[num].count(word_int) / total_word_count) * idf[word_int]

                scores.append(score / (len(candidate) - 2))

            best_index = scores.index(max(scores))

            chosen_summaries.append(candidates[best_index])

        return chosen_summaries, y[0]
