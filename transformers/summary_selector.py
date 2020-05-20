import math


class SummarySelector:
    def __init__(self):
        self.start = 0
        self.end = 1

    def transform(self, summaries):

        vocab = dict()
        vocab['<start>'] = self.start
        vocab['<end>'] = self.end
        count = 2
        for ind in range(len(summaries)):
            for row in summaries[ind]:
                for word in row:
                    if word not in vocab:
                        vocab[word] = count
                        count += 1
        idf = dict()

        total_word_count = 0
        total_document_count = len(summaries[0])

        candidate_corpus = []

        for num in range(len(summaries[0])):
            temp = summaries[0][num] + summaries[1][num] + summaries[2][num]
            candidate_corpus.append(temp)

        for word in list(vocab.keys()):
            belonging_doc = 0

            for doc in candidate_corpus:
                if word in doc:
                    belonging_doc += 1

            if belonging_doc == 0:
                continue

            idf[word] = math.log(total_document_count / belonging_doc)

        for candidate in candidate_corpus:
            total_word_count += len(candidate) - 6  # start end x3

        print(total_word_count)

        chosen_summaries = []
        for num in range(len(summaries[0])):
            scores = []
            candidates = []
            candidates.append(summaries[0][num][:])
            candidates.append(summaries[1][num][:])
            candidates.append(summaries[2][num][:])

            for candidate in candidates:
                score = 0
                for word in candidate:
                    if word == '<start>' or word == '<end>':
                        continue
                    score += (candidate_corpus[num].count(word) / total_word_count) * idf[word]

                scores.append(score / (len(candidate) - 2))

            best_index = scores.index(max(scores))

            chosen_summaries.append(candidates[best_index])

        return chosen_summaries






