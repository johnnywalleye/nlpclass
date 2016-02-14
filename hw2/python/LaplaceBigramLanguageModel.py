import collections
import math
import pandas as pd


class LaplaceBigramLanguageModel:
    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.bigram_counts = collections.defaultdict(lambda: 0)
        self.unique_words_set = set()
        self.unique_words_count = None
        self.bigram_counts_by_first_word = None
        self.train(corpus)
        self.create_bigram_counts_by_first_word()

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            last_word = None
            for datum in sentence.data:
                word = datum.word
                self.unique_words_set.add(word)
                if last_word is not None:
                    token = ':'.join([last_word, word])  # in HolbrookCorpus.py we remove all ':' chars
                    self.bigram_counts[token] += 1
                last_word = word

    def create_bigram_counts_by_first_word(self):
        unique = sorted(list(self.unique_words_set))
        df = pd.DataFrame(index=unique, columns=unique)
        df = df.fillna(0).astype(int)
        for k, v in self.bigram_counts.iteritems():
            words = k.split(':')
            df.loc[words[0], words[1]] = v
        self.unique_words_count = len(self.unique_words_set)
        self.bigram_counts_by_first_word = dict(df.sum(axis=1))

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        last_token = None
        for token in sentence:
            if last_token is not None:
                bigram_lookup_token = ':'.join([last_token, token])
                bigram_count = self.bigram_counts[bigram_lookup_token]
                score += math.log(bigram_count + 1)
                bigram_count_first_word = self.bigram_counts_by_first_word[last_token] if last_token in self.bigram_counts_by_first_word else 0
                score -= math.log(bigram_count_first_word + self.unique_words_count)
            last_token = token
        return score
