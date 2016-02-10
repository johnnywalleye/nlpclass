import collections
import math
import pandas as pd


class StupidBackoffLanguageModel:
    def __init__(self, corpus, alpha=0.05):
        """Initialize your data structures in the constructor."""
        self.alpha = alpha
        self.unique_words = set()
        self.total = 0
        self.bigram_counts = collections.defaultdict(lambda: 0)
        self.num_words = None
        self.unigram_counts = None
        self.train(corpus)
        self.create_bigram_counts_df()

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
            Compute any counts or other corpus statistics in this function.
        """
        for sentence in corpus.corpus:
            last_word = None
            for datum in sentence.data:
                word = datum.word
                self.unique_words.add(word)
                if last_word is not None:
                    token = ':'.join([last_word, word])
                    self.bigram_counts[token] += 1
                last_word = word
                self.total += 1

    def create_bigram_counts_df(self):
        unique = sorted(list(self.unique_words))
        df = pd.DataFrame(index=unique, columns=unique)
        df = df.fillna(0).astype(int)
        for k, v in self.bigram_counts.iteritems():
            words = k.split(':')
            df.loc[words[0], words[1]] = v
        self.bigram_counts_df = df
        self.num_words = len(df)
        self.unigram_counts = dict(df.sum(axis=1))

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
            sentence using your language model. Use whatever data you computed in train() here.
        """
        score = 0.0
        last_token = None
        for token in sentence:
            if last_token is not None:
                bigram_lookup_token = ':'.join([last_token, token])
                count = self.bigram_counts[bigram_lookup_token]
                if count > 0:
                    score += math.log(count + 1)
                    unigram_count = self.unigram_counts[last_token] if last_token in self.unigram_counts else 0
                    score -= math.log(unigram_count + self.num_words)
                else:
                    unigram_count = self.unigram_counts[token] if token in self.unigram_counts else 0
                    score += math.log(self.alpha * (unigram_count + 1))
                    score -= math.log(self.total + self.num_words)
            last_token = token
        return score
