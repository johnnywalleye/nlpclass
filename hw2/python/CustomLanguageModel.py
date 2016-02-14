import collections
import math
import pandas as pd

class CustomLanguageModel:
    """
    Interpolated Kneser-Ney Smoothing on bigrams, as described in
    Chen & Goodman, 'An Empirical Study of Smoothing Techniques for Language Modeling'

    Some differences though, around words unseen in the training corpus, as I did not find
    Chen & Goodman to be totally clear on this.  (See comments in 'score' method)
    """
    def __init__(self, corpus, d_val=0.75):
        """Initialize your data structures in the constructor."""
        self.d_val = d_val
        self.unique_words_set = set()
        self.unique_words_count = None
        self.bigram_counts = collections.defaultdict(lambda: 0)
        self.bigram_counts_by_first_word = None
        self.unique_words_count_following_this_word = None
        self.unique_words_count_preceding_this_word = None
        self.unique_bigrams_count = None
        self.following_sets = collections.defaultdict(lambda: set())
        self.preceding_sets = collections.defaultdict(lambda: set())
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
                    token = ':'.join([last_word, word])
                    self.bigram_counts[token] += 1
                last_word = word

    def create_bigram_counts_by_first_word(self):
        unique = sorted(list(self.unique_words_set))
        df = pd.DataFrame(index=unique, columns=unique)
        df = df.fillna(0).astype(int)
        for k, v in self.bigram_counts.iteritems():
            words = k.split(':')
            df.loc[words[0], words[1]] = v
        self.set_count_values(df)

    def set_count_values(self, df):
        df_binary = df.copy()
        df_binary[df_binary > 0] = 1
        self.unique_words_count = len(self.unique_words_set)
        self.bigram_counts_by_first_word = dict(df.sum(axis=1))
        self.unique_bigrams_count = df_binary.sum().sum()
        self.unique_words_count_following_this_word = dict(df_binary.sum(axis=1))
        self.unique_words_count_preceding_this_word = dict(df_binary.sum(axis=0))

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
                bigram_count_first_word = self.bigram_counts_by_first_word[last_token]\
                    if last_token in self.bigram_counts_by_first_word else 0
                if bigram_count_first_word == 0:
                    bigram_count_first_word = self.d_val  # Assumption to avoid divide by zero
                first_term = max(bigram_count - self.d_val, 0) / bigram_count_first_word

                unique_words_following_token = self.unique_words_count_following_this_word[last_token]\
                    if last_token in self.unique_words_count_following_this_word else 0
                backoff_factor = (self.d_val / bigram_count_first_word) * unique_words_following_token
                unique_words_preceding_token = self.unique_words_count_preceding_this_word[token]\
                    if token in self.unique_words_count_preceding_this_word else 0
                second_term = backoff_factor * unique_words_preceding_token / self.unique_bigrams_count

                if second_term == 0:  # Assumption to avoid zero probability, based on discussion here: http://stats.stackexchange.com/questions/114863/in-kneser-ney-smoothing-how-are-unseen-words-handled
                    second_term = float(self.d_val) / self.unique_words_count
                score += math.log(first_term + second_term)
            last_token = token
        return score
