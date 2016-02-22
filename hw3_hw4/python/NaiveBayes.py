# NLP Programming Assignment #3
# NaiveBayes
# 2012

#
# The area for you to implement is marked with TODO!
# Generally, you should not need to touch things *not* marked TODO
#
# Remember that when you submit your code, it is not run from the command line
# and your main() will *not* be run. To be safest, restrict your changes to
# addExample() and classify() and anything you further invoke from there.
#


import sys
import getopt
import os
import math
import collections
import numpy as np
import operator
import re
import sklearn as sl
import sklearn.feature_extraction
import sklearn.svm


class NaiveBayes(object):
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """

        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """

        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.stopList = set(self.readFile('../data/english.stop'))
        self.numFolds = 10
        self.classCountsByClass = collections.Counter()
        self.wordCountsByClass = collections.defaultdict(collections.Counter)
        self.wordCountsTotal = collections.Counter()
        self.alpha = 1  # add-one smoothing factor

    #############################################################################
    # TODO TODO TODO TODO TODO

    def classify(self, words):
        """ TODO
          'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """
        classScores = dict()
        numDocumentsTotal = sum(self.classCountsByClass.values())
        for klass in self.wordCountsByClass.keys():
            score = math.log(self.classCountsByClass[klass]) / float(numDocumentsTotal)
            for word in words:
                numerator = self.wordCountsByClass[klass][word] + self.alpha
                denominator = sum(self.wordCountsByClass[klass].values()) + self.alpha * len(self.wordCountsByClass[klass])
                score += math.log(numerator / float(denominator))
            classScores[klass] = score
        return sorted(classScores.items(), key=operator.itemgetter(1))[-1][0]

    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier
         * in the NaiveBayes class.
         * Returns nothing
        """
        self.classCountsByClass.update([klass])
        self.wordCountsByClass[klass].update(words)
        self.wordCountsTotal.update(words)

    def filterStopWords(self, words):
        """
        * TODO
        * Filters stop words found in self.stopList.
        """
        return [word for word in words if not word in self.stopList]

    # TODO TODO TODO TODO TODO
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words = self.filterStopWords(words)
            self.addExample(example.klass, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            splits.append(split)
        return splits

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for example in split.test:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """Builds the splits for training/testing"""
        trainData = []
        testData = []
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    example.klass = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                for fileName in negTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    example.klass = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                split.train.append(example)

            posTestFileNames = os.listdir('%s/pos/' % testDir)
            negTestFileNames = os.listdir('%s/neg/' % testDir)
            for fileName in posTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                example.klass = 'pos'
                split.test.append(example)
            for fileName in negTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                example.klass = 'neg'
                split.test.append(example)
            splits.append(split)
        return splits


class NaiveBayesBoolean(NaiveBayes):
    def __init__(self):
        super(NaiveBayesBoolean, self).__init__()

    def classify(self, words):
        words_set = set(words)
        return super(NaiveBayesBoolean, self).classify(words_set)

    def addExample(self, klass, words):
        words_set = set(words)
        super(NaiveBayesBoolean, self).addExample(klass, words_set)


class NaiveBayesBooleanWithNegation(NaiveBayes):
    def __init__(self):
        super(NaiveBayesBooleanWithNegation, self).__init__()
        self.negation_re = r'[Nn]ot|n\'t$|[Nn]ever'
        self.punctuation_set = set('.,?!()')
        self.negation_prefix = 'NOT_'

    def classify(self, words):
        words_with_negation = self._negate_words(words)
        return super(NaiveBayesBooleanWithNegation, self).classify(words_with_negation)

    def addExample(self, klass, words):
        words_with_negation = self._negate_words(words)
        super(NaiveBayesBooleanWithNegation, self).addExample(klass, words_with_negation)

    def _negate_words(self, words):
        words_with_negation = list(words)
        negate = False
        for idx in range(1, len(words_with_negation)):
            if words_with_negation[idx] in self.punctuation_set:
                negate = False
            elif negate or re.match(self.negation_re, words[idx - 1]) is not None:
                words_with_negation[idx] = self.negation_prefix + words_with_negation[idx]
                negate = True
        return words_with_negation


class Svm(NaiveBayes):
    """
    Despite this being an SVM, for now inherit from NaiveBayes as a hack to reuse loading, cross validation methods
    """
    def __init__(self, C, kernel):
        super(Svm, self).__init__()
        self._is_model_trained = False
        self._corpuses = []
        self._classes = []
        self._val_map = {'neg': 0, 'pos': 1}
        self._model = None
        self._vectorizer = None
        self._C = C
        self._kernel = kernel

    def classify(self, words):
        if not self._is_model_trained:
            self._train_model()
            self._is_model_trained = True
        x = self._vectorizer.transform([' '.join(words)]).toarray()
        if self._model.predict([x[-1, :]])[0] == 0:
            return 'neg'
        else:
            return 'pos'

    def addExample(self, klass, words):
        if self._is_model_trained:
            self._corpuses = [] # sl.feature_extraction.text.CountVectorizer(min_df=1)
            self._classes = []
            self._is_model_trained = False
        self._classes.append(klass)
        self._corpuses.append(' '.join(words))

    def _train_model(self):
        self._vectorizer = sl.feature_extraction.text.CountVectorizer(min_df=1)
        X_sparse = self._vectorizer.fit_transform(self._corpuses)
        X = X_sparse.toarray()
        y = np.array([self._val_map[val] for val in self._classes])
        self._model = sl.svm.SVC(C=self._C, kernel=self._kernel)
        self._model.fit(X, y)
        self._is_model_trained = True


class GenericFactory(object):
    def __init__(self):
        pass


class NaiveBayesFactory(GenericFactory):
    def get_model(self):
        return NaiveBayes()


class NaiveBayesBooleanFactory(GenericFactory):
    def get_model(self):
        return NaiveBayesBoolean()


class NaiveBayesBooleanWithNegationFactory(GenericFactory):
    def get_model(self):
        return NaiveBayesBooleanWithNegation()


class SvmFactory(GenericFactory):
    def __init__(self, C):
        super(SvmFactory, self).__init__()
        self.C = C  # Reminder: in sklearn, higher magnitude C means less regularization
        self.kernel = 'linear'

    def get_model(self):
        return Svm(C=self.C, kernel=self.kernel)


def main():
    nb = NaiveBayes()

    # default parameters: no stop word filtering, and
    # training/testing on ../data/imdb1
    if len(sys.argv) < 2:
        options = [('', '')]
        args = ['../data/imdb1/']
    else:
        (options, args) = getopt.getopt(sys.argv[1:], 'fbns:')
    if ('-f', '') in options:
        nb.FILTER_STOP_WORDS = True
    if ('-b', '') in options:
        model_factory = NaiveBayesBooleanFactory()
    elif('-n', '') in options:
        model_factory = NaiveBayesBooleanWithNegationFactory()
    elif('-s', '1') in options:
        model_factory = SvmFactory(1)
    elif('-s', '1e3') in options:
        model_factory = SvmFactory(1e3)
    elif('-s', '1e6') in options:
        model_factory = SvmFactory(1e6)
    else:
        model_factory = NaiveBayesFactory()

    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = model_factory.get_model()
        accuracy = 0.0
        for example in split.train:
            words = example.words
            if nb.FILTER_STOP_WORDS:
                words = classifier.filterStopWords(words)
            classifier.addExample(example.klass, words)
        for example in split.test:
            words = example.words
            if nb.FILTER_STOP_WORDS:
                words = classifier.filterStopWords(words)
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


if __name__ == "__main__":
    main()
