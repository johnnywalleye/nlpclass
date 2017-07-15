import base64
import collections
import json, sys
from Datum import Datum
import operator
import re

class FeatureFactory:
    """
    Add any necessary initialization steps for your features here
    Using this constructor is optional. Depending on your
    features, you may not need to intialize anything.
    """
    MIN_COMMON_NOUN_LENGTH = 3
    MIN_COMMON_NOUN_NUM_OCCURRENCES = 75

    def __init__(self, to_exclude_set):
        self.to_exclude_set = to_exclude_set
        self.of_words = {'de', 'van', 'der', 'della'}
        self.first_names_set = set()
        self.last_names_set = set()
        self.names_except_first_set = set()
        self.last_names_set_b = set()
        self.names_except_first_set_b = set()
        self.non_names = set()
        self.follows_name_set = set()
        self.lower_case_words_set = set()
        self.days_of_week = {'Monday', 'Tuesday', 'Wendesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        self.months_of_year = {'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'}
        self.common_nouns = self.initialize_common_nouns()
        self.precedes_non_name = self.initialize_word_sets(-1, 'O')
        self.precedes_name = self.initialize_word_sets(-1, 'P')
        self.after_non_name = self.initialize_word_sets(1, 'O')
        self.after_name = self.initialize_word_sets(1, 'P')
        self.precedes_non_name_b = self.initialize_word_sets(-2, 'O')
        self.precedes_name_b = self.initialize_word_sets(-2, 'P')
        self.after_non_name_b = self.initialize_word_sets(2, 'O')
        self.after_name_b = self.initialize_word_sets(2, 'P')
        self.top_words = set()
        self.articles = {'a', 'an', 'the'}
        self.titles = {'mrs', 'mr', 'ms', 'prof', 'dr', 'gen', 'rep', 'sen', 'st'}

        # names from http://www.quietaffiliate.com/free-first-name-and-last-name-databases-csv-and-sql/ (don't help much)
        f = open('additionalData/CSV_Database_of_First_Names.csv')
        self.first_names_k = set()
        for line in f:
            k = line.split('\r')
            for word in k:
                self.first_names_k.add(word)

        f = open('additionalData/CSV_Database_of_Last_Names.csv')
        self.last_names_k = set()
        for line in f:
            k = line.split('\r')
            for word in k:
                self.last_names_k.add(word)


    """
    Words is a list of the words in the entire corpus, previousLabel is the label
    for position-1 (or O if it's the start of a new sentence), and position
    is the word you are adding features for. PreviousLabel must be the
    only label that is visible to this method. 
    """

    def computeFeatures(self, words, previousLabel, position):

        features = []
        currentWord = words[position]

        """ Baseline Features """
        features.append("word=" + currentWord)
        features.append("prevLabel=" + previousLabel)
        features.append("word=" + currentWord + ", prevLabel=" + previousLabel)
	"""
        Warning: If you encounter "line search failure" error when
        running the program, considering putting the baseline features
	back. It occurs when the features are too sparse. Once you have
        added enough features, take out the features that you don't need. 
	"""


	""" TODO: Add your features here """
        cWord = currentWord
        pWord = ''
        if position > 0:
            pWord = words[position - 1]
        ppWord = ''
        if position > 1:
            ppWord = words[position - 2]
        nWord = ''
        if position < len(words) - 1:
            nWord = words[position + 1]
        nnWord = ''
        if position < len(words) - 2:
            nnWord = words[position + 2]
        nnnWord = ''
        if position < len(words) - 3:
            nnnWord = words[position + 3]

        if position == 0:
            for word in words:
                if word == word.lower():
                    self.lower_case_words_set.add(word)

        if len(self.top_words) == 0:
            self.initialize_top_words(words)

        if previousLabel == 'PERSON' and not self.is_camel(ppWord):
            self.first_names_set.add(pWord)
        if self.is_camel(ppWord) and previousLabel == 'PERSON' and not self.is_camel(cWord)\
                and not pWord.lower() in self.lower_case_words_set and not pWord in self.common_nouns and not pWord in self.months_of_year:
            self.last_names_set.add(pWord)
        if previousLabel == 'PERSON' and not pWord.lower() in self.lower_case_words_set and not pWord in self.common_nouns\
                and not pWord in self.months_of_year:
            self.names_except_first_set.add(pWord)

        if previousLabel == 'PERSON' and self.is_camel(cWord) and not self.is_camel(nWord)\
                and not cWord.lower() in self.lower_case_words_set and not cWord in self.common_nouns:
            self.last_names_set_b.add(cWord)
        if previousLabel == 'PERSON' and self.is_camel(cWord) and not cWord.lower() in self.lower_case_words_set\
                and not cWord in self.common_nouns:
            self.names_except_first_set_b.add(cWord)

        if previousLabel != 'PERSON':
            self.non_names.add(pWord.lower())

        if previousLabel == 'PERSON' and cWord == cWord.lower():
            self.follows_name_set.add(cWord)

        features.append('prefix1=' + cWord[:1])
        features.append('prefix2=' + cWord[:2])
        features.append('suffix1=' + cWord[-1:])
        features.append('suffix2=' + cWord[-2:])

        features.append('hasHyphen=' + str(self.has_hyphen(cWord)))
        features.append('cWordShape=' + self.word_shape(cWord))
        features.append('cWordShapeShort=' + self.short_word_shape(cWord))
        features.append('pWordShapeShort=' + self.short_word_shape(pWord))
        features.append('nWordShapeShort=' + self.short_word_shape(nWord))

        if cWord in self.articles:
            features.append('cWordIsArticle=True')
        if pWord in self.articles:
            features.append('pWordIsArticle=True')
        if nWord in self.articles:
            features.append('nWordIsArticle=True')

        if self.is_camel(cWord):
            numConsecutive = 1
            idx = position - 1
            while idx >= 0 and self.is_camel(words[idx]):
                idx -= 1
                numConsecutive += 1
            idx = position + 1
            while idx < len(words) and self.is_camel(words[idx]):
                idx += 1
                numConsecutive += 1
            features.append('numConsecutiveCaps=' + str(numConsecutive))

        if self.is_camel(cWord)\
            and self.is_camel(nWord)\
            and self.is_camel(nnWord)\
            and self.is_camel(nnnWord):
            features.append('manyCapsFollowing=True')

        if cWord.lower() in self.titles or cWord.lower()[:-1] in self.titles:
            features.append('cIsTitle=True')
        if pWord.lower() in self.titles or pWord.lower()[:-1] in self.titles:
            features.append('pIsTitle=True')
        if nWord.lower() in self.titles or nWord.lower()[:-1] in self.titles:
            features.append('nIsTitle=True')

        if pWord in self.top_words:
            features.append('pWord=' + pWord)
        else:
            features.append('pWord=' + 'NA')
        if nWord in self.top_words:
            features.append('nWord=' + nWord)
        else:
            features.append('nWord=' + 'NA')
        if ppWord in self.top_words:
            features.append('ppWord=' + ppWord)
        else:
            features.append('ppWord=' + 'NA')
        if nnWord in self.top_words:
            features.append('nnWord=' + nnWord)
        else:
            features.append('nnWord=' + 'NA')

        if cWord.lower() in self.lower_case_words_set:
            features.append("appearsAsLowerCase=True")

        if cWord in self.days_of_week:
            features.append('dayOfWeek=True')

        if cWord in self.months_of_year:
            features.append('monthOfYear=True')

        if cWord in self.months_of_year:
            features.append('commonNoun=True')

        if pWord in self.months_of_year:
            features.append('pCommonNoun=True')

        if self.is_of(cWord):
            features.append("isOf=True")
        if self.is_punc(cWord):
            features.append('isPunc=True')
        self.appendCombinations(features, self.is_initial, 'Initial', pWord, cWord, nWord)
        if cWord == '.':
            features.append("cIsPeriod=True")
        if pWord == '.':
            features.append("pIsPeriod=True")
        if cWord == nWord:
            features.append("sameAsNext=True")

        features.append('hasNumber=' + str(self.has_number(cWord)))

        if cWord in self.first_names_k:
            features.append('firstNameK=True')
        if cWord in self.last_names_k:
            features.append('lastNameK=True')
        if cWord in self.first_names_set:
            features.append('firstName=True')
        if cWord in self.last_names_set and self.is_camel(cWord):
            features.append("lastName=True")
        if cWord in self.names_except_first_set and self.is_camel(cWord):
            features.append('nonFirstName=True')
        if cWord in self.last_names_set_b and self.is_camel(cWord):
            features.append("lastNameB=True")
        if cWord in self.names_except_first_set_b and self.is_camel(cWord):
            features.append('nonFirstNameB=True')

        if nWord == ',' and nnWord == 'who':
            features.append('commaWho=True')

        if nWord.lower() in self.non_names:
            features.append("nNonNameLower=True")

        if pWord in self.precedes_name:
            features.append("pWordPrecedesNames=True")
        if pWord in self.precedes_non_name:
            features.append("pWordPrecedesNonNames=True")
        if nWord in self.after_name:
            features.append("nWordPrecedesNames=True")
        if nWord in self.after_non_name:
            features.append("nWordPrecedesNonNames=True")

        if ppWord in self.precedes_name_b:
            features.append("ppWordPrecedesNames=True")
        if ppWord in self.precedes_non_name_b:
            features.append("ppWordPrecedesNonNames=True")
        if nnWord in self.after_name_b:
            features.append("nnWordPrecedesNames=True")
        if nnWord in self.after_non_name_b:
            features.append("nnWordPrecedesNonNames=True")

        if cWord in self.follows_name_set:
            features.append("FollowsName=True")

        if nWord in self.follows_name_set:
            features.append("nextFollowsName=True")

        if nWord == ',':
            features.append('commaNext=True')

        if pWord == '\'s':
            features.append('possesivePrev=True')

        if nWord == '\'s':
            features.append('possesiveNext=True')

        if self.is_camel(cWord) and not self.is_camel(pWord) and not self.is_camel(nWord):
            features.append('loneCamel=True')

        if self.is_all_caps(cWord):
            features.append('allCaps=True')

        if pWord != '.' and self.is_camel(cWord):
            features.append('camelNotAtStartOfSentence=True')

        if cWord.islower():
            features.append('isLower=True')
        if pWord.islower():
            features.append('pIsLower=True')
        if nWord.islower():
            features.append('nIsLower=True')

        return features

    def initialize_word_sets(self, offset, word_type):

        word_counts_near_p = collections.defaultdict(lambda: 1)
        word_counts_near_o = collections.defaultdict(lambda: 1)

        word_and_cats = []
        for line in open('../data/train'):
            word_and_cats.append(line.split('\t'))

        word_and_cats = []
        for line in open('../data/train'):
            word_and_cats.append(line.split('\t'))

        for idx in range(len(word_and_cats)):
            if idx >= offset and idx > 0 and (idx + offset) < len(word_and_cats) and idx < len(word_and_cats):

                if len(word_and_cats[idx]) < 2 or len(word_and_cats[idx + offset]) < 2:
                    continue

                word = word_and_cats[idx][0]
                cat = word_and_cats[idx][1]

                prev_word = word_and_cats[idx - 1][0]

                other_word = word_and_cats[idx + offset][0]
                other_cat = word_and_cats[idx + offset][1]

                if prev_word != '.':
                    if cat[0] == 'O' and self.is_camel(word):
                        word_counts_near_o[other_word] += 1
                    elif cat[0] == 'P' and self.is_camel(word):
                        word_counts_near_p[other_word] += 1

        all_keys = set(word_counts_near_p.keys()).union(set(word_counts_near_o.keys()))
        ratios = dict()

        near_p_set = set()
        near_o_set = set()

        for key in all_keys:
            ratio = word_counts_near_p[key] / float(word_counts_near_o[key])
            if ratio < 0.2:
                near_o_set.add(key)
            elif ratio > 5.0:
                near_p_set.add(key)

        if word_type == 'P':
            return near_p_set
        elif word_type == 'O':
            return near_o_set

    def initialize_common_nouns(self):
        prev_word = ''
        word_counts = collections.defaultdict(lambda: 0)

        with open('../data/train') as f:
            for idx, line in enumerate(f.readlines()):
                word = line.split('\t')[0]
                if self.is_camel(word) and prev_word != '.':
                    word_counts[word] += 1
                prev_word = word

        common_nouns_list = list()
        for item in sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True):
            word = item[0]
            count = item[0]
            if len(word) > FeatureFactory.MIN_COMMON_NOUN_LENGTH and count > FeatureFactory.MIN_COMMON_NOUN_NUM_OCCURRENCES:
                common_nouns_list.append(word)
        return set(common_nouns_list)

    def initialize_top_words(self, words):
        word_counts = dict()
        for word in words:
            if word in word_counts.keys():
                word_counts[word] = word_counts[word] + 1
            else:
                word_counts[word] = 1
        sorted_tuples = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
        for tuple in sorted_tuples[:1000]:
            self.top_words.add(tuple[0])

    def has_hyphen(self, word):
        return word.find('-') != -1

    def word_shape(self, word):
        result = []
        for letter in word:
            if re.match(r'[A-Z]', letter):
                result.append('X')
            elif re.match(r'[a-z]', letter):
                result.append('x')
            elif re.match(r'[0-9]', letter):
                result.append('d')
            else:
                result.append(letter)
        return ''.join(result)

    def short_word_shape(self, word):
        result = []
        result_idx = 0
        for idx, letter in enumerate(word):
            if re.match(r'[A-Z]', letter) and (idx == 0 or result[result_idx - 1] != 'X'):
                result.append('X')
                result_idx += 1
            elif re.match(r'[a-z]', letter) and (idx == 0 or result[result_idx - 1] != 'x'):
                result.append('x')
                result_idx += 1
            elif re.match(r'[0-9]', letter) and (idx == 0 or result[result_idx - 1] != 'd'):
                result.append('d')
                result_idx += 1
            elif (not re.match(r'[A-Z]', letter)) and (not re.match(r'[a-z]', letter)) and (not re.match(r'[0-9]', letter)):
                result.append(letter)
        return ''.join(result)


    def appendCombinations(self, features, func, func_name, pWord, cWord, nWord):
        prev_name = "p" + func_name + "="
        curr_name = "c" + func_name + "="
        next_name = "n" + func_name + "="

        prev_f = str(func(pWord))
        curr_f = str(func(cWord))
        next_f = str(func(nWord))

        features.append(prev_name + prev_f)
        features.append(curr_name + curr_f)
        features.append(next_name + next_f)
        features.append(prev_name + prev_f + curr_name + curr_f)
        features.append(prev_name + prev_f + curr_name + curr_f + next_name + next_f)


    def is_camel(self, word):
        return self.re_match(r'[A-Z][a-z]+', word)

    def is_all_caps(self, word):
        return self.re_match(r'^[A-Z]+$', word)

    def has_number(self, word):
        return self.re_match(r'[0-9]', word)

    def is_punc(self, word):
        return self.re_match(r'^[^A-Za-z]$', word)

    def is_of(self, word):
        if word in self.of_words:
            return True
        else:
            return False

    def is_initial(self, word):
        return self.re_match(r'^[A-Z][.]$', word)

    def re_match(self, regex, word):
        if re.search(regex, word):
            return True
        else:
            return False

    """ Do not modify this method """
    def readData(self, filename):
        data = [] 
        
        for line in open(filename, 'r'):
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data

    """ Do not modify this method """
    def readTestData(self, ch_aux):
        data = [] 
        
        for line in ch_aux.splitlines():
            line_split = line.split()
            # remove emtpy lines
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]

            datum = Datum(word, label)
            data.append(datum)

        return data


    """ Do not modify this method """
    def setFeaturesTrain(self, data):
        newData = []
        words = []

        for datum in data:
            words.append(datum.word)

        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        previousLabel = "O"
        for i in range(0, len(data)):
            datum = data[i]

            newDatum = Datum(datum.word, datum.label)
            newDatum.features = self.computeFeatures(words, previousLabel, i)
            newDatum.previousLabel = previousLabel
            newData.append(newDatum)

            previousLabel = datum.label

        return newData

    """
    Compute the features for all possible previous labels
    for Viterbi algorithm. Do not modify this method
    """
    def setFeaturesTest(self, data):
        newData = []
        words = []
        labels = []
        labelIndex = {}

        for datum in data:
            words.append(datum.word)
            if not labelIndex.has_key(datum.label):
                labelIndex[datum.label] = len(labels)
                labels.append(datum.label)
        
        ## This is so that the feature factory code doesn't
        ## accidentally use the true label info
        for i in range(0, len(data)):
            datum = data[i]

            if i == 0:
                previousLabel = "O"
                datum.features = self.computeFeatures(words, previousLabel, i)

                newDatum = Datum(datum.word, datum.label)
                newDatum.features = self.computeFeatures(words, previousLabel, i)
                newDatum.previousLabel = previousLabel
                newData.append(newDatum)
            else:
                for previousLabel in labels:
                    datum.features = self.computeFeatures(words, previousLabel, i)

                    newDatum = Datum(datum.word, datum.label)
                    newDatum.features = self.computeFeatures(words, previousLabel, i)
                    newDatum.previousLabel = previousLabel
                    newData.append(newDatum)

        return newData

    """
    write words, labels, and features into a json file
    Do not modify this method
    """
    def writeData(self, data, filename):
        outFile = open(filename + '.json', 'w')
        for i in range(0, len(data)):
            datum = data[i]
            jsonObj = {}
            jsonObj['_label'] = datum.label
            jsonObj['_word']= base64.b64encode(datum.word)
            jsonObj['_prevLabel'] = datum.previousLabel

            featureObj = {}
            features = datum.features
            for j in range(0, len(features)):
                feature = features[j]
                featureObj['_'+feature] = feature
            jsonObj['_features'] = featureObj
            
            outFile.write(json.dumps(jsonObj) + '\n')
            
        outFile.close()

