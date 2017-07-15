import collections
import copy
import numpy as np
import optparse

from ling.Tree import Tree
import ling.Trees as Trees
import pennParser.EnglishPennTreebankParseEvaluator as \
        EnglishPennTreebankParseEvaluator
import io.PennTreebankReader as PennTreebankReader
import io.MASCTreebankReader as MASCTreebankReader


class Parser:

    def train(self, train_trees):
        pass

    def get_best_parse(self, sentence):
        """
        Should return a Tree
        """
        pass


class Backpointer(object):

    def __init__(self, type, word, coord_x, coord_y):
        self.type = type
        self.word = word
        self.coord_x = coord_x
        self.coord_y = coord_y


class ValueForType(object):

    def __init__(self, probability, backpointers):
        self.probability = probability
        self.backpointers = backpointers


class CKYCell(object):

    def __init__(self, types, probabilities, backpointers):
        self.types_dict = dict()
        for idx, type in enumerate(types):
            self.types_dict[type] = ValueForType(probabilities[idx], backpointers[idx])

    def add_type(self, type, value_for_type):
        self.types_dict[type] = value_for_type


class PCFGParser(Parser):

    def __init__(self):
        self.baseline_parser = BaselineParser()  # used as a fallback

    def train(self, train_trees):
        self.baseline_parser.train(train_trees)
        train_trees_bin = train_trees
        train_trees_bin = [TreeAnnotations.binarize_tree(tree) for tree in train_trees_bin]
        train_trees_bin = [TreeAnnotations.verical_markovize_tree(tree) for tree in train_trees_bin]
        self.lexicon = Lexicon(train_trees_bin)
        self.grammar = Grammar(train_trees_bin)
        # self.print_preterminal_rules()
        # self.print_unary_rules()
        # self.print_binary_rules()

    def print_preterminal_rules(self):
        print 'Possible Preterminal Tags:'
        print self.lexicon.get_all_tags()

    # for debugging
    def print_unary_rules(self):
        items = self.grammar.unary_rules_by_child.items()
        pretty_rules = set()
        for child, rules in items:
            for rule in rules:
                pretty_rules.add(rule.parent + ' -> ' + child)
        print 'Unary Rules:'
        print sorted(pretty_rules)

    # for debugging
    def print_binary_rules(self):
        items = self.grammar.binary_rules_by_left_child.items()
        pretty_rules = set()
        for left_child, rules in items:
            for rule in rules:
                right_child = rule.right_child
                pretty_rules.add(rule.parent + ' -> ' + left_child + ' ' + right_child)
        print 'Binary Rules:'
        print sorted(pretty_rules)

    def add_all_tags(self, word):
        tags = self.lexicon.get_all_tags()
        probabilities = []
        backpointers = []
        nonzero_tags = []
        for tag in tags:
            score = self.lexicon.score_tagging(word, tag)
            if score > 0.0:
                probabilities.append(score)
                backpointers.append([Backpointer(None, word, None, None)])
                nonzero_tags.append(tag)
        return CKYCell(nonzero_tags, probabilities, backpointers)

    def get_best_parse(self, sentence):
        """
        Should return a Tree.
        'sentence' is a list of strings (words) that form a sentence.
        """
        len_sentence = len(sentence)
        # n by n + 1 since top leftmost cky cell is [0,1]
        cky_matrix = np.zeros([len_sentence, len_sentence + 1], dtype='object')
        for i in range(len(cky_matrix)):
            for j in range(len(cky_matrix[0])):
                cky_matrix[i, j] = CKYCell([], [], [])

        # 1a. populate the diagonal of the CKY matrix (level 0)
        for idx, word in enumerate(sentence):
            cky_matrix[idx, idx + 1] = self.add_all_tags(word)

        # 1b. apply unary rules
        self.apply_unary_rules(cky_matrix, 0, sentence)

        # 2a. dynamic programming (with splits)
        for level in range(1, len(sentence)): # diagonal to iterate over
            for idx in range(len(sentence) - level): # index within the diagonal
                cell_idx_row = idx
                cell_idx_col = 1 + level + idx
                cky_cell = cky_matrix[cell_idx_row, cell_idx_col]
                for split in range(level):
                    left_cell_idx_row = cell_idx_row
                    left_cell_idx_col = cell_idx_col + split - level
                    right_cell_idx_row = cell_idx_row + split + 1
                    right_cell_idx_col = cell_idx_col

                    left_cell = cky_matrix[left_cell_idx_row, left_cell_idx_col]
                    right_cell = cky_matrix[right_cell_idx_row, right_cell_idx_col]
                    left_rules = set()
                    for left_type in left_cell.types_dict:
                        left_rules.update(self.grammar.get_binary_rules_by_left_child(left_type))
                    right_rules = set()
                    for right_type in right_cell.types_dict:
                        right_rules.update(self.grammar.get_binary_rules_by_right_child(right_type))
                    all_rules = left_rules.intersection(right_rules)
                    for rule in all_rules:
                        left_type = rule.left_child
                        left_probability = left_cell.types_dict[left_type].probability
                        right_type = rule.right_child
                        right_probability = right_cell.types_dict[right_type].probability
                        new_probability = left_probability * right_probability * rule.score
                        new_type = rule.parent
                        try:
                            if new_probability > 0.0 and (new_type not in cky_cell.types_dict\
                                or new_probability > cky_cell.types_dict[new_type].probability):
                                backpointers = [Backpointer(left_type, None, left_cell_idx_row, left_cell_idx_col),\
                                                Backpointer(right_type, None, right_cell_idx_row, right_cell_idx_col)]
                                value_for_type = ValueForType(new_probability, backpointers)
                                cky_cell.add_type(new_type, value_for_type)
                        except ValueError:
                            import pdb; pdb.set_trace()

            # 2b. apply unary rules
            self.apply_unary_rules(cky_matrix, level, sentence)

        tree = Tree('ROOT', [])
        if 'ROOT' not in cky_matrix[0, len(sentence)].types_dict:
            return self.baseline_parser.get_best_parse(sentence)
        backpointers = cky_matrix[0, len(sentence)].types_dict['ROOT'].backpointers
        self.add_nodes_to_tree(tree, cky_matrix, backpointers)
        unannotated = TreeAnnotations.unannotate_tree(tree)
        # try:
        #     print 'Tree (no word labels)'
        #     print unannotated
        # except UnicodeDecodeError:
        #     pass
        return unannotated

    def add_nodes_to_tree(self, tree, cky_matrix, backpointers):
        if backpointers is not None:
            if backpointers[0].word is not None:
                new_tree = Tree(backpointers[0].word, [])
                tree.children.append(new_tree)
            else:
                for backpointer in backpointers:
                    new_tree = Tree(backpointer.type, [])
                    tree.children.append(new_tree)
                    types_dict = cky_matrix[backpointer.coord_x, backpointer.coord_y].types_dict
                    if backpointer.type in types_dict:
                        child_backpointer = types_dict[backpointer.type].backpointers
                    else:
                        return
                    self.add_nodes_to_tree(new_tree, cky_matrix, child_backpointer)

    def apply_unary_rules(self, cky_matrix, level, sentence):
        added = True
        while added:
            added = False
            for unary_child in self.grammar.unary_rules_by_child:
                for unary_rule in self.grammar.get_unary_rules_by_child(unary_child):
                    unary_parent = unary_rule.parent
                    for idx in range(len(sentence) - level):
                        types_dict = cky_matrix[idx, idx + 1 + level].types_dict
                        if unary_child in types_dict:
                            probability = types_dict[unary_child].probability
                            new_probability = probability * unary_rule.score
                            if new_probability > 0.0 and (unary_parent not in types_dict\
                                    or new_probability > types_dict[unary_parent].probability):
                                backpointers = [Backpointer(unary_child, None, idx, idx + 1 + level)]
                                value_for_type = ValueForType(new_probability, backpointers)
                                cky_matrix[idx, idx + 1 + level].add_type(unary_parent, value_for_type)
                                added = True
        # normalize probabilities
        for idx in range(len(sentence) - level):
            types_dict = cky_matrix[idx, idx + 1 + level].types_dict
            arr = np.array([item[1].probability for item in types_dict.items()]).sum()
            sum = float(arr.sum())
            new_types_dict = dict()
            for item in types_dict.items():
                new_prob = item[1].probability / sum
                new_value_for_type = ValueForType(new_prob, item[1].backpointers)
                new_types_dict[item[0]] = new_value_for_type
            cky_matrix[idx, idx + 1 + level].types_dict = new_types_dict


class BaselineParser(Parser):

    def train(self, train_trees):
        self.lexicon = Lexicon(train_trees)
        self.known_parses = {}
        self.span_to_categories = {}
        for train_tree in train_trees:
            tags = train_tree.get_preterminal_yield()
            tags = tuple(tags)  # because lists are not hashable, but tuples are
            if tags not in self.known_parses:
                self.known_parses[tags] = {}
            if train_tree not in self.known_parses[tags]:
                self.known_parses[tags][train_tree] = 1
            else:
                self.known_parses[tags][train_tree] += 1
            self.tally_spans(train_tree, 0)

    def get_best_parse(self, sentence):
        tags = self.get_baseline_tagging(sentence)
        tags = tuple(tags)
        if tags in self.known_parses:
            return self.get_best_known_parse(tags, sentence)
        else:
            return self.build_right_branch_parse(sentence, list(tags))

    def build_right_branch_parse(self, words, tags):
        cur_position = len(words) - 1
        right_branch_tree = self.build_tag_tree(words, tags, cur_position)
        while cur_position > 0:
            cur_position -= 1
            right_branch_tree = self.merge(
                    self.build_tag_tree(words, tags, cur_position),
                    right_branch_tree)
        right_branch_tree = self.add_root(right_branch_tree)
        return right_branch_tree

    def merge(self, left_tree, right_tree):
        span = len(left_tree.get_yield()) + len(right_tree.get_yield())
        maxval = max(self.span_to_categories[span].values())
        for key in self.span_to_categories[span]:
            if self.span_to_categories[span][key] == maxval:
                most_freq_label = key
                break
        return Tree(most_freq_label, [left_tree, right_tree])

    def add_root(self, tree):
        return Tree("ROOT", [tree])

    def build_tag_tree(self, words, tags, cur_position):
        leaf_tree = Tree(words[cur_position])
        tag_tree = Tree(tags[cur_position], [leaf_tree])
        return tag_tree

    def get_best_known_parse(self, tags, sentence):
        maxval = max(self.known_parses[tags].values())
        for key in self.known_parses[tags]:
            if self.known_parses[tags][key] == maxval:
                parse = key
                break
        parse = copy.deepcopy(parse)
        parse.set_words(sentence)
        return parse

    def get_baseline_tagging(self, sentence):
        tags = [self.get_best_tag(word) for word in sentence]
        return tags

    def get_best_tag(self, word):
        best_score = 0
        best_tag = None
        for tag in self.lexicon.get_all_tags():
            score = self.lexicon.score_tagging(word, tag)
            if best_tag is None or score > best_score:
                best_score = score
                best_tag = tag
        return best_tag

    def tally_spans(self, tree, start):
        if tree.is_leaf() or tree.is_preterminal():
            return 1
        end = start
        for child in tree.children:
            child_span = self.tally_spans(child, end)
            end += child_span
        category = tree.label
        if category != "ROOT":
            if end-start not in self.span_to_categories:
                self.span_to_categories[end-start] = {}
            if category not in self.span_to_categories[end-start]:
                self.span_to_categories[end-start][category] = 1
            else:
                self.span_to_categories[end-start][category] += 1
        return end - start


class TreeAnnotations:

    @classmethod
    def annotate_tree(cls, unannotated_tree):
        """
        Currently, the only annotation done is a lossless binarization
        """

        # TODO: change the annotation from a lossless binarization to a
        # finite-order markov process (try at least 1st and 2nd order)
        # mark nodes with the label of their parent nodes, giving a second
        # order vertical markov process

        return TreeAnnotations.binarize_tree(unannotated_tree)

    @classmethod
    def binarize_tree(cls, tree):
        label = tree.label
        if tree.is_leaf():
            return Tree(label)
        if len(tree.children) == 1:
            return Tree(label, [TreeAnnotations.binarize_tree(tree.children[0])])

        intermediate_label = "@%s->" % label
        intermediate_tree = TreeAnnotations.binarize_tree_helper(
                tree, 0, intermediate_label)
        return Tree(label, intermediate_tree.children)

    @classmethod
    def binarize_tree_helper(cls, tree, num_children_generated,
            intermediate_label):
        left_tree = tree.children[num_children_generated]
        children = []
        children.append(TreeAnnotations.binarize_tree(left_tree))
        if num_children_generated < len(tree.children) - 1:
            right_tree = TreeAnnotations.binarize_tree_helper(
                    tree, num_children_generated + 1,
                    intermediate_label + "_" + left_tree.label)
            children.append(right_tree)
        return Tree(intermediate_label, children)

    @classmethod
    def verical_markovize_tree(cls, tree):
        return TreeAnnotations.vertical_markovize_tree_helper(tree, '')

    @classmethod
    def vertical_markovize_tree_helper(cls, tree, parent_label):
        if tree.is_preterminal() or tree.is_leaf():
            return tree
        else:
            this_label = tree.label
            if parent_label == '':
                new_label = this_label
            else:
                new_label = this_label + '^' + parent_label
            tree.label = new_label
            new_children = [TreeAnnotations.vertical_markovize_tree_helper(child_tree, this_label) for child_tree in tree.children]
            tree.children = new_children
            return tree

    @classmethod
    def at_filter(cls, string):
        if string.startswith('@'):
            return True
        else:
            return False

    @classmethod
    def unannotate_tree(cls, annotated_tree):
        """
        Remove intermediate nodes (labels beginning with "@")
        Remove all material on node labels which follow their base
        symbol (cuts at the leftmost -, ^, or : character)
        Examples: a node with label @NP->DT_JJ will be spliced out,
        and a node with label NP^S will be reduced to NP
        """
        debinarized_tree = Trees.splice_nodes(annotated_tree,
                TreeAnnotations.at_filter)
        unannotated_tree = Trees.FunctionNodeStripper.transform_tree(
                debinarized_tree)
        return unannotated_tree


class Lexicon:
    """
    Simple default implementation of a lexicon, which scores word,
    tag pairs with a smoothed estimate of P(tag|word)/P(tag).

    Instance variables:
    word_to_tag_counters
    total_tokens
    total_word_types
    tag_counter
    word_counter
    type_tag_counter
    """

    def __init__(self, train_trees):
        """
        Builds a lexicon from the observed tags in a list of training
        trees.
        """
        self.total_tokens = 0.0
        self.total_word_types = 0.0
        self.word_to_tag_counters = collections.defaultdict(lambda: \
                collections.defaultdict(lambda: 0.0))
        self.tag_counter = collections.defaultdict(lambda: 0.0)
        self.word_counter = collections.defaultdict(lambda: 0.0)
        self.type_to_tag_counter = collections.defaultdict(lambda: 0.0)

        for train_tree in train_trees:
            words = train_tree.get_yield()
            tags = train_tree.get_preterminal_yield()
            for word, tag in zip(words, tags):
                self.tally_tagging(word, tag)


    def tally_tagging(self, word, tag):
        if not self.is_known(word):
            self.total_word_types += 1
            self.type_to_tag_counter[tag] += 1
        self.total_tokens += 1
        self.tag_counter[tag] += 1
        self.word_counter[word] += 1
        self.word_to_tag_counters[word][tag] += 1


    def get_all_tags(self):
        return self.tag_counter.keys()


    def is_known(self, word):
        return word in self.word_counter


    def score_tagging(self, word, tag):
        p_tag = float(self.tag_counter[tag]) / self.total_tokens
        c_word = float(self.word_counter[word])
        c_tag_and_word = float(self.word_to_tag_counters[word][tag])
        if c_word < 10:
            c_word += 1
            c_tag_and_word += float(self.type_to_tag_counter[tag]) \
                    / self.total_word_types
        p_word = (1.0 + c_word) / (self.total_tokens + self.total_word_types)
        p_tag_given_word = c_tag_and_word / c_word
        return p_tag_given_word / p_tag * p_word


class Grammar:
    """
    Simple implementation of a PCFG grammar, offering the ability to
    look up rules by their child symbols.  Rule probability estimates
    are just relative frequency estimates off of training trees.

    self.binary_rules_by_left_child
    self.binary_rules_by_right_child
    self.unary_rules_by_child
    """

    def __init__(self, train_trees):
        self.unary_rules_by_child = collections.defaultdict(lambda: [])
        self.binary_rules_by_left_child = collections.defaultdict(
                lambda: [])
        self.binary_rules_by_right_child = collections.defaultdict(
                lambda: [])

        unary_rule_counter = collections.defaultdict(lambda: 0)
        binary_rule_counter = collections.defaultdict(lambda: 0)
        symbol_counter = collections.defaultdict(lambda: 0)

        for train_tree in train_trees:
            # print 'tallyin da tree'
            # print Trees.PennTreeRenderer.render(train_tree)
            self.tally_tree(train_tree, symbol_counter,
                    unary_rule_counter, binary_rule_counter)
        for unary_rule in unary_rule_counter:
            unary_prob = float(unary_rule_counter[unary_rule]) \
                    / symbol_counter[unary_rule.parent]
            unary_rule.score = unary_prob
            self.add_unary(unary_rule)
        for binary_rule in binary_rule_counter:
            binary_prob = float(binary_rule_counter[binary_rule]) \
                    / symbol_counter[binary_rule.parent]
            binary_rule.score = binary_prob
            self.add_binary(binary_rule)


    def __unicode__(self):
        rule_strings = []
        for left_child in self.binary_rules_by_left_child:
            for binary_rule in self.get_binary_rules_by_left_child(
                    left_child):
                rule_strings.append(str(binary_rule))
        for child in self.unary_rules_by_child:
            for unary_rule in self.get_unary_rules_by_child(child):
                rule_strings.append(str(unary_rule))
        return "%s\n" % "".join(rule_strings)


    def add_binary(self, binary_rule):
        self.binary_rules_by_left_child[binary_rule.left_child].\
                append(binary_rule)
        self.binary_rules_by_right_child[binary_rule.right_child].\
                append(binary_rule)


    def add_unary(self, unary_rule):
        self.unary_rules_by_child[unary_rule.child].append(unary_rule)


    def get_binary_rules_by_left_child(self, left_child):
        return self.binary_rules_by_left_child[left_child]


    def get_binary_rules_by_right_child(self, right_child):
        return self.binary_rules_by_right_child[right_child]


    def get_unary_rules_by_child(self, child):
        return self.unary_rules_by_child[child]


    def tally_tree(self, tree, symbol_counter, unary_rule_counter,
            binary_rule_counter):
        if tree.is_leaf():
            return
        if tree.is_preterminal():
            return
        if len(tree.children) == 1:
            unary_rule = self.make_unary_rule(tree)
            symbol_counter[tree.label] += 1
            unary_rule_counter[unary_rule] += 1
        if len(tree.children) == 2:
            binary_rule = self.make_binary_rule(tree)
            symbol_counter[tree.label] += 1
            binary_rule_counter[binary_rule] += 1
        if len(tree.children) < 1 or len(tree.children) > 2:
            raise Exception("Attempted to construct a Grammar with " \
                    + "an illegal tree (most likely not binarized): " \
                    + str(tree))
        for child in tree.children:
            self.tally_tree(child, symbol_counter, unary_rule_counter,
                    binary_rule_counter)


    def make_unary_rule(self, tree):
        return UnaryRule(tree.label, tree.children[0].label)


    def make_binary_rule(self, tree):
        return BinaryRule(tree.label, tree.children[0].label,
                tree.children[1].label)


class BinaryRule:
    """
    A binary grammar rule with score representing its probability.
    """

    def __init__(self, parent, left_child, right_child):
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.score = 0.0


    def __str__(self):
        return "%s->%s %s %% %s" % (self.parent, self.left_child, self.right_child, self.score)


    def __hash__(self):
        result = hash(self.parent)
        result = 29 * result + hash(self.left_child)
        result = 29 * result + hash(self.right_child)
        return result


    def __eq__(self, o):
        if self is o:
            return True

        if not isinstance(o, BinaryRule):
            return False

        if (self.left_child != o.left_child):
            return False
        if (self.right_child != o.right_child):
            return False
        if (self.parent != o.parent):
            return False
        return True


class UnaryRule:
    """
    A unary grammar rule with score representing its probability.
    """

    def __init__(self, parent, child):
        self.parent = parent
        self.child = child
        self.score = 0.0

    def __str__(self):
        return "%s->%s %% %s" % (self.parent, self.child, self.score)

    def __hash__(self):
        result = hash(self.parent)
        result = 29 * result + hash(self.child)
        return result

    def __eq__(self, o):
        if self is o:
            return True

        if not isinstance(o, UnaryRule):
            return False

        if (self.child != o.child):
            return False
        if (self.parent != o.parent):
            return False
        return True


MAX_LENGTH = 10

def test_parser(parser, test_trees):
    evaluator = EnglishPennTreebankParseEvaluator.LabeledConstituentEval(
            ["ROOT"], set(["''", "``", ".", ":", ","]))
    for test_tree in test_trees:
        test_sentence = test_tree.get_yield()
        if len(test_sentence) > MAX_LENGTH:
            continue
        guessed_tree = parser.get_best_parse(test_sentence)
        print "Guess:\n%s" % Trees.PennTreeRenderer.render(guessed_tree)
        print "Gold:\n%s" % Trees.PennTreeRenderer.render(test_tree)
        evaluator.evaluate(guessed_tree, test_tree)
    print ""
    return evaluator.display(True)


def read_trees(base_path, low=None, high=None):
    trees = PennTreebankReader.read_trees(base_path, low, high)
    return [Trees.StandardTreeNormalizer.transform_tree(tree) \
        for tree in trees]


def read_masc_trees(base_path, low=None, high=None):
    print "Reading MASC from %s" % base_path
    trees = MASCTreebankReader.read_trees(base_path, low, high)
    return [Trees.StandardTreeNormalizer.transform_tree(tree) \
        for tree in trees]


if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--path", dest="path",
            default="../data/")
    # TODO: Choose the dataset here: "masc" or "miniTest"
    opt_parser.add_option("--data", dest="data", default = "masc")
    # TODO: Choose the parser here: "BaselineParser" or "PCFGParser"
    opt_parser.add_option("--parser", dest="parser",
            default="BaselineParser")
    opt_parser.add_option("--maxLength", dest="max_length",
            default="20")
    opt_parser.add_option("--testData", dest="test_data", default="")

    (options, args) = opt_parser.parse_args()
    options = vars(options)

    print "PCFGParserTest options:"
    for opt in options:
        print "  %-12s: %s" % (opt, options[opt])
    print ""
    MAX_LENGTH = int(options['max_length'])

    parser = globals()[options['parser']]()
    print "Using parser: %s" % parser.__class__.__name__

    base_path = options['path']
    pre_base_path = base_path
    data_set = options['data']
    if not base_path.endswith('/'):
        base_path += '/'

    print "Data will be loaded from: %s" % base_path

    train_trees = []
    validation_trees = []
    test_trees = []

    if data_set == 'miniTest':
        base_path += 'parser/%s' % data_set

        # training data: first 3 of 4 datums
        print "Loading training trees..."
        train_trees = read_trees(base_path, 1, 3)
        print "done."

        # test data: last of 4 datums
        print "Loading test trees..."
        test_trees = read_trees(base_path, 4, 4)
        print "done."

    if data_set == "masc":
        base_path += "parser/"

        # training data: MASC train
        print "Loading MASC training trees... from: %smasc/train" % base_path
        train_trees.extend(read_masc_trees("%smasc/train" % base_path, 0, 38))
        print "done."
        print "Train trees size: %d" % len(train_trees)
        print "First train tree: %s" % \
                Trees.PennTreeRenderer.render(train_trees[0])
        print "Last train tree: %s" % \
                Trees.PennTreeRenderer.render(train_trees[-1])

        # test data: MASC devtest
        print "Loading MASC test trees..."
        test_trees.extend(read_masc_trees("%smasc/devtest" % base_path, 0, 11))
        #test_trees.extend(read_masc_trees("%smasc/blindtest" % base_path, 0, 8))
        print "done."
        # test_trees = test_trees[:10]
        print "Test trees size: %d" % len(test_trees)
        print "First test tree: %s" % \
                Trees.PennTreeRenderer.render(test_trees[0])
        print "Last test tree: %s" % \
                Trees.PennTreeRenderer.render(test_trees[-1])
        counts = collections.defaultdict(lambda: 0)
        counts_child = collections.defaultdict(lambda: 0)

        for test_tree in test_trees:
            counts[test_tree.label] += 1
            if len(test_tree.children) > 0:
                counts_child["|".join([child.label for child in test_tree.children])] += 1

        print "counts"
        print counts
        print "counts child"
        print counts_child

    if data_set not in ["miniTest", "masc"]:
        raise Exception("Bad data set: %s: use miniTest or masc." % data_set)

    print ""
    print "Training parser..."
    parser.train(train_trees)

    print "Testing parser"
    test_parser(parser, test_trees)
