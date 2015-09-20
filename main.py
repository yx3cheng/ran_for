import math
import decisionTree
import random
from collections import defaultdict
import operator

f_traindata = open('traindata.txt')
f_trainlabel = open('trainlabel.txt')
f_testdata = open('testdata.txt')
f_testlabel = open('testlabel.txt')
f_words = open('words.txt')

words = f_words.read().splitlines()
f_words.seek(0)

words_reverse_dict = {}
with f_words:
    for line in f_words.readlines():
        words_reverse_dict[line.rstrip('\n')] = len(words_reverse_dict) + 1

trainlabel = f_trainlabel.read().split()
traindata = f_traindata.read().split()
traindata_tuples = zip(traindata[::2], traindata[1::2])
traindata_dict = {k: [] for k in range(1, len(trainlabel) + 1)}
for traindata_tuple in traindata_tuples:
    traindata_dict[int(traindata_tuple[0])].append(int(traindata_tuple[1]))

testlabel = f_testlabel.read().split()
testdata = f_testdata.read().split()
testdata_tuples = zip(testdata[::2], testdata[1::2])
testdata_dict = {k: [] for k in range(1, len(testlabel) + 1)}
for testdata_tuple in testdata_tuples:
    testdata_dict[int(testdata_tuple[0])].append(int(testdata_tuple[1]))


def entropy(x, y):
    if x + y == 0:
        return 0
    f_p = float(x) / float(x + y)
    f_n = float(y) / float(x + y)
    entropy_sum = 0
    if f_p > 0:
        entropy_sum -= f_p * math.log(f_p, 2)
    if f_n > 0:
        entropy_sum -= f_n * math.log(f_n, 2)
    return entropy_sum


def remainder(x, y, p_total):
    if x + y == 0:
        return 1
    return float(x + y) / p_total * entropy(x, y)


def ig(p1, n1, p2, n2, p_total):
    ig = entropy(p1 + p2, n1 + n2)
    if p1 + n1 > 0:
        ig -= remainder(p1, n1, p_total)
    if p2 + n2 > 0:
        ig -= remainder(p2, n2, p_total)
    return ig


def prob_comparator(item_x, item_y):
    x = item_x[1]
    y = item_y[1]

    if x[2] > y[2]:
        return 1
    elif x[2] == y[2]:
        return 0
    else:
        return -1


def evaluate_accuracy(data, labels, trees, words_reverse_dictionary):
    data_set = set(data)
    correct = 0
    incorrect = 0
    indeterminate = 0

    for i in range(1, len(labels) + 1):
        tally = [0, 0, 0]
        for tree in trees:
            current_node = tree

            while current_node:
                if current_node.data == 1 or current_node.data == 2:
                    break
                if (str(i), str(words_reverse_dictionary.get(str(current_node.data)))) in data_set:
                    current_node = current_node.if_true
                else:
                    current_node = current_node.if_false

            if current_node.data == -1:
                tally[0] += 1
            else:
                tally[current_node.data] += 1

        node = tally.index(max(tally))

        if str(node) == labels[i - 1]:
            correct += 1
        else:
            incorrect += 1

    return correct, correct + incorrect


def choose_attribute(attributes, examples, labels, attribute_subset):
    # compute total
    ca_prob_dict = {}

    attributes_filtered = [attribute for attribute in attributes if random.random() < attribute_subset]
    #print attributes_filtered

    for attribute in attributes_filtered:
        ca_prob_dict[attribute] = (0, 0)

    ca_total = len(examples.keys())
    ca_total1 = 0
    ca_total2 = 0

    # compute number of 1s and 2s in sample
    for example_key in examples.keys():
        if labels[example_key - 1] == '1':
            ca_total1 += 1
        elif labels[example_key - 1] == '2':
            ca_total2 += 1

    # compute number of TFs with each word
    for attribute in attributes_filtered:
        for key, value in examples.iteritems():
            if labels[key - 1] == '1' and attribute in value:
                ca_prob_dict[attribute] = (ca_prob_dict[attribute][0] + 1, ca_prob_dict[attribute][1])
            elif labels[key - 1] == '2' and attribute in value:
                ca_prob_dict[attribute] = (ca_prob_dict[attribute][0], ca_prob_dict[attribute][1] + 1)

    # compute information gain
    ca_ig_dict = defaultdict.fromkeys(attributes_filtered)

    for key, value in ca_prob_dict.iteritems():
        ca_ig_dict[key] = ig(value[0], value[1], ca_total1 - value[0],
                             ca_total2 - value[1], ca_total)

    # print(value[0], value[1], ca_total1 - value[0], ca_total2 - value[1])
    gnf_max_item = max(ca_ig_dict.iteritems(), key=operator.itemgetter(1))
    # print(words[gnf_max_item[0] - 1] , gnf_max_item)
    return gnf_max_item[0], ca_prob_dict[gnf_max_item[0]]


def find_classification(examples, labels):
    current_classification = None
    for key in examples.keys():
        if not current_classification:
            current_classification = int(labels[key - 1])
        elif current_classification != int(labels[key - 1]):
            return 0
    return current_classification


def find_mode(examples, labels):
    count_dict = {1: 0, 2: 0}

    for key in examples.keys():
        count_dict[int(labels[key - 1])] += 1
    return 1 if count_dict[1] > count_dict[2] else 2


def dtl(examples, labels, attributes, height, current_height, default, attribute_subset):
    if not examples:
        node = decisionTree.Node()
        node.data = default  # unknown
        return node

    classification = find_classification(examples, labels)
    if classification:
        node = decisionTree.Node()
        node.data = classification  # all examples agree, return classification
        return node

    if current_height == height or not attributes:
        classification = find_mode(examples, labels)
        node = decisionTree.Node()
        node.data = classification
        return node

    temp_factor = choose_attribute(attributes, examples, labels, attribute_subset)
    node = decisionTree.Node()
    node.data = words[temp_factor[0] - 1]

    attributes_i = list(attributes)
    attributes_i.remove(temp_factor[0])

    # type 1
    examples_i = dict(examples)
    examples_i = {k: examples_i[k] for k in examples_i if temp_factor[0] in examples_i[k]}
    subtree = dtl(examples_i, labels, attributes_i, height, current_height + 1, default, attribute_subset)
    node.if_true = subtree

    # type 2
    examples_i = dict(examples)
    examples_i = {k: examples_i[k] for k in examples_i if temp_factor[0] not in examples_i[k]}
    subtree = dtl(examples_i, labels, attributes_i, height, current_height + 1, default, attribute_subset)
    node.if_false = subtree

    return node


g_attributes = list(range(1, len(words) + 1))
g_default = 2

data_subset = 0.5
attribute_subset = 0.5

trees = []
tree_height = 20

for tree_height in range(1, 7):
    print 'tree height ', tree_height
    for i in xrange(200):
        traindata_dict_filtered = {key: traindata_dict[key] for key in traindata_dict if random.random() < data_subset}
        trainlabel_filtered = trainlabel

        ##print traindata_dict_filtered
        ##print trainlabel_filtered
        # print 'height = ', tree_height

        g_tree = dtl(traindata_dict_filtered, trainlabel_filtered, g_attributes, tree_height, 0, g_default,
                     attribute_subset)
        trees.append(g_tree)
        print i,
    print ''

    print 'accuracy = ', evaluate_accuracy(traindata_tuples, trainlabel, trees, words_reverse_dict), \
        evaluate_accuracy(testdata_tuples, testlabel, trees, words_reverse_dict)
    print''
