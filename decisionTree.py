__author__ = 'Jerry Cheng'


class Node(object):
    def __init__(self):
        self.if_true = None
        self.if_false = None
        self.data = -1


def insert(tree, decisions, factor):
    node = tree
    for decision in decisions:
        if decision[1]:
            node = node.if_true
        else:
            node = node.if_false
    node.data = factor[0]
    return tree


def create_tree(factor):
    node = Node()
    node.data = factor[0]
    return node


def print_tree(node, words):
    queue = [node]
    temp_queue = []
    while True:
        while queue:
            current_node = queue.pop(0)
            if not current_node:
                print('terminate'),
            else:
                print(current_node.data),

                temp_queue.append(current_node.if_true)
                temp_queue.append(current_node.if_false)
        print()
        queue = temp_queue
        temp_queue = []
        if not queue:
            break;
