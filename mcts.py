import datetime
from model import Color
import numpy as np

class MCTS():
    def __init__(self, model, **kwargs):
        self.seconds = kwargs.get('time', 3)
        self.learning = kwargs.get('learning', False)
        self.model = model
        self.time_limit = datetime.timedelta(seconds=self.seconds)

    def search(self, gomoku):
        self.root = Node(gomoku, None, None, 0)
        self.root.visit(self.model, noise=self.learning)
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.time_limit:
            self.run_one()
        return self.root

    def choose_best_child(self, parent_node):
        best_node = parent_node.children[0]
        for child in parent_node.children:
            if child.N > best_node.N:
                best_node = child
        return best_node

    def run_one(self):
        last_node = self.simulation()
        last_node.visit(self.model)
        self.backpropagation(last_node)

    def backpropagation(self, node):
        current_node = node.parent
        while not current_node is None:
            current_node.Q_blue += node.Q(Color.BLUE)
            current_node.Q_red += node.Q(Color.RED)
            current_node.N += 1
            current_node = current_node.parent

    def simulation(self):
        node = self.root
        while node.visited() and not node.is_terminal:
            node = self.choose(node)
        return node

    def choose(self, node):
        best_child = node.children[0]
        side = node.gomoku.turn
        for child in node.children:
            if child.value(side) > best_child.value(side):
                best_child = child
        return best_child

    def rank_children(self, node):
        ranks = np.column_stack((np.arange(16)+1, np.zeros(16)))
        for i, child in enumerate(node.children):
            ranks[child.move-1, 1] = child.N
        ranks[:,1] /= ranks[:,1].sum()
        return ranks

    def stats(self):
        inf = {}
        inf['max_depth'] = self.root.max_depth()
        inf['prediction'] = self.root.value(self.root.gomoku.turn)
        inf['blue_q'] = self.root.value(Color.BLUE)
        inf['red_q'] = self.root.value(Color.RED)
        inf['n'] = self.root.N
        inf['node/s'] = self.root.N / self.seconds
        ranks = self.rank_children(self.root)
        ranks = ranks[ranks[:,1].argsort(kind='mergesort')]
        ranks = ranks[::-1]
        inf['ranks'] = ranks
        return inf


class Node():
    def __init__(self, gomoku, move, parent, p):
        self.move = move
        self.gomoku = gomoku
        self.parent = parent
        self.Q_blue = 0
        self.Q_red = 0
        self.p = p
        self.N = 1
        self.children = []
        self.is_terminal = False

    def value(self, color):
        return self.Q(color) + self.p / self.N

    def visited(self):
        return not self.N == 1

    def max_depth(self):
        if self.gomoku.is_over() or not self.visited():
            return 0
        else:
            m = 0
            for child in self.children:
                current = child.max_depth()
                if current > m:
                    m = current
            return m + 1

    def visit(self, model, noise=False):
        if self.gomoku.is_over():
            self.visit_end()
            self.is_terminal = True
        else:
            self.visit_non_end(model, noise)
        self.N += 1

    def visit_end(self):
        if self.gomoku.winner == Color.DRAW:
            self.Q_blue += 0
            self.Q_red += 0
        elif self.gomoku.winner == Color.BLUE:
            self.Q_blue += 1
            self.Q_red += -1
        else:
            self.Q_blue += -1
            self.Q_red += 1

    def visit_non_end(self, model, noise=False):
        s = np.expand_dims(self.gomoku.get_state(), axis=0)
        v, p = model.predict(s)
        v = v.flatten()
        p = p.flatten()
        if noise:
            p += np.random.dirichlet([0.0625] * 16)
        self.__set_Q(v[0])
        moves = self.gomoku.valid_moves()
        for move in moves:
            child = self.gomoku.move(move)
            current_p = p[move-1]
            self.children.append(Node(child, move, self, current_p))


    def Q(self, color):
        if color == Color.BLUE:
            return self.Q_blue / self.N
        else:
            return self.Q_red / self.N

    def __set_Q(self, x):
        self.Q_blue += x if self.gomoku.turn == Color.BLUE else -x
        self.Q_red += x if self.gomoku.turn == Color.RED else -x


