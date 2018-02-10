import numpy as np
from learner import Student

class Human():
    def __init__(self):
        self.name = 'HUMAN'
    def move(self, gomoku, last_move):
        print(gomoku.valid_moves())
        move = input()
        return int(move)
    
class RandomAgent():
    def __init__(self, insight=True):
        self.name = 'RANDOM'
        self.insight = insight
    def move(self, gomoku, last_move):
        move = gomoku.sample()
        if self.insight:
            print('Move: ' + str(move))
        return move
    
class Bot():
    def __init__(self, model_path, insight=True, time=1):
        self.name = 'BOT'
        self.insight = insight
        self.student = Student(model_path, insight=insight, time=time)
        self.last_node = None
    def move(self, gomoku, last_move=None):
        root = None
        if (not last_move is None) and (not self.last_node is None):
            for child in self.last_node.children:
                if child.move == last_move:
                    root = child
                    break

        best_node, p = self.student.move(gomoku, root)
        if self.insight:
            print('Move: ' + str(best_node.move))
            print(self.student.mcts.stats())
        self.last_node = best_node
        return best_node.move