import numpy as np
from learner import Student

class Human():
    def __init__(self):
        self.name = 'HUMAN'
    def move(self, gomoku):
        print(gomoku.valid_moves())
        move = input()
        gomoku = gomoku.move(int(move))
        return gomoku
    
class RandomAgent():
    def __init__(self, insight=True):
        self.name = 'RANDOM'
        self.insight = insight
    def move(self, gomoku):
        move = gomoku.sample()
        gomoku = gomoku.move(move)
        if self.insight:
            print('Move: ' + str(move))
        return gomoku
    
class Bot():
    def __init__(self, model_path, insight=True):
        self.name = 'BOT'
        self.insight = insight
        self.student = Student(model_path, insight)
    def move(self, gomoku):
        move, p = self.student.move(gomoku)
        if self.insight:
            print('Move: ' + str(move))
            print('Probabilities: ' + str(np.round(p,4)))
        gomoku = gomoku.move(move)
        return gomoku