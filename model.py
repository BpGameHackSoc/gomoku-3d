import numpy as np
from enum import IntEnum

class Color(IntEnum):
    DRAW = 0
    BLUE = 1
    RED = -1

class Gomoku():
    def __init__(self, board=None, turn=None):
        self.board = np.zeros((4,4,4), dtype=np.int) if board is None else board
        self.turn = Color.BLUE if turn is None else turn
        
    def move(self, n):
        if n not in self.valid_moves():
            raise ValueError('Invalid move (' + str(n) + ').')
        n = n-1
        row = n // 4
        column = n % 4
        board = np.copy(self.board)
        for i in range(4):
            if board[i,row,column] == 0:
                board[i,row,column] = self.turn.value
                break
        turn = Color(self.turn.value * -1)
        return Gomoku(board, turn)
        
    def sample(self):
        return np.random.choice(self.valid_moves())
        
    def is_over(self):
        layers = self.get_layers()
        sum1 = layers[:].sum(axis=1).flatten()
        sum2 = layers[:].sum(axis=2).flatten()
        sum3 = layers.reshape(-1,16)[:,3:13:3].sum(axis=1)
        sum4 = layers.reshape(-1,16)[:,0:16:5].sum(axis=1)
        s = np.concatenate((sum1, sum2, sum3, sum4))
        if (s == 4).any():
            self.winner = Color.BLUE
            return True
        elif (s == -4).any():
            self.winner = Color.RED
            return True
        elif self.valid_moves().size == 0:
            self.winner = Color.DRAW
            return True
        else:
            return False
        
    def get_layers(self):
        x = np.zeros((14,4,4), dtype=int)
        for dim in range(3):
            for j in range(4):
                i = dim * 4 + j
                if dim == 0:
                    x[i] = self.board[j,:,:]
                elif dim == 1:
                    x[i] = self.board[:,j,:]
                else:
                    x[i] = self.board[:,:,j]
        x[12] = self.board.reshape(-1,16)[:,0:16:5].reshape(4,4)
        x[13] = self.board.reshape(-1,16)[:,3:14:3].reshape(4,4)
        return x

    def get_states_in_all_rotations(self):
        gomokus = self.get_permutations()
        states = np.zeros((8,28,4,4))
        for i in range(8):
            states[i] = gomokus[i].get_state()
        return states

    def get_permutations(self):
        gomokus = []
        for i in range(4):
            gomokus.append(self.rotate(i, False))
            gomokus.append(self.rotate(i, True))
        return gomokus

    def rotate(self, half_radians, flip):
        b = self.board.copy()
        for i in range(4):
            b[i] = np.rot90(b[i], half_radians)
            if flip:
                b[i] = b[i].transpose()
        return Gomoku(board=b, turn=self.turn)
    
    def get_state(self):
        s = np.zeros((28,4,4))
        layers = self.get_layers()
        first_mask = layers == self.turn.value
        second_mask = layers == self.turn.value * -1
        s[:14][first_mask] = 1
        s[14:][second_mask] = 1
        return s
    
    def valid_moves(self):
        mask = (self.board[3] == 0).reshape(16)
        return np.arange(16)[mask] + 1
    
    def save(self):
        return [self.turn, np.copy(self.board)]
    def load(self, settings):
        self.turn = settings[0]
        self.board = np.copy(settings[1])
    def copy(self):
        g = Gomoku()
        settings = self.save()
        g.turn = settings[0]
        g.board = settings[1]
        return g
