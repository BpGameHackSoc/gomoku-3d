import numpy as np
import keras
from keras.layers import Conv2D, Dense, Flatten, LeakyReLU, Input
from keras.models import Sequential, load_model, Model
from keras.initializers import RandomUniform
from collections import deque
import math
import random
import string
from model import Gomoku, Color
from mcts import MCTS

class Memory():
    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)
    def __len__(self):
        return len(self.memory)
    def add(self, array):
        self.memory.extend(array)
    def sample(self):
        n = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, n)
    def clear(self):
        self.memory.clear()

class Student():
    def __init__(self, model_path=None, insight=False, time=1, learning=False):
        self.brain = Brain(model_path)
        self.mcts = MCTS(self.brain, time=time, learning=learning)
        self.memory = Memory(1000, 400)
        self.insight = insight
    def move(self, gomoku):
        root = self.mcts.search(gomoku)
        probabilities = self.mcts.rank_children(root)
        best_child = self.mcts.choose_best_child(root)
        return best_child.move, probabilities
    def save_knowledge(self, classes):
        self.memory.add(classes)
    def learn(self):
        samples = self.memory.sample()
        states = np.array([a[0] for a in samples])
        values = np.array([a[1] for a in samples])
        probabilities = np.array([a[2] for a in samples])
        self.brain.learn(states, values, probabilities)

class Teacher():
    def __init__(self, model_path=None, time=1, insight=False):
        self.student = Student(model_path, insight=insight, time=time, learning=True)
    def teach(self, no_of_games, verbose=0):
        for game_index in range(no_of_games):
            gomoku = Gomoku()
            states = []
            probabilities = []
            while not gomoku.is_over():
                move, p = self.student.move(gomoku)
                states.extend(gomoku.get_states_in_all_rotations())
                probabilities.extend(np.tile(p[:,1], 4).reshape(4,-1))
                gomoku = gomoku.move(move)
            states = np.array(states)
            probabilities = np.array(probabilities)
            values = self.get_values(states.shape[0]//4, gomoku.winner)
            to_learn = list(zip(states.tolist(), values.tolist(), probabilities.tolist()))
            states = [a[0] for a in to_learn]
            values = [a[1] for a in to_learn]
            probabilities = [a[2] for a in to_learn]
            self.student.save_knowledge(to_learn)
            self.student.learn()
            if not verbose == 0:
                if game_index % verbose == 0:
                    print('Game ' + str(game_index) + '...')
        self.student.brain.save()

    def get_values(self, n, winner):
        if winner == Color.DRAW:
            v = np.zeros((n, 1))
        else:
            v = np.ones((n, 1))
            if winner == Color.BLUE:
                v[1::2] = -1
            else:
                v[0::2] = -1
        return np.tile(v, 4).flatten()

class Brain():
    def __init__(self, model_path=None):
        if model_path is None:
            self.new_model()
        else:
            self.load(model_path)

    def new_model(self):
        init = RandomUniform(minval=0.0001, maxval=0.0002, seed=None)

        inp = Input((28, 4, 4))
        conv = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu',
                           data_format='channels_first', kernel_initializer=init)(inp)
        conv2 = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu',
                            data_format='channels_first', kernel_initializer=init)(conv)
        flat = Flatten()(conv2)

        value1 = Dense(256, kernel_initializer=init, activation='tanh')(flat)
        value2 = Dense(256, kernel_initializer=init, activation='tanh')(value1)
        value3 = Dense(256, kernel_initializer=init, activation='tanh')(value2)

        action1 = Dense(256, kernel_initializer=init, activation='sigmoid')(flat)
        action2 = Dense(256, kernel_initializer=init, activation='sigmoid')(action1)
        action3 = Dense(256, kernel_initializer=init, activation='sigmoid')(action2)

        out1 = Dense(1, activation="tanh", name="v")(value3)
        out2 = Dense(16, activation="softmax", name="p")(action3)

        model = Model(inp, [out1,out2])
        model.compile(loss=['mean_squared_error', 'categorical_crossentropy'],
                      optimizer='adam')
        self.model = model

    def learn(self, states, values, probabilities):
        states = np.array(states)
        values = np.array(values)
        probabilities = np.array(probabilities)
        self.model.train_on_batch(states, [values, probabilities])

    def predict(self, states):
        return self.model.predict(states)

    def save(self):
        path = 'models/' + self.__random_string(15) + '.h5'
        self.model.save(path)
        print('Model saved in ' + path)
    def load(self, path):
        self.model = load_model(path)
    def __random_string(self, n):
        return ''.join(np.random.choice(list(string.ascii_lowercase+ string.digits), n))


# class Teacher():
#     def __init__(self, model_path=None, T_start = 3, T_end = 0.1):
#         self.student = Student(model_path, insight=True)
#         self.T_start = T_start
#         self.T_end = T_end
#     def teach(self, no_of_games, verbose=0):
#         for game_index in range(no_of_games):
#             gomoku = Gomoku()
#             states = [gomoku.get_state()]
#             values = [self.student.model.predict(np.array(states)).flatten()[0]]
#             while not gomoku.is_over():
#                 T = self.__temp(game_index, no_of_games, self.T_start, self.T_end)
#                 gomoku, move, expected_value = self.student.move(gomoku, learn=True, T=T)
#                 states.append(gomoku.get_state())
#                 values.append(expected_value)
#             states = np.array(states)
#             values = np.array(values)
#             outcome = gomoku.winner.value
#             values = self.__regrade(values, outcome)
#             to_learn = list(zip(states.tolist(), values.tolist()))
#             self.student.save_knowledge(to_learn)
#             self.student.learn()
#             if not verbose == 0:
#                 if game_index % verbose == 0:
#                     print('Game ' + str(game_index) + '...')
#         save_name = self.__random_string(15)
#         self.student.save('models/' + save_name + '.h5')

        
            
                
#     def __temp(self, x, no_of_games, start, end):
#         d = math.log(start)
#         k = -math.log10(end) + 1
#         return np.exp(d-(x/no_of_games*k*d))
    
#     def __regrade(self, x, target, LAMDA=0.3):
#         x[-1] = target
#         for i in reversed(range(x.size-1)):
#                 x[i] = (1-LAMDA) * x[i] + LAMDA * x[i+1]
#         return x
    
#     def __random_string(self, n):
#         return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


# class Student():
#     def __init__(self, model_path=None, insight=False):
#         if model_path is None:
#             self.new_model()
#         else:
#             self.load(model_path)
#         self.memory = Memory(120, 40)
#         self.insight=insight
#     def new_model(self):
#         init = RandomUniform(minval=0.0001, maxval=0.0002, seed=None)
#         a = LeakyReLU(alpha=0.3)
#         model = Sequential()
#         model.add(Conv2D(256, kernel_size=(3,3), padding='same', data_format='channels_first', input_shape=(28, 4, 4), kernel_initializer=init))
#         model.add(LeakyReLU(alpha=.03))
#         model.add(Flatten())
#         model.add(Dense(512, kernel_initializer=init))
#         model.add(LeakyReLU(alpha=.03))
#         model.add(Dense(51, kernel_initializer=init))
#         model.add(LeakyReLU(alpha=.03))
#         model.add(Dense(1, activation='tanh'))
#         model.compile(optimizer='rmsprop', loss='mean_squared_error')
#         self.model = model
    
#     def move(self, gomoku, learn=False, T=1):
#         child_states = []
#         moves = []
#         valid_moves = gomoku.valid_moves()
#         for move in valid_moves:
#             child = gomoku.move(move)
#             child_states.append(child)
#             moves.append(move)
#         child_states = np.array(child_states)
#         predictions = (self.model.predict(child_states) * gomoku.turn.value).flatten()
#         probabilities = self.__softmax(predictions, T)
#         if not learn:
#             if self.insight:
#                 print(predictions)
#             index = predictions.argmax()
#             chosen_move = moves[index]
#             value = predictions[index]
#         else:
#             possibilities = np.arange(probabilities.size)
#             index = np.random.choice(possibilities, p=probabilities)
#             chosen_move = moves[index]
#             value = predictions[index]
#         value *= gomoku.turn.value
#         g = gomoku.move(chosen_move)
#         return g, chosen_move, value
    
#     def save_knowledge(self, classes):
#         self.memory.add(classes)
    
#     def learn(self):
#         samples = self.memory.sample()
#         x = [a[0] for a in samples]
#         y = [a[1] for a in samples]
#         self.model.train_on_batch(x, y)
        
#     def __softmax(self, x, T=1):
#         e_x = np.exp((x - np.max(x))/T)
#         return e_x / e_x.sum(axis=0)
    
#     def save(self, path):
#         self.model.save(path)
#     def load(self, path):
#         self.model = load_model(path)
