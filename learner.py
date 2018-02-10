import numpy as np
import keras
from keras.layers import Activation, Conv2D, Dense, Flatten, Input
from keras.models import Sequential, load_model, Model
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform
from collections import deque
import math
import random
import string
from model import Gomoku, Color
from mcts import MCTS
import datetime

TAUGHT_GAMES = 10
EXPECTED_GAME_LENTH = 25
NO_OF_TRANSFORMATION = 8
NO_OF_PICKS = 20
EPOCHS = 1
PICK_SIZE = int((TAUGHT_GAMES * NO_OF_TRANSFORMATION * EXPECTED_GAME_LENTH) / NO_OF_PICKS)
BATCH_SIZE = int(PICK_SIZE * NO_OF_PICKS / 10)

class Memory():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.memory = deque()
    def add(self, array):
        self.memory.extend(array)
    def sample(self):
        n = min(self.batch_size, len(self.memory))
        return random.sample(list(self.memory), n)
    def clear(self):
        self.memory.clear()

class Student():
    def __init__(self, model_path=None, insight=False, time=1, learning=False):
        self.brain = Brain(model_path)
        self.mcts = MCTS(self.brain, time=time, learning=learning)
        self.memory = Memory(PICK_SIZE)
        self.insight = insight
    def move(self, gomoku, root=None):
        root = self.mcts.search(gomoku, root)
        probabilities = self.mcts.rank_children(root)
        best_child = self.mcts.choose_best_child(root)
        return best_child, probabilities
    def save_knowledge(self, classes):
        self.memory.add(classes)
    def learn(self, save=False, name=None):
        states = []
        values = []
        probabilities = []
        for i in range(NO_OF_PICKS):
            samples = self.memory.sample()
            states.extend([a[0] for a in samples])
            values.extend([a[1] for a in samples])
            probabilities.extend([a[2] for a in samples])
        self.brain.learn(states, values, probabilities, save, name)
        self.memory.clear()

class Teacher():
    def __init__(self, think_time=1, insight=False, model_path=None):
        self.student = Student(model_path, insight=insight, time=think_time, learning=True)
        self.played_games = self.get_no_of_games_played(model_path)
    def teach(self, minutes, checkpoint):
        time_limit = datetime.timedelta(minutes=minutes)
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < time_limit:
            gomoku = Gomoku()
            states = []
            probabilities = []
            root = None
            while not gomoku.is_over():
                best_child, p = self.student.move(gomoku, root)
                states.extend(gomoku.get_states_in_all_rotations())
                probabilities.extend(np.tile(p[:,1], NO_OF_TRANSFORMATION).reshape(NO_OF_TRANSFORMATION,-1))
                gomoku = gomoku.move(best_child.move)
                root = best_child
            self.played_games += 1
            states = np.array(states)
            probabilities = np.array(probabilities)
            values = self.get_values(states.shape[0]//NO_OF_TRANSFORMATION, gomoku.winner)
            to_learn = list(zip(states.tolist(), values.tolist(), probabilities.tolist()))
            states = [a[0] for a in to_learn]
            values = [a[1] for a in to_learn]
            probabilities = [a[2] for a in to_learn]
            self.student.save_knowledge(to_learn)
            if self.played_games % TAUGHT_GAMES == 0:
                print('Game ' + str(self.played_games) + '...')
                self.student.learn()
            if self.played_games % checkpoint == 0:
                self.student.brain.save(name='game_'+str(self.played_games))

        print('Last game : ' + str(self.played_games))
        self.student.learn(save=True, name='game_'+str(self.played_games))


    def get_values(self, n, winner):
        if winner == Color.DRAW:
            v = np.zeros((n, 1))
        else:
            v = np.ones((n, 1))
            if winner == Color.BLUE:
                v[1::2] = -1
            else:
                v[0::2] = -1
        return np.tile(v, NO_OF_TRANSFORMATION).flatten()

    def get_no_of_games_played(self, s):
        if s is None:
            return 0
        x = s.find('game_')
        if not x == -1:
            start = x + 5
            end = s.find('.')
            return int(s[start:end])
        return 0

class Brain():
    def __init__(self, model_path=None):

        self.settings = {}
        self.settings['filter_n'] = 128
        self.settings['kernel_size'] = 3
        self.settings['res_layer_n'] = 3
        self.settings['init'] = RandomUniform(minval=0.0001, maxval=0.0002, seed=None)

        if model_path is None:
            self.new_model()
        else:
            self.load(model_path)

    def new_model(self):
        inp = Input((28, 4, 4))
        x = self.conv_layer(inp, 'firstconv_')
        for i in range(self.settings['res_layer_n']):
            x = self.res_layer(x, i+1)
        value = self.value_head(x)
        policy = self.policy_head(x)
        model = Model(inp, [value, policy])
        model.compile(loss=['mean_squared_error', 'categorical_crossentropy'],
                      optimizer='rmsprop')
        self.model = model


    def conv_layer(self, x, prefix, suffix='', original_x=None):
        x = Conv2D(
            filters=self.settings['filter_n'],
            kernel_size=self.settings['kernel_size'],
            padding='same',
            strides=1,
            data_format='channels_first',
            kernel_initializer=self.settings['init'],
            name=prefix+"_conv"+suffix
        )(x)
        x = BatchNormalization(axis=1, name=prefix+"_batchnorm"+suffix)(x)
        if not original_x is None:
            x = Add(name=prefix+"_sum"+suffix)([original_x, x])
        x = Activation("relu", name=prefix+"_relu"+suffix)(x)
        return x

    def res_layer(self, x, index):
        original_x = x
        prefix = 'res_' + str(index)
        x = self.conv_layer(x, prefix, '_1')
        x = self.conv_layer(x, prefix, '_2', original_x)
        return x

    def value_head(self, x):
        kernel_size = self.settings['kernel_size']
        filter_n = self.settings['filter_n']
        self.settings['filter_n'] = 1
        self.settings['kernel_size'] = 1
        x = self.conv_layer(x, 'value_')
        self.settings['filter_n'] = filter_n
        self.settings['filter_n'] = kernel_size
        x = Flatten()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('tanh')(x)
        return x

    def policy_head(self, x):
        kernel_size = self.settings['kernel_size']
        filter_n = self.settings['filter_n']
        self.settings['filter_n'] = 2
        self.settings['kernel_size'] = 1
        x = self.conv_layer(x, 'policy_')
        self.settings['filter_n'] = filter_n
        self.settings['filter_n'] = kernel_size
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation('sigmoid')(x)
        return x

    def learn(self, states, values, probabilities, save=False, name=None):
        states = np.array(states)
        values = np.array(values)
        probabilities = np.array(probabilities)
        self.model.fit(states, [values, probabilities], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
        if save:
            self.save(name)


    def predict(self, states):
        return self.model.predict(states)

    def save(self, name=None):
        name = self.__random_string(15) if name is None else name
        path = 'models/' + name + '.h5'
        self.model.save(path)
        print('Model saved in ' + path)
    def load(self, path):
        self.model = load_model(path)
    def __random_string(self, n):
        return ''.join(np.random.choice(list(string.ascii_lowercase+ string.digits), n))
