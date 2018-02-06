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
        return random.sample(list(self.memory), n)
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
                      optimizer='adam')
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
