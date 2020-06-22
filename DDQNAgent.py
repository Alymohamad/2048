# -*- coding: utf-8 -*-
"""
TODO: Parameter namen Refactoren
"""
import numpy as np  # For numerical fast numerical calculations
import matplotlib.pyplot as plt  # For making plots
import pandas as pd  # Deals with data
import seaborn as sns  # Makes beautiful plots
from sklearn.preprocessing import StandardScaler  # Testing sklearn
import tensorflow  # Imports tensorflow
from tensorflow.keras.optimizers import Adam  # Imports keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Dropout, Flatten


# Agents Memory
# allows to sample non sequential memories
##############################################################################################################################################################################################################################################################


class ReplayBuffer(object):

    #TODO: Hier input dims parametarisieren oder löschen
    # max size -> size of the buffer - input shape 1 4x4 matrix - n_actions -> 4 actions - discrete allows the buffer to work with DDPG when action space is couninous ist es aber hier nicht
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        # terminal_memory da expexted reward nach game over = 0
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_new, done):
        # will go from 0 to size -1 then goes back again to 0 and starts same process again -> we overrite memories from the oldest memories
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        # when the episode is over it is 0 -> when episode is over it is true which ist 1 thats why 1 - 1 = 0 what we want
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    # not gonna samle whole replaybuffer, only subset
    def sample_buffer(self, batch_size):
        # want to sample only the ones that are filled not the zeros deswegen entweder alles oder den counter nehmen
        max_mem = min(self.mem_cntr, self.mem_size)
        # gibt mir ein array an ingerest von 0 bis max_mem-1 in der größe von batch_size
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

##############################################################################################################################################################################################################################################################


# Build Deep Q Network
##############################################################################################################################################################################################################################################################

    # TODO: Die Parameter des Creates in der Methode parameterisieren
    # def create_model(lr, n_actions, input_dims, fcl_dims, fcl2_dims):
     #   model = Sequential([
      #          Dense(fcl_dims,input_shape(input_dims,)),
       #         Activation('relu')
        #        Dense(n_actions)])
        #model.compile(optimizer=Adam(lr=lr), loss='mse')
        # return model

def create_model(self, lr, n_actions, input_dims, conv1_dims, dropout_conv1, conv2_dims, dropout_conv2, fcl_dims):
    model = Sequential()
    # add model layers
    # conv1_dims = 128 -- input_dims =  4,4,1
    #TODO: input_shape=(4,4,1),) should be parameterised so it can also play other grid sizes
    model.add(Conv2D(conv1_dims, kernel_size=(2, 2),
                     activation='relu', padding='same', input_shape=(input_dims, input_dims, 1,)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(dropout_conv1))  # dropout_conv1 = 0.2
    # conv2_dims = 128
    model.add(Conv2D(conv2_dims, kernel_size=(2, 2),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(dropout_conv2))  # dropout_conv2 = 0.2
    model.add(Flatten())
    model.add(Dense(fcl_dims))  # fcl_dims = 265
    model.add(Dense(n_actions))  # n_actions = 4
    model.add(Activation('softmax'))
    # learning rate = 0.001
    model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

##############################################################################################################################################################################################################################################################


# Class Handles Agent
# Agent has memory, chooses action, stores memory, learns
##############################################################################################################################################################################################################################################################

class DDQNAgent(object):

    # alpha = learning rate - gamma = discount factor - epsilon - batch_size - input dimensions
    # um wv epsilon weniger wird - epsilon minimum wert, max memory size = 1 million
    # name of file - nach wv er syncen soll zwischen den 2 Networks replace target ist hyper parameter
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decr=0.999966, epsilon_end=0.001,
                 mem_size=1000000, fname='ddqn_model.h5', replace_target=100):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decr = epsilon_decr
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.fname = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        # evaluate DQN - fitting Model
        self.q_eval = create_model(
            self, alpha, n_actions, input_dims, 128, 0.2, 128, 0.2, 265)
        # alpha, n_actions, input_dims, 128, 0.2, 128, 0.2, 265
        # Target DQN - never gonna perform any fitting on it going to sync the weights every replace_target steps
        self.target = create_model(
            self, alpha, n_actions, input_dims, 128, 0.2, 128, 0.2, 265)


    # stores state transitions
    def remember(self, state, action, reward, new_states, terminal):
        self.memory.store_transition(state, action, reward, new_states, terminal)

    #TODO: Hier ausschließen das eine nicht mögliche aktion ausgewählt wird
    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            #TODO: possible action space übergeben und nicht normale action space
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(np.expand_dims(state, -1))
            action = np.argmax(actions) #TODO: dann mach ich ja trotzdem max Operation???
        return action

    #at first start playing to fill agents memory than start learning when filled
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)
            state = np.expand_dims(state, -1)
            new_state = np.expand_dims(new_state, -1)

            #TODO: Weil ansonsten shapes nicht passen aufgrund der channels siehe https://stackoverflow.com/questions/56008114/tensorflow-expected-conv2d-input-to-have-4-dimensions
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            # Hier DQN Q Fromel von goolgles paper Y_t DQN = ....

            # used for values for the actions that we calculate to be maximum
            q_next = self.target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)

            # q preticted to handle y - q(s,a) part
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval,axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward +\
                                                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done
            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_decr if self.epsilon > self.epsilon_end else self.epsilon_end

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.fname)

    def load_model(self):
        self.q_eval = load_model(self.fname)

        # da jetzt target network nach dem laden zufällige gewichtungen hat und das eval network optimale hat müssen wir das target anpassen
        if self.epsilon <= self.epsilon_end:
            self.update_network_parameters()















"""triple quotes 
    def step(self, action):

        reward, done = 0, 0

        if action == 0:  # if action is 0, move paddle to left
            paddle_left()
            reward -= .1  # reward of -0.1 for moving the paddle

        if action == 2:  # if action is 2, move paddle to right
            paddle_right()
            reward -= .1  # reward of -0.1 for moving the paddle

        run_frame()  # run the game for one frame, reward is also updated inside this function

        # creating the state vector
        state = [paddle.xcor(), ball.xcor(), ball.ycor(), ball.dx, ball.dy]

        return reward, state, done """







