#!/usr/bin/env python

from __future__ import division
from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle
import random
import json

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import he_normal
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

from rl_glue import RL_num_episodes

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_ROWS = 6
NUM_COLUMNS = 9
GOAL_STATE = (5, 8)
#6 rows by 9 columns = 54 when unrolled
FEATURE_VECTOR_SIZE = 54

#Parameters
EPSILON = None
ALPHA = None
GAMMA = None
NUM_ACTIONS = 4
EPSILON_MIN = 0.1

#Agents
RANDOM = 'random'
NEURAL = 'neural'
AUX = 'aux'
TABULAR = 'tabularQ'

def agent_init(random_seed):
    global state_action_values, observed_state_action_pairs, observed_states, model, num_steps, cur_epsilon

    #Different parts of the program use np.random (via utils.py) and others use just random,
    #seeding both with the same seed here to make sure they both start in the same place per run of the program
    np.random.seed(random_seed)
    random.seed(random_seed)

    #Reset epsilon, as we may want to decay it per episode
    cur_epsilon = EPSILON
    print("Epsilon at run start: {}".format(cur_epsilon))

    if AGENT == TABULAR:
        #The real world estimates for each state action pair
        state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]
    elif AGENT == NEURAL:

        #Initialize the neural network
        model = Sequential()
        init_weights = he_normal()

        model.add(Dense(164, kernel_initializer=init_weights, input_shape=(FEATURE_VECTOR_SIZE,)))
        model.add(Activation('relu'))

        model.add(Dense(150, kernel_initializer=init_weights))
        model.add(Activation('relu'))

        model.add(Dense(4, kernel_initializer=init_weights))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        #sgd = SGD()
        #rms = RMSprop()
        #adagrad = Adagrad()
        #adadelta = Adadelta()
        #adam = Adam()
        #adamax = Adamax()
        nadam = Nadam()
        model.compile(loss='mse', optimizer=nadam)


def agent_start(state):
    global state_action_values, cur_state, cur_action

    cur_state = state
    if AGENT == TABULAR or AGENT == RANDOM:
        #All value functions are initialized to zero, so we can just select randomly for the first action, since they all tie
        cur_action = rand_in_range(NUM_ACTIONS)
    elif AGENT == NEURAL:
        cur_action = get_max_action(state)
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action, observed_state_action_pairs, model, num_steps, cur_epsilon

    next_state = state

    if AGENT == TABULAR:
        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            #Need to ensure that an action is picked uniformly at random from among those that tie for maximum
            cur_max = state_action_values[state[0]][state[1]][0]
            max_indices = [0]
            for i in range(1, len(state_action_values[state[0]][state[1]])):
                if state_action_values[state[0]][state[1]][i] > cur_max:
                    cur_max = state_action_values[state[0]][state[1]][i]
                    max_indices = [i]
                elif state_action_values[state[0]][state[1]][i] == cur_max:
                    max_indices.append(i)
            next_action = max_indices[rand_in_range(len(max_indices))]
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Update the state action values
        next_state_max_action = state_action_values[next_state[0]][next_state[1]].index(max(state_action_values[next_state[0]][next_state[1]]))
        state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward + GAMMA * state_action_values[next_state[0]][next_state[1]][next_state_max_action] - state_action_values[cur_state[0]][cur_state[1]][cur_action])

    elif AGENT == RANDOM:
        next_action = rand_in_range(NUM_ACTIONS)

    elif AGENT == NEURAL:
        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:

            #Get the best action over all actions possible in the next state,
            q_vals = model.predict(encode_1_hot(next_state), batch_size=1)
            q_max = np.max(q_vals)
            next_action = np.argmax(q_vals)
            cur_action_target = reward + GAMMA * q_max

            #Get the value for the current state for which the action was just taken
            cur_state_1_hot = encode_1_hot(cur_state)
            q_vals = model.predict(cur_state_1_hot, batch_size=1)
            q_vals[0][cur_action] = cur_action_target
            model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=0)
        else:
            next_action = rand_in_range(NUM_ACTIONS)

    cur_state = next_state
    cur_action = next_action
    return next_action

def agent_end(reward):
    global state_action_values, cur_state, cur_action, cur_epsilon, model
    if AGENT == TABULAR:
        state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward - state_action_values[cur_state[0]][cur_state[1]][cur_action])
    elif AGENT == NEURAL:
        #Update the network weights
        cur_state_1_hot = encode_1_hot(cur_state)
        q_vals = model.predict(cur_state_1_hot, batch_size=1)
        q_vals[0][cur_action] = reward
        model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=1)
    return

def agent_cleanup():
    global EPSILON, EPSILON_MIN, cur_epsilon

    #Decay epsilon at the end of the episode
    cur_epsilon = max(EPSILON_MIN, cur_epsilon - (1 / (RL_num_episodes() + 1)))
    print("Epsilon at episode end: {}".format(cur_epsilon))
    return

def agent_message(in_message):
    global EPSILON, ALPHA, GAMMA, AGENT, SEED
    params = json.loads(in_message)
    EPSILON = params["EPSILON"]
    ALPHA = params['ALPHA']
    GAMMA = params['GAMMA']
    AGENT = params['AGENT']
    return

def get_max_action(state):
    "Return the maximum action to take given the current state"

    q_vals = model.predict(encode_1_hot(state), batch_size=1)

    return np.argmax(q_vals)

def encode_1_hot(state):
    "Return a one hot encoding of the current state vector"

    state_1_hot = np.zeros((NUM_ROWS, NUM_COLUMNS))
    state_1_hot[state[0]][state[1]] = 1
    #Need to unroll the vector for input to the neural network
    return state_1_hot.reshape(1, FEATURE_VECTOR_SIZE)
