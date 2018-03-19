#!/usr/bin/env python

from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle
import json

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import he_normal
from keras.optimizers import RMSprop

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_ROWS = 7
NUM_COLUMNS = 10
GOAL_STATE = (3, 7)

EPSILON = None
ALPHA = None
NUM_ACTIONS = None
GAMMA = None

#Agents
RANDOM = 'random'
NEURAL = 'neural'
AUX = 'aux'
TABULAR = 'tabularQ'

state_action_values, policy = (None, None)

def agent_init():
    global state_action_values

    if AGENT == TABULAR:
        state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]
        state_action_values[GOAL_STATE[0]][GOAL_STATE[1]] = 0
    elif AGENT == NEURAL:
        #Initialize the neural network
        model = Sequential()
        init_weights = he_normal()

        model.add(Dense(164, kernel_initializer=init_weights, input_shape=(FEATURE_VECTOR_SIZE,)))
        model.add(Activation('relu'))

        model.add(Dense(150, kernel_initializer=init_weights))
        model.add(Activation('relu'))

        model.add(Dense(4, kernel_initializer=init_weights))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

def agent_start(state):
    global state_action_values, cur_state, cur_action

    if AGENT == TABULAR:
        #All value functions are initialized to zero, so we can just select randomly for the first action, since they all tie
        cur_action = rand_in_range(NUM_ACTIONS)
    elif AGENT == NEURAL:
        cur_action = get_max_action(state)
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action

    next_state = state
    if AGENT == TABULAR:
        #Choose the next action, epsilon greedy style
        cur_state = state
        if rand_un() < 1 - EPSILON:
            next_action = state_action_values[state[0]][state[1]].index(max(state_action_values[state[0]][state[1]]))
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Update the state action values
        next_state_max_action = state_action_values[next_state[0]][next_state[1]].index(max(state_action_values[next_state[0]][next_state[1]]))
        state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward + GAMMA * state_action_values[next_state[0]][next_state[1]][next_state_max_action] - state_action_values[cur_state[0]][cur_state[1]][cur_action])

        #print(state_action_values)
    elif AGENT == NEURAL:
        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - EPSILON:

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

    #print(AGENT)
    #print(next_action)
    cur_state = next_state
    cur_action = next_action
    return next_action

def agent_end(reward):
    global state_action_values, cur_state, cur_action
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
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global EPSILON, ALPHA, GAMMA, NUM_ACTIONS, AGENT
    params = json.loads(in_message)
    EPSILON = params["EPSILON"]
    ALPHA = params['ALPHA']
    NUM_ACTIONS = params['NUM_ACTIONS']
    AGENT = params['AGENT']
    GAMMA = params['GAMMA']

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
