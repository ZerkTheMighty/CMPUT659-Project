#!/usr/bin/env python

from __future__ import division
from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle
import random
import json

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Concatenate
from keras.initializers import he_normal
from keras.optimizers import RMSprop

from rl_glue import RL_num_episodes

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_ROWS = 6
NUM_COLUMNS = 9
GOAL_STATE = (5, 8)

#Parameters
EPSILON = 1.0
ALPHA = None
GAMMA = None
EPSILON_MIN = None
NUM_ACTIONS = 4

#6 rows by 9 columns = 54 when unrolled
FEATURE_VECTOR_SIZE = 54
#6 rows by 9 columns * 3 states = 162
AUX_FEATURE_VECTOR_SIZE = 162

#Used for sampling in the auxiliary tasks
BUFFER_SIZE = 200

#Agents
RANDOM = 'random'
NEURAL = 'neural'
AUX = 'aux'
TABULAR = 'tabularQ'

#TODO: Refactor some of the nerual network and auxiliary task code to reduce duplication
def agent_init(random_seed):
    global state_action_values, observed_state_action_pairs, observed_states, model, cur_epsilon, replay_buffer, buffer_count

    #Reset epsilon, as we may want to decay it on a per run basis
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

        model.add(Dense(NUM_ACTIONS, kernel_initializer=init_weights))
        model.add(Activation('linear'))

        rms = RMSprop(lr=ALPHA)
        model.compile(loss='mse', optimizer=rms)

    elif AGENT == AUX:

        reward_buffer = np.empty(shape=BUFFER_SIZE)
        buffer_count = 0

        init_weights = he_normal()
        main_input = Input(shape=(FEATURE_VECTOR_SIZE,))
        aux_input = Input(shape=(AUX_FEATURE_VECTOR_SIZE,))
        merged_input = Concatenate([main_input, aux_input])

        shared1 = Dense(164, activation='relu', kernel_initializer=init_weights)(merged_input)
        shared2 = Dense(150, activation='relu', kernel_initializer=init_weights)(shared2)

        main_output = Dense(NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(shared2)
        aux_output = Dense(1, activation='linear', kernel_initializer=init_weights, name='aux_output')(shared2)

        rms = RMSprop(lr=ALPHA)
        model = Model(inputs=inputs, outputs=[main_output, aux_output])
        model.compile(optimizer=rms, loss='mse')


def agent_start(state):
    global state_action_values, cur_state, cur_action

    cur_state = state
    if AGENT == TABULAR or AGENT == RANDOM:
        #All value functions are initialized to zero, so we can just select randomly for the first action, since they all tie
        cur_action = rand_in_range(NUM_ACTIONS)
    elif AGENT == NEURAL or AGENT == AUX:
        cur_action = get_max_action(state)
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action, observed_state_action_pairs, model, cur_epsilon, replay_buffer

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

    elif AGENT == AUX:

        replay_buffer[buffer]

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            #Get the best action over all actions possible in the next state,
            (q_vals, aux_output) = model.predict(encode_1_hot(next_state), batch_size=1)[0]
            q_max = np.max(q_vals)
            next_action = np.argmax(q_vals)
            cur_action_target = reward + GAMMA * q_max

            #Get the value for the current state for which the action was just taken
            cur_state_1_hot = encode_1_hot(cur_state)
            q_vals = model.predict(cur_state_1_hot, batch_size=1)
            q_vals[cur_action] = cur_action_target
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
    global EPSILON_MIN, cur_epsilon

    #Decay epsilon at the end of the episode
    cur_epsilon = max(EPSILON_MIN, cur_epsilon - (1 / (RL_num_episodes() + 1)))
    return

def agent_message(in_message):
    global EPSILON_MIN, ALPHA, GAMMA, AGENT
    params = json.loads(in_message)
    EPSILON_MIN = params["EPSILON"]
    ALPHA = params['ALPHA']
    GAMMA = params['GAMMA']
    AGENT = params['AGENT']
    return

def get_max_action(state):
    "Return the maximum action to take given the current state"
    outputs = model.predict(encode_1_hot(state), batch_size=1)
    #exit()
    #The main output should be the first in the list of outputs returned from the call to predict
    return np.argmax(outputs[0])

# def encode_1_hot(state):
#     "Return a one hot encoding of the current state vector"
#
#     state_1_hot = np.zeros((NUM_ROWS, NUM_COLUMNS))
#     state_1_hot[state[0]][state[1]] = 1
#     #Need to unroll the vector for input to the neural network
#     #print(state_1_hot.reshape(1, FEATURE_VECTOR_SIZE))
#     #exit(1)
#     return state_1_hot.reshape(1, FEATURE_VECTOR_SIZE)

def encode_1_hot(*states):
    "Return a one hot encoding representation for the current set of states"

    #Create an n-ndimensional array, where n = num_states * number of entries in each raw state vector (which is two, 1 for each row and column)
    #The row and column sizes determine the size of each dimension
    dimension_shapes = []
    for i in range(len(states)):
        dimension_shapes.extend([NUM_ROWS, NUM_COLUMNS])
    state_1_hot = np.zeros(tuple(dimension_shapes))

    #Construct the indices to index into the n dimensional array
    num_dimensions = len(states) * len(states[0])
    indices = [slice(None) for i in range(num_dimensions)]
    #print(states)
    i = 0
    for state in states:
        indices[i] = state[0]
        indices[i + 1] = state[1]
        i += 2

    #Index into the array n times to set the 1 hot digit of the vector which corresponds to the current set of states
    #print(indices)
    state_1_hot[tuple(indices)] = 1

    #Need to unroll the vector for input to the neural network
    #print(state_1_hot.reshape(1, FEATURE_VECTOR_SIZE))
    #exit(1)
    return state_1_hot.reshape(1, FEATURE_VECTOR_SIZE)
