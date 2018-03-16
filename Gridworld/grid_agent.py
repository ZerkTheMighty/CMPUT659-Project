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
from keras.initializers import lecun_uniform
from keras.optimizers import SGD

from rl_glue import RL_num_episodes

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_ROWS = 6
NUM_COLUMNS = 9
GOAL_STATE = (5, 8)

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

#TODO: replaceall of the regular tuples with named tuples to improve readability
def agent_init(random_seed):
    global state_action_values, observed_state_action_pairs, observed_states, model, num_steps, cur_epsilon

    #Different parts of the program use np.random (via utils.py) and others use just random,
    #seeding both with the same seed here to make sure they both start in the same place per run of the program
    np.random.seed(random_seed)
    random.seed(random_seed)

    #Reset epsilon, as we may want to decay it per episode
    cur_epsilon = EPSILON
    print("epsilon start: {}".format(cur_epsilon))

    if AGENT == TABULAR:
        #The real world estimates for each state action pair
        state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]
    elif AGENT == NEURAL:
        num_steps = 0
        #The row, column, and action comprise the state vector
        input_layer_size = 3

        #Initialize the neural network
        model = Sequential()
        init_weights = lecun_uniform(random_seed)
        model.add(Dense(164, kernel_initializer=init_weights, input_shape=(input_layer_size,)))
        model.add(Activation('sigmoid'))

        model.add(Dense(150, kernel_initializer=init_weights))
        model.add(Activation('sigmoid'))

        model.add(Dense(1, kernel_initializer=init_weights))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        sgd = SGD(lr=ALPHA, momentum=0., decay=0., nesterov=False)
        model.compile(loss='mse', optimizer=sgd)


def agent_start(state):
    global state_action_values, cur_state, cur_action

    cur_state = state
    if AGENT == TABULAR or AGENT == RANDOM:
        #All value functions are initialized to zero, so we can just select randomly for the first action, since they all tie
        cur_action = rand_in_range(NUM_ACTIONS)
    elif AGENT == NEURAL:
        cur_action = get_max_action_val(model, state)[0]
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

            #Update the network weights
            (next_action, max_action_val) = get_max_action_val(model, next_state)
            cur_update_target = reward + GAMMA * max_action_val
            cur_input = np.array(cur_state + [cur_action]).reshape(1, 3)
            model.fit(cur_input, cur_update_target, batch_size=1, epochs=1, verbose=0)
            #print("Next action: {}".format(next_action))
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
        cur_input = np.array(cur_state + [cur_action]).reshape(1, 3)
        cur_update_target = np.array(reward).reshape(1, 1)
        model.fit(cur_input, cur_update_target, batch_size=1, epochs=1, verbose=0)
    #Decay epsilon at the end of the episode
    if cur_epsilon > EPSILON_MIN:
        cur_epsilon -= 1 / (RL_num_episodes() + 1)
        print("cur epsilon: {}".format(cur_epsilon))
    return

def agent_cleanup():
    return

def agent_message(in_message):
    global EPSILON, ALPHA, GAMMA, AGENT, SEED
    params = json.loads(in_message)
    EPSILON = params["EPSILON"]
    ALPHA = params['ALPHA']
    GAMMA = params['GAMMA']
    AGENT = params['AGENT']
    return

def get_max_action_val(approximator, state):
    "Returns a tuple composed of the maximum action and its value"
    #print(np.array(state + [0]).reshape(1, 3))
    #print(approximator.predict(np.array(state + [0]).reshape(1, 3)))

    #state vector should be 1 X 3: 1 row and 3 columns, 1 for each of the row, column and action of the feature vector we want to use for the neural net
    cur_state_action_values = [approximator.predict(np.array(state + [action]).reshape(1, 3)) for action in range(NUM_ACTIONS)]
    #print(cur_state_action_values)
    #print(max(cur_state_action_values))
    #print(cur_state_action_values.index(max(cur_state_action_values)))
    return (cur_state_action_values.index(max(cur_state_action_values)), max(cur_state_action_values))

def get_action_val(approximator, state, action):
    "Returns the state action value for the given state and action using approximator"

    return approximator.predict(np.array(state + [action]).reshape(1, 3))
