#!/usr/bin/env python

from __future__ import division
from collections import namedtuple
from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle
import random
import json

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, concatenate
from keras.initializers import he_normal
from keras.optimizers import RMSprop

from rl_glue import RL_num_episodes, RL_num_steps

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
N = None
NUM_ACTIONS = 4

#6 rows by 9 columns = 54 when unrolled
FEATURE_VECTOR_SIZE = 54
#6 ** 3 * 9 ** 3 =
AUX_FEATURE_VECTOR_SIZE = 157464

#Used for sampling in the auxiliary tasks
BUFFER_SIZE = 10

#Agents
RANDOM = 'random'
NEURAL = 'neural'
AUX = 'aux'
TABULAR = 'tabularQ'

#TODO: Actually Incorporate the auxiliary task ang get results
#TODO: Refactor some of the neural network and auxiliary task code to reduce duplication
#TODO: Look into replacing the state vector with a named tuple for rows and columns to make things more readable
#TODO: Look into making how globals are used more consistent, sometimes thye are passed into local functions and sometimes they are just declared global in those functions

def agent_init(random_seed):
    global state_action_values, observed_state_action_pairs, observed_states, model, cur_epsilon, zero_reward_buffer, zero_buffer_count, non_zero_reward_buffer, non_zero_buffer_count

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

        #Initialize the replay buffer for use by the auxiliary prediction tasks
        non_zero_reward_buffer = []
        zero_reward_buffer = []
        non_zero_buffer_count = 0
        zero_buffer_count = 0

        init_weights = he_normal()
        main_input = Input(shape=(FEATURE_VECTOR_SIZE,))
        aux_input = Input(shape=(AUX_FEATURE_VECTOR_SIZE,))
        merged_input = concatenate([main_input, aux_input])

        shared1 = Dense(164, activation='relu', kernel_initializer=init_weights)(merged_input)
        shared2 = Dense(150, activation='relu', kernel_initializer=init_weights)(shared1)

        main_output = Dense(NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(shared2)
        aux_output = Dense(1, activation='linear', kernel_initializer=init_weights, name='aux_output')(shared2)

        rms = RMSprop(lr=ALPHA)
        model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
        model.compile(optimizer=rms, loss='mse')


def agent_start(state):
    global state_action_values, cur_state, cur_action, cur_context

    #Context is a sliding window of the previous n states that gets added to the replay buffer used by auxiliary tasks
    cur_context = []
    cur_state = state
    if rand_un() < 1 - cur_epsilon:
        if AGENT == TABULAR:
            cur_action = get_max_action_tabular(cur_state)
        elif AGENT == NEURAL:
            cur_action = get_max_action(cur_state)
        elif AGENT == AUX:
            cur_action = get_max_action_aux(cur_state)
        else:
            cur_action = rand_in_range(NUM_ACTIONS)
    else:
        cur_action = rand_in_range(NUM_ACTIONS)
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action, cur_epsilon, zero_reward_buffer, zero_buffer_count, non_zero_reward_buffer, non_zero_buffer_count

    next_state = state

    if AGENT == TABULAR:
        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            next_action = get_max_action_tabular(next_state)
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Update the state action values
        next_state_max_action = state_action_values[next_state[0]][next_state[1]].index(max(state_action_values[next_state[0]][next_state[1]]))
        state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward + GAMMA * state_action_values[next_state[0]][next_state[1]][next_state_max_action] - state_action_values[cur_state[0]][cur_state[1]][cur_action])

    elif AGENT == NEURAL:
        #Get the best action over all actions possible in the next state, ie max_a(Q, a)
        q_vals = model.predict(encode_1_hot([next_state]), batch_size=1)
        q_max = np.max(q_vals)
        cur_action_target = reward + GAMMA * q_max

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            next_action = np.argmax(q_vals)
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Get the value for the current state of the action which was just taken ie Q(S, A)
        cur_state_1_hot = encode_1_hot([cur_state])
        q_vals = model.predict(cur_state_1_hot, batch_size=1)
        q_vals[0][cur_action] = cur_action_target

        #Update the weights
        model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=0)

    elif AGENT == AUX:

        update_replay_buffer(cur_state, reward, next_state)
        #print(zero_reward_buffer)
        #print(non_zero_reward_buffer)

        #Get the best action over all actions possible in the next state, ie max_a(Q, a)
        aux_dummy = np.zeros(shape=(1, AUX_FEATURE_VECTOR_SIZE,))
        q_vals, _ = model.predict([encode_1_hot([next_state]), aux_dummy], batch_size=1)
        q_max = np.max(q_vals)
        cur_action_target = reward + GAMMA * q_max

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            next_action = np.argmax(q_vals)
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Get the appropriate q-value for the current state
        cur_state_1_hot = encode_1_hot([cur_state])
        q_vals, _ = model.predict([cur_state_1_hot, aux_dummy], batch_size=1)
        q_vals[0][cur_action] = cur_action_target

        #Sample a transition from the replay buffer to use for auxiliary task training
        if zero_reward_buffer and non_zero_reward_buffer:
            if RL_num_steps() % 2 == 0:
                cur_transition = zero_reward_buffer[rand_in_range(len(zero_reward_buffer))]
            else:
                cur_transition = non_zero_reward_buffer[rand_in_range(len(non_zero_reward_buffer))]
            cur_context_1_hot = encode_1_hot(cur_transition.states)

            #Update the current q-value and auxiliary task output towards their respective targets
            model.fit([cur_state_1_hot, cur_context_1_hot], [q_vals, np.array([cur_transition.reward])], batch_size=1, epochs=1, verbose=0)
        else:
            #Update the weights
            #model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=0)
            pass
        #print(RL_num_steps())

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
        cur_state_1_hot = encode_1_hot([cur_state])
        q_vals = model.predict(cur_state_1_hot, batch_size=1)
        q_vals[0][cur_action] = reward
        model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=1)

    elif AGENT == AUX:
        update_replay_buffer(cur_state, reward, None)

        #Get the best action over all actions possible in the next state, ie max_a(Q, a)
        cur_state_1_hot = encode_1_hot([cur_state])
        aux_dummy = np.zeros(shape=(1, AUX_FEATURE_VECTOR_SIZE,))
        q_vals, _ = model.predict([cur_state_1_hot, aux_dummy], batch_size=1)
        q_vals[0][cur_action] = reward = reward

        #Sample a transition from the replay buffer to use for auxiliary task training
        if zero_reward_buffer and non_zero_reward_buffer:
            if RL_num_steps() % 2 == 0:
                cur_transition = zero_reward_buffer[rand_in_range(len(zero_reward_buffer))]
            else:
                cur_transition = non_zero_reward_buffer[rand_in_range(len(non_zero_reward_buffer))]
            cur_context_1_hot = encode_1_hot(cur_transition.states)
            #Update the current q-value and auxiliary task output towards their respective targets
            model.fit([cur_state_1_hot, cur_context_1_hot], [q_vals, np.array([cur_transition.reward])], batch_size=1, epochs=1, verbose=1)
        else:
            #Update the weights
            #model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=0)
            pass
    return

def agent_cleanup():
    global EPSILON_MIN, cur_epsilon

    #Decay epsilon at the end of the episode
    cur_epsilon = max(EPSILON_MIN, cur_epsilon - (1 / (RL_num_episodes() + 1)))
    return

def agent_message(in_message):
    global EPSILON_MIN, ALPHA, GAMMA, AGENT, N
    params = json.loads(in_message)
    EPSILON_MIN = params["EPSILON"]
    ALPHA = params['ALPHA']
    GAMMA = params['GAMMA']
    AGENT = params['AGENT']
    N = params['N']
    return

def get_max_action(state):
    "Return the maximum action to take given the current state"

    q_vals = model.predict(encode_1_hot([state]), batch_size=1)
    return np.argmax(q_vals[0])

def get_max_action_tabular(state):
    global state_action_values
    "Return the maximum action to take given the current state."

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
    return next_action

def get_max_action_aux(state):
    "Return the maximum acton to take given the current state"

    dummy_aux = np.zeros(shape=(1, AUX_FEATURE_VECTOR_SIZE))
    q_vals, _ = model.predict([encode_1_hot([state]), dummy_aux], batch_size=1)

    return np.argmax(q_vals[0])

def encode_1_hot(states):
    "Return a one hot encoding representation for the current list of states"

    #Create an n-ndimensional array, where n = num_states * number of entries in each raw state vector (which is two, 1 for each row and column)
    #The row and column sizes determine the size of each dimension
    dimension_shapes = []
    for i in range(len(states)):
        dimension_shapes.extend([NUM_ROWS, NUM_COLUMNS])
    state_1_hot = np.zeros(shape=tuple(dimension_shapes))

    #Construct the indices to index into the n dimensional array
    num_dimensions = len(states) * len(states[0])
    indices = [slice(None) for i in range(num_dimensions)]
    i = 0
    for state in states:
        indices[i] = state[0]
        indices[i + 1] = state[1]
        i += 2

    #Index into the array n times to set the 1 hot digit of the vector which corresponds to the current set of states
    state_1_hot[tuple(indices)] = 1

    #Need to unroll the vector for input to the neural network
    if len(states) == N:
        return state_1_hot.reshape(1 ,AUX_FEATURE_VECTOR_SIZE)
    return state_1_hot.reshape(1, FEATURE_VECTOR_SIZE)

def update_replay_buffer(cur_state, reward, next_state):
    global cur_context, zero_reward_buffer, non_zero_reward_buffer, zero_buffer_count, non_zero_buffer_count
    """
    Update the replay buffer with the most recent transition, adding cur_state to the current global historical context,
    and mapping that to reward and next_state if the current context == N, the user set parameter for the context size
    """

    #Construct the historical context used in the prediciton tasks, and store them in the replay buffer according to their reward valence
    cur_context.append(cur_state)
    cur_transition = None
    if len(cur_context) == N:
        cur_transition = namedtuple("Transition", ["states", "reward", "next_state"])
        cur_transition.states = list(cur_context)
        cur_transition.reward = reward
        cur_transition.next_state = next_state
        cur_context.pop(0)

    if cur_transition is not None:
        if reward == 0:
            add_to_buffer(zero_reward_buffer, cur_transition, zero_buffer_count)
            zero_buffer_count += 1
            if zero_buffer_count == BUFFER_SIZE:
                zero_buffer_count = 0
        else:
            add_to_buffer(non_zero_reward_buffer, cur_transition, non_zero_buffer_count)
            non_zero_buffer_count += 1
            if non_zero_buffer_count == BUFFER_SIZE:
                non_zero_buffer_count = 0

def add_to_buffer(cur_buffer, to_add, buffer_count):
    "Add item to_add to cur_buffer at index buffer_count, otherwise append it to the end of the buffer in the case of buffer overflow"

    try:
        cur_buffer[buffer_count] = to_add
    except IndexError:
        cur_buffer.append(to_add)
