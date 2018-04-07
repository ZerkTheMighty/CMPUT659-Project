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
OBSTACLE_STATES = [[2, 2], [3, 2], [4, 2], [1, 5], [3, 7], [4, 7], [5, 7]]

#Parameters
EPSILON = 1.0
ALPHA = None
GAMMA = None
EPSILON_MIN = None
N = None
IS_STOCHASTIC = None
NUM_ACTIONS = 4

FEATURE_VECTOR_SIZE = NUM_ROWS * NUM_COLUMNS
AUX_FEATURE_VECTOR_SIZE = NUM_ROWS * NUM_COLUMNS * NUM_ACTIONS

#Used for sampling in the auxiliary tasks
BUFFER_SIZE = 10

#Number of output nodes used in the noisy and redundant auxiliary tasks, respectively
NUM_NOISE_NODES = 10
NUM_REDUNDANT_TASKS = 4

#The number of times to run the auxiliary task during a single time step
SAMPLES_PER_STEP = 1

#Agents: non auxiliary task based
RANDOM = 'random'
NEURAL = 'neural'
TABULAR = 'tabularQ'

#Agents: auxiliary task based
REWARD = 'reward'
STATE = 'state'
REDUNDANT = 'redundant'
NOISE = 'noise'


#TODO: Refactor some of the neural network and auxiliary task code to reduce duplication
#TODO: Refactor the update exeprience replay code to reduce duplication
#TODO: Look into making how globals are used more consistent, sometimes they are passed into local functions and sometimes they are just declared global in those functions

def agent_init():
    global state_action_values, observed_state_action_pairs, observed_states, model, cur_epsilon, zero_reward_buffer, zero_buffer_count, non_zero_reward_buffer, non_zero_buffer_count, deterministic_state_buffer, deterministic_state_buffer_count, stochastic_state_buffer, stochastic_state_buffer_count

    #Reset epsilon, as we want to decay it on a per run basis
    cur_epsilon = EPSILON
    print("Epsilon at run start: {}".format(cur_epsilon))

    if AGENT == RANDOM:
        pass

    elif AGENT == TABULAR:
        #The real world estimates for each state action pair
        state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]
    elif AGENT == NEURAL:

        #Initialize the neural network
        model = Sequential()
        init_weights = he_normal()

        model.add(Dense(164, activation='relu', kernel_initializer=init_weights, input_shape=(FEATURE_VECTOR_SIZE,)))
        model.add(Dense(150, activation='relu', kernel_initializer=init_weights))
        model.add(Dense(NUM_ACTIONS, activation='linear', kernel_initializer=init_weights))

        rms = RMSprop(lr=ALPHA)
        model.compile(loss='mse', optimizer=rms)

    else:

        #Initialize the replay buffers for use by the auxiliary prediction tasks
        non_zero_reward_buffer = []
        zero_reward_buffer = []
        non_zero_buffer_count = 0
        zero_buffer_count = 0

        deterministic_state_buffer = []
        stochastic_state_buffer = []
        deterministic_state_buffer_count = 0
        stochastic_state_buffer_count = 0

        if AGENT == REWARD:
            num_outputs = 1
            cur_activation = 'sigmoid'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'binary_crossentropy'}

        elif AGENT == NOISE:
            num_outputs = NUM_NOISE_NODES
            cur_activation = 'linear'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        elif AGENT == STATE:
            num_outputs = FEATURE_VECTOR_SIZE
            cur_activation = 'softmax'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'categorical_crossentropy'}

        elif AGENT == REDUNDANT:
            num_outputs = NUM_ACTIONS * NUM_REDUNDANT_TASKS
            cur_activation = 'linear'
            loss={'main_output': 'mean_squared_error', 'aux_output': 'mean_squared_error'}

        init_weights = he_normal()
        main_input = Input(shape=(FEATURE_VECTOR_SIZE,))

        shared_1 = Dense(164, activation='relu', kernel_initializer=init_weights)(main_input)
        shared_2 = Dense(150, activation='relu', kernel_initializer=init_weights)(shared_1)

        main_output = Dense(NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(shared_2)

        if AGENT == REDUNDANT:
            aux_input = Input(shape=(FEATURE_VECTOR_SIZE,))
        else:
            aux_input = Input(shape=(AUX_FEATURE_VECTOR_SIZE * N,))
        merged = concatenate([aux_input, shared_2])

        aux_1 = Dense(128, activation='relu', kernel_initializer=init_weights)(merged)

        aux_output = Dense(num_outputs, activation=cur_activation, kernel_initializer=init_weights, name='aux_output')(aux_1)
        rms = RMSprop(lr=ALPHA)
        model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
        model.compile(optimizer=rms, loss=loss)


def agent_start(state):
    global state_action_values, cur_state, cur_action, cur_context, cur_context_actions

    #Context is a sliding window of the previous n states that gets added to the replay buffer used by auxiliary tasks
    cur_context = []
    cur_context_actions = []
    cur_state = state

    if rand_un() < 1 - cur_epsilon:
        if AGENT == TABULAR:
            cur_action = get_max_action_tabular(cur_state)
        elif AGENT == NEURAL:
            cur_action = get_max_action(cur_state)
        elif AGENT == RANDOM:
            cur_action = rand_in_range(NUM_ACTIONS)
        else:
            cur_action = get_max_action_aux(cur_state)
    else:
        cur_action = rand_in_range(NUM_ACTIONS)
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action, cur_epsilon, zero_reward_buffer, zero_buffer_count, non_zero_reward_buffer, non_zero_buffer_count, deterministic_state_buffer, deterministic_state_buffer_count, stochastic_state_buffer, stochastic_state_buffer_count

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
        q_vals = model.predict(state_encode_1_hot([next_state]), batch_size=1)
        q_max = np.max(q_vals)
        cur_action_target = reward + GAMMA * q_max

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            next_action = np.argmax(q_vals)
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Get the value for the current state of the action which was just taken ie Q(S, A)
        cur_state_1_hot = state_encode_1_hot([cur_state])
        q_vals = model.predict(cur_state_1_hot, batch_size=1)
        q_vals[0][cur_action] = cur_action_target

        #Update the weights
        model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=0)

    elif AGENT == RANDOM:
        next_action = rand_in_range(NUM_ACTIONS)

    #All auxiliary tasks
    else:

        update_replay_buffer(cur_state, cur_action, reward, next_state)

        if AGENT == REDUNDANT:
            aux_dummy = np.zeros(shape=(1, FEATURE_VECTOR_SIZE,))
        else:
            aux_dummy = np.zeros(shape=(1, AUX_FEATURE_VECTOR_SIZE * N,))

        #Get the best action over all actions possible in the next state, ie max_a(Q, a)
        q_vals, _ = model.predict([state_encode_1_hot([next_state]), aux_dummy], batch_size=1)
        q_max = np.max(q_vals)
        cur_action_target = reward + GAMMA * q_max

        #Choose the next action, epsilon greedy style
        if rand_un() < 1 - cur_epsilon:
            next_action = np.argmax(q_vals)
        else:
            next_action = rand_in_range(NUM_ACTIONS)

        #Get the appropriate q-value for the current state
        cur_state_1_hot = state_encode_1_hot([cur_state])
        q_vals, _ = model.predict([cur_state_1_hot, aux_dummy], batch_size=1)
        q_vals[0][cur_action] = cur_action_target

        #Sample a transition from the replay buffer to use for auxiliary task training
        cur_transition = None
        if zero_reward_buffer and non_zero_reward_buffer and AGENT != STATE:
            cur_transition = sample_from_buffers(zero_reward_buffer, non_zero_reward_buffer)
        elif AGENT == STATE and IS_STOCHASTIC:
            if deterministic_state_buffer and stochastic_state_buffer:
                cur_transition = sample_from_buffers(deterministic_state_buffer, stochastic_state_buffer)
        elif AGENT == STATE and not IS_STOCHASTIC:
            if deterministic_state_buffer:
                cur_transition = sample_from_buffers(deterministic_state_buffer)

        #Update the current q-value and auxiliary task output towards their respective targets
        if cur_transition is not None:
            #Set the auxiliary input depending on the task
            if AGENT == REDUNDANT:
                aux_input = cur_state_1_hot
            else:
                 aux_input = encode_1_hot(cur_transition.states, cur_transition.actions)

            if AGENT == REWARD:
                #We make the rewards positive since we care only about the binary
                #distinction between zero and non zero rewards and theano binary
                #cross entropy loss requires targets to be 0 or 1
                aux_target = np.array([abs(cur_transition.reward)])
                pred_q, pred_reward = model.predict([cur_state_1_hot, aux_input])
            elif AGENT == STATE:
                aux_target = state_encode_1_hot([cur_transition.next_state])
                pred_q, pred_reward = model.predict([cur_state_1_hot, aux_input])
            elif AGENT == NOISE:
                aux_target = np.array([rand_un() for i in range(NUM_NOISE_NODES)]).reshape(1, NUM_NOISE_NODES)
            elif AGENT == REDUNDANT:
                nested_q_vals = [q_vals for i in range(NUM_REDUNDANT_TASKS)]
                aux_target = np.array([item for sublist in nested_q_vals for item in sublist]).reshape(1, NUM_ACTIONS * NUM_REDUNDANT_TASKS)
            # print('vals')
            # print(q_vals)
            # print('au')
            # print(aux_target)
            model.fit([cur_state_1_hot, aux_input], [q_vals, aux_target], batch_size=1, epochs=1, verbose=0)


    cur_state = next_state
    cur_action = next_action
    return next_action

def agent_end(reward):
    global state_action_values, cur_state, cur_action, cur_epsilon, model, zero_reward_buffer, zero_buffer_count, non_zero_reward_buffer, non_zero_buffer_count, deterministic_state_buffer, deterministic_state_buffer_count, stochastic_state_buffer, stochastic_state_buffer_count
    if AGENT == TABULAR:
        state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward - state_action_values[cur_state[0]][cur_state[1]][cur_action])
    elif AGENT == NEURAL:
        #Update the network weights
        cur_state_1_hot = state_encode_1_hot([cur_state])
        q_vals = model.predict(cur_state_1_hot, batch_size=1)
        q_vals[0][cur_action] = reward
        model.fit(cur_state_1_hot, q_vals, batch_size=1, epochs=1, verbose=1)

    elif AGENT == RANDOM:
        pass

    #All auxiliary tasks
    else:
        update_replay_buffer(cur_state, cur_action, reward, GOAL_STATE)

        if AGENT == REDUNDANT:
            aux_dummy = np.zeros(shape=(1, FEATURE_VECTOR_SIZE,))
        else:
            aux_dummy = np.zeros(shape=(1, AUX_FEATURE_VECTOR_SIZE * N,))

        #Get the best action over all actions possible in the next state, ie max_a(Q, a)
        cur_state_1_hot = state_encode_1_hot([cur_state])
        q_vals, _ = model.predict([cur_state_1_hot, aux_dummy], batch_size=1)
        q_vals[0][cur_action] = reward = reward

        #Sample a transition from the replay buffer to use for auxiliary task training
        cur_transition = None
        if zero_reward_buffer and non_zero_reward_buffer and AGENT != STATE:
            cur_transition = sample_from_buffers(zero_reward_buffer, non_zero_reward_buffer)
        elif AGENT == STATE and IS_STOCHASTIC:
            if deterministic_state_buffer and stochastic_state_buffer:
                cur_transition = sample_from_buffers(deterministic_state_buffer, stochastic_state_buffer)
        elif AGENT == STATE and not IS_STOCHASTIC:
            if deterministic_state_buffer:
                cur_transition = sample_from_buffers(deterministic_state_buffer)

        #Update the current q-value and auxiliary task output towards their respective targets
        if cur_transition is not None:
            #Set the auxiliary input depending on the task
            if AGENT == REDUNDANT:
                aux_input = cur_state_1_hot
            else:
                aux_input = encode_1_hot(cur_transition.states, cur_transition.actions)

            if AGENT == REWARD:
                #We make the rewards positive since we care only about the binary
                #distinction between zero and non zero rewards and theano binary
                #cross entropy loss requires targets to be 0 or 1
                aux_target = np.array([abs(cur_transition.reward)])
                pred_q, pred_reward = model.predict([cur_state_1_hot, aux_input])
            elif AGENT == STATE:
                aux_target = state_encode_1_hot([cur_transition.next_state])
                pred_q, pred_reward = model.predict([cur_state_1_hot, aux_input])
            elif AGENT == NOISE:
                aux_target = np.array([rand_un() for i in range(NUM_NOISE_NODES)]).reshape(1, NUM_NOISE_NODES)
            elif AGENT == REDUNDANT:
                nested_q_vals = [q_vals for i in range(NUM_REDUNDANT_TASKS)]
                aux_target = np.array([item for sublist in nested_q_vals for item in sublist]).reshape(1, NUM_ACTIONS * NUM_REDUNDANT_TASKS)
            model.fit([cur_state_1_hot, aux_input], [q_vals, aux_target], batch_size=1, epochs=1, verbose=1)
    return

def agent_cleanup():
    global EPSILON_MIN, cur_epsilon

    #Decay epsilon at the end of the episode
    cur_epsilon = max(EPSILON_MIN, cur_epsilon - (1 / (RL_num_episodes() + 1)))
    return

def agent_message(in_message):
    global EPSILON_MIN, ALPHA, GAMMA, AGENT, N, IS_STOCHASTIC
    params = json.loads(in_message)
    EPSILON_MIN = params["EPSILON"]
    ALPHA = params['ALPHA']
    GAMMA = params['GAMMA']
    AGENT = params['AGENT']
    N = params['N']
    IS_STOCHASTIC = params['IS_STOCHASTIC']
    return

def get_max_action(state):
    "Return the maximum action to take given the current state"

    q_vals = model.predict(state_encode_1_hot([state]), batch_size=1)
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

    if AGENT == REDUNDANT:
        aux_dummy = np.zeros(shape=(1, FEATURE_VECTOR_SIZE,))
    else:
        aux_dummy = np.zeros(shape=(1, AUX_FEATURE_VECTOR_SIZE * N,))
    q_vals, _ = model.predict([state_encode_1_hot([state]), aux_dummy], batch_size=1)

    return np.argmax(q_vals[0])

def state_encode_1_hot(states):
    "Return a one hot encoding of the current list of states"

    all_states_1_hot = []
    for state in states:
        state_1_hot = np.zeros((NUM_ROWS, NUM_COLUMNS))
        state_1_hot[state[0]][state[1]] = 1
        state_1_hot = state_1_hot.reshape(1, FEATURE_VECTOR_SIZE)
        all_states_1_hot.append(state_1_hot)

    return np.concatenate(all_states_1_hot, 1)

def encode_1_hot(states, actions):
    "Return a 1 hot encoding of the current list of states and the accompanying actions"

    all_states_1_hot = []
    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        state_1_hot = np.zeros((NUM_ROWS, NUM_COLUMNS, NUM_ACTIONS))
        state_1_hot[state[0]][state[1]][action] = 1
        state_1_hot = state_1_hot.reshape(1, AUX_FEATURE_VECTOR_SIZE)
        all_states_1_hot.append(state_1_hot)

    return np.concatenate(all_states_1_hot, 1)

def update_replay_buffer(cur_state, cur_action, reward, next_state):
    global cur_context, cur_context_actions, zero_reward_buffer, non_zero_reward_buffer, zero_buffer_count, non_zero_buffer_count, deterministic_state_buffer, deterministic_state_buffer_count, stochastic_state_buffer, stochastic_state_buffer_count
    """
    Update the replay buffer with the most recent transition, adding cur_state to the current global historical context,
    and mapping that to reward and next_state if the current context == N, the user set parameter for the context size
    """

    #Construct the historical context used in the prediciton tasks, and store them in the replay buffer according to their reward valence
    cur_context.append(cur_state)
    cur_context_actions.append(cur_action)
    cur_transition = None
    if len(cur_context) == N:
        cur_transition = namedtuple("Transition", ["states", "actions", "reward", "next_state"])
        cur_transition.states = list(cur_context)
        cur_transition.reward = reward
        cur_transition.next_state = next_state
        cur_transition.actions = list(cur_context_actions)
        cur_context.pop(0)
        cur_context_actions.pop(0)

    if cur_transition is not None:
        if AGENT == STATE:
            if  cur_transition.states[-1] in OBSTACLE_STATES:
                add_to_buffer(stochastic_state_buffer, cur_transition, stochastic_state_buffer_count)
                stochastic_state_buffer_count += 1
                if stochastic_state_buffer_count == BUFFER_SIZE:
                    stochastic_state_buffer_count = 0
            else:
                add_to_buffer(deterministic_state_buffer, cur_transition, deterministic_state_buffer_count)
                deterministic_state_buffer_count += 1
                if deterministic_state_buffer_count == BUFFER_SIZE:
                    deterministic_state_buffer_coount = 0
        else:
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
    """
    Add item to_add to cur_buffer at index buffer_count, otherwise append it to
    the end of the buffer in the case of buffer overflow
    """

    try:
        cur_buffer[buffer_count] = to_add
    except IndexError:
        cur_buffer.append(to_add)

def sample_from_buffers(buffer_one, buffer_two=None):
    """
    Sample a transiton uniformly at random from one of buffer_one and buffer_two.
    Which buffer is sampled is dependent on the current time step, and done in a
    way so as to sample equally from both buffers throughout an episode"
    """
    if RL_num_steps() % 2 == 0 or buffer_two is None:
        cur_transition = buffer_one[rand_in_range(len(buffer_one))]
    else:
        cur_transition = buffer_two[rand_in_range(len(buffer_two))]
    return cur_transition
