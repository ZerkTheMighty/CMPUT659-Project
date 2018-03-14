#!/usr/bin/env python

from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle
import random
import json

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_ROWS = 6
NUM_COLUMNS = 9
GOAL_STATE = (5, 8)

EPSILON = None
ALPHA = None
GAMMA = None
NUM_ACTIONS = 4
N = None

def agent_init():
    global state_action_values, observed_state_action_pairs, observed_states, model

    #The real world estimates for each state action pair
    state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]

    #The model values for each state action pair: model[row][column][action] yields a 2-element list [reward, next_state]
    model = [[[[] for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]

    observed_states = set()
    observed_state_action_pairs = set()

def agent_start(state):
    global state_action_values, cur_state, cur_action
    cur_state = state
    #All value functions are initialized to zero, so we can just select randomly for the first action, since they all tie
    cur_action = rand_in_range(NUM_ACTIONS)
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action, observed_state_action_pairs, model

    next_state = state
    #Choose the next action, epsilon greedy style
    if rand_un() < 1 - EPSILON:
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

    #Update the model
    cur_state_action_pair = (tuple(cur_state), cur_action)
    if cur_state_action_pair not in observed_state_action_pairs:
        observed_state_action_pairs.add(cur_state_action_pair)
        observed_states.add(tuple(cur_state))

        model[cur_state[0]][cur_state[1]][cur_action].append(reward)
        model[cur_state[0]][cur_state[1]][cur_action].append(next_state)

    #Do planning
    for step in range(N):
        #Choose a previously observed state at random
        simulated_state = random.choice(tuple(observed_states))
        #Choose a previously observed action that corresponds to the chosen random state at random
        simulated_state_action_pairs = [state_action_pair for state_action_pair in observed_state_action_pairs if state_action_pair[0] == simulated_state]
        simulated_action = random.choice(tuple(simulated_state_action_pairs))[1]

        #Consult the model
        model_reward = model[simulated_state[0]][simulated_state[1]][simulated_action][0]
        model_next_state = model[simulated_state[0]][simulated_state[1]][simulated_action][1]
        simulated_max_next_action = state_action_values[model_next_state[0]][model_next_state[1]].index(max(state_action_values[model_next_state[0]][model_next_state[1]]))

        #Update the action value
        state_action_values[simulated_state[0]][simulated_state[1]][simulated_action] += ALPHA * (model_reward + GAMMA * state_action_values[model_next_state[0]][model_next_state[1]][simulated_max_next_action] - state_action_values[simulated_state[0]][simulated_state[1]][simulated_action])

    cur_state = next_state
    cur_action = next_action
    return next_action

def agent_end(reward):
    global state_action_values, cur_state, cur_action
    state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward + 0 - state_action_values[cur_state[0]][cur_state[1]][cur_action])
    return

def agent_cleanup():
    """
    This function is not used
    """
    #print(state_action_values)
    return

def agent_message(in_message): # returns string, in_message: string
    global EPSILON, ALPHA, GAMMA, N
    params = json.loads(in_message)
    EPSILON = params["EPSILON"]
    ALPHA = params['ALPHA']
    GAMMA = params['GAMMA']
    N = params['N']
    return
