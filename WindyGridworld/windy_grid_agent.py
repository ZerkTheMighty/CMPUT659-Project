#!/usr/bin/env python

from utils import rand_in_range, rand_un
from random import randint
import numpy as np
import pickle
import json

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

NUM_ROWS = 7
NUM_COLUMNS = 10
GOAL_STATE = (3, 7)

EPSILON = None
ALPHA = None
NUM_ACTIONS = None

state_action_values, policy = (None, None)

def agent_init():
    global state_action_values

    state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]
    state_action_values[GOAL_STATE[0]][GOAL_STATE[1]] = 0

def agent_start(state):
    global state_action_values, cur_state, cur_action

    #Choose the next action, epsilon greedy style
    cur_state = state
    if rand_un() < 1 - EPSILON:
        cur_action = state_action_values[state[0]][state[1]].index(max(state_action_values[state[0]][state[1]]))
    else:
        cur_action = rand_in_range(NUM_ACTIONS)

    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action

    next_state = state
    #Choose the next action, epsilon greedy style
    if rand_un() < 1 - EPSILON:
        next_action = state_action_values[state[0]][state[1]].index(max(state_action_values[state[0]][state[1]]))
    else:
        next_action = rand_in_range(NUM_ACTIONS)

    #Update the state action value function
    state_action_values[cur_state[0]][cur_state[1]][cur_action] += ALPHA * (reward + state_action_values[next_state[0]][next_state[1]][next_action] - state_action_values[cur_state[0]][cur_state[1]][cur_action])

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
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global EPSILON, ALPHA, NUM_ACTIONS
    params = json.loads(in_message)
    EPSILON = params["EPSILON"]
    ALPHA = params['ALPHA']
    NUM_ACTIONS = params['NUM_ACTIONS']

    return
