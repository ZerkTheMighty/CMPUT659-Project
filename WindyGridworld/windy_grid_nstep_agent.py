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
N = None
NUM_ACTIONS = None

#Time step trackers
T = None
t = None

state_action_values = None

def agent_init():
    global state_action_values, state_store, action_store, reward_store

    state_action_values = [[[0 for action in range(NUM_ACTIONS)] for column in range(NUM_COLUMNS)] for row in range(NUM_ROWS)]
    state_action_values[GOAL_STATE[0]][GOAL_STATE[1]] = 0


def agent_start(state):
    global state_action_values, cur_state, cur_action, state_store, action_store, reward_store, t, T

    state_store = []
    action_store = []
    reward_store = []

    cur_state = state
    if rand_un() < 1 - EPSILON:
        cur_action = state_action_values[state[0]][state[1]].index(max(state_action_values[state[0]][state[1]]))
    else:
        cur_action = rand_in_range(NUM_ACTIONS)

    state_store.append(state)
    action_store.append(cur_action)
    T = float("inf")
    t = 0
    return cur_action


def agent_step(reward, state):
    global state_action_values, cur_state, cur_action, state_store, action_store, reward_store, t, T

    reward_store.append(reward)

    state_store.append(state)
    next_state = state

    if next_state == GOAL_STATE:
        T = t + 1

    #Select the next action
    if rand_un() < 1 - EPSILON:
        next_action = state_action_values[state[0]][state[1]].index(max(state_action_values[state[0]][state[1]]))
    else:
        next_action = rand_in_range(NUM_ACTIONS)
    action_store.append(next_action)

    tau = t - N + 1
    if tau >= 0:
        n_step = min(tau + N, T) + 1
        G = sum(reward_store[tau:n_step])
        if tau + N < T:
            S_tau_N = state_store[tau + N]
            A_tau_N = action_store[tau + N]
            G += state_action_values[S_tau_N[0]][S_tau_N[1]][A_tau_N]

        #Actual update occurs here
        S_tau = state_store[tau]
        A_tau = action_store[tau]
        state_action_values[S_tau[0]][S_tau[1]][A_tau] += ALPHA * (G - state_action_values[S_tau[0]][S_tau[1]][A_tau])

    t += 1
    cur_state = next_state
    cur_action = next_action
    return next_action

def agent_end(reward):
    global state_action_values

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global EPSILON, ALPHA, NUM_ACTIONS, N
    params = json.loads(in_message)
    EPSILON = params["EPSILON"]
    ALPHA = params['ALPHA']
    NUM_ACTIONS = params['NUM_ACTIONS']
    N = params['N']
    return
