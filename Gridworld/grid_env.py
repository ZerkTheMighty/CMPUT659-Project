#!/usr/bin/env python

from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import json

current_state = None

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ACTION_SET = [NORTH, EAST, SOUTH, WEST]

MAX_ROW = 5
MAX_COLUMN = 8
MIN_ROW = 0
MIN_COLUMN = 0

START_STATE = [3, 0]
GOAL_STATE = [5, 8]
OBSTACLE_STATES = [[2, 2], [3, 2], [4, 2], [1, 5], [3, 7], [4, 7], [5, 7]]

def env_init():
    return

def env_start():
    global current_state
    current_state = START_STATE
    return current_state

def env_step(action):
    global current_state
    #print(current_state)
    if not action in ACTION_SET:
        print "Invalid action taken!!"
        print "action : ", action
        print "current_state : ", current_state
        exit(1)

    old_state = current_state
    cur_row = current_state[0]
    cur_column = current_state[1]

    #Change the state based on the agent action and wind strength
    if action == NORTH:
        current_state = [cur_row + 1, cur_column]
    elif action == EAST:
        current_state = [cur_row, cur_column + 1]
    elif action == SOUTH:
        current_state = [cur_row - 1, cur_column]
    elif action == WEST:
        current_state = [cur_row, cur_column - 1]

    #Enforce the constraint that actions do not leave the grid world
    if current_state[0] > MAX_ROW:
        current_state[0] = MAX_ROW
    elif current_state[0] < MIN_ROW:
        current_state[0] = MIN_ROW

    if current_state[1] > MAX_COLUMN:
        current_state[1] = MAX_COLUMN
    elif current_state[1] < MIN_COLUMN:
        current_state[1] = MIN_COLUMN

    #Enforce the constraint that some squares are out of bounds
    if current_state in OBSTACLE_STATES:
        current_state = old_state

    if current_state == GOAL_STATE:
        is_terminal = True
        reward = 1
    else:
        is_terminal = False
        reward = 0

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    return

def env_message(in_message): # returns string, in_message: string
    return
