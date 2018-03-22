#!/usr/bin/env python

from utils import rand_norm, rand_in_range, rand_un
import numpy as np
import json

current_state = None
IS_STOCHASTIC = None
IS_SPARSE = None

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
NORTHEAST = 4
SOUTHEAST = 5
SOUTHWEST = 6
NORTHWEST = 7
NO_MOVEMENT = 8


COLUMN_WIND_STRENGTH = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 2, 3], [1, 2, 3], [0, 1, 2], [0, 0, 1]]
MAX_ROW = 6
MAX_COLUMN = 9
MIN_ROW = 0
MIN_COLUMN = 0

START_STATE = [3, 0]
GOAL_STATE = [3, 7]

def env_init():
    global current_state
    current_state = None


def env_start():
    """ returns numpy array """
    global current_state
    current_state = START_STATE
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state

    if not action in ACTION_SET:
        print "Invalid action taken!!"
        print "action : ", action
        print "current_state : ", current_state
        exit(1)

    old_state = current_state
    cur_row = current_state[0]
    cur_column = current_state[1]

    if IS_STOCHASTIC:
        wind_strength_idx = rand_in_range(2)
    else:
        wind_strength_idx = 1

    #Change the state based on the agent action and wind strength
    if action == NORTH:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx] + 1, cur_column]
    elif action == NORTHEAST:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx] + 1, cur_column + 1]
    elif action == EAST:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx], cur_column + 1]
    elif action == SOUTHEAST:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx] - 1, cur_column + 1]
    elif action == SOUTH:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx] - 1, cur_column]
    elif action == SOUTHWEST:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx] - 1, cur_column - 1]
    elif action == WEST:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx], cur_column - 1]
    elif action == NORTHWEST:
        current_state = [cur_row + COLUMN_WIND_STRENGTH[cur_column][wind_strength_idx] + 1, cur_column - 1]

    #Enforce the constraint that actions do not leave the grid world
    if current_state[0] > MAX_ROW:
        current_state[0] = MAX_ROW
    elif current_state[0] < MIN_ROW:
        current_state[0] = MIN_ROW

    if current_state[1] > MAX_COLUMN:
        current_state[1] = MAX_COLUMN
    elif current_state[1] < MIN_COLUMN:
        current_state[1] = MIN_COLUMN

    if IS_SPARSE:
        if current_state == GOAL_STATE:
            is_terminal = True
            reward = 1
        else:
            is_terminal = False
            reward = 0
    else:
        if current_state == GOAL_STATE:
            is_terminal = True
            reward = 0
        else:
            is_terminal = False
            reward = -1

    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    global ACTION_SET, IS_STOCHASTIC, IS_SPARSE
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    params = json.loads(in_message)
    if params['NUM_ACTIONS'] == 4:
        ACTION_SET = [NORTH, EAST, SOUTH, WEST]
    elif params['NUM_ACTIONS'] == 8:
        ACTION_SET = [NORTH, EAST, SOUTH, WEST, NORTHEAST, SOUTHEAST, SOUTHWEST, NORTHWEST]
    elif params['NUM_ACTIONS'] == 9:
        ACTION_SET = [NORTH, EAST, SOUTH, WEST, NORTHEAST, SOUTHEAST, SOUTHWEST, NORTHWEST, NO_MOVEMENT]

    IS_STOCHASTIC = params['IS_STOCHASTIC']
    IS_SPARSE = params['IS_SPARSE']

    return
