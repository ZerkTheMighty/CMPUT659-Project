#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
  Modified by Cody Rosevear for use in CMPUT659, Winter 2018, University of Alberta
"""

from __future__ import division
from rl_glue import *  # Required for RL-Glue
from time import sleep

import argparse
import json
import random

import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

GRAPH_COLOURS = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
AGENTS = ['random', 'tabularQ', 'neural', 'reward', 'state', 'redundant', 'noise']

VALID_MOVE_SETS = [4, 8, 9]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solves the gridworld maze problem, as described in Sutton & Barto, 2018')
    parser.add_argument('-e', nargs='?', type=float, default=0.1, help='Epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.1 This represents the minimum value epislon will decay to, since it initially starts at 1')
    parser.add_argument('-a', nargs='?', type=float, default=0.001, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.001')
    parser.add_argument('-g', nargs='?', type=float, default=0.9, help='Discount factor, which determines how far ahead from the current state the agent takes into consideraton when updating its values. Default = 1.0')
    parser.add_argument('-n', nargs='?', type=int, default=3, help='The number of states to use in the auxiliary prediction tasks. Default n = 3')
    parser.add_argument('-actions', nargs='?', type=int, default=4, help='The number of moves considered valid for the agent must be 4, 8, or 9. This only applies to the windy gridwordl experiment. Default value is actions = 4')
    parser.add_argument('--windy', action='store_true', help='Specify whether to use a single step or multistep agent.')
    parser.add_argument('--stochastic', action='store_true', help='Specify whether to train the agent with stochastic obstacle states, rather than simple wall states that the agent can\'t pass through.')
    parser.add_argument('--sparse', action='store_true', help='Specify whether the environment reward structure is rich or sparse. Rich rewards include -1 at every non-terminal state and 0 at the terminal state. Sparse rewards include 0 at every non-terminal state, and 1 at the terminal state.')
    parser.add_argument('-name', nargs='?', type=str, help='The name of the file to save the experiment results to. File format is png.')

    args = parser.parse_args()

    if args.e < 0 or args.e > 1 or args.a < 0 or args.a > 1 or args.g < 0 or args.g > 1:
        exit("Epsilon, Alpha, and Gamma parameters must be a value between 0 and 1, inclusive.")

    if args.actions not in VALID_MOVE_SETS:
        exit("The valid move sets are 4, 8, and 9. Please choose one of those.")

    if args.windy:
        RLGlue("windy_grid_env", "grid_agent")
    else:
        if args.actions != 4:
            exit("The only valid action set for the non-windy gridworld is 4 moves.")
        RLGlue("grid_env", "grid_agent")


    #Agent and environment parameters, and experiment settings
    EPSILON = args.e
    ALPHA = args.a
    GAMMA = args.g
    N = args.n
    IS_STOCHASTIC = args.stochastic
    NUM_ACTIONS = args.actions
    IS_SPARSE = args.sparse
    RESULTS_FILE_NAME = args.name

    num_episodes = 50
    max_steps = 1000
    num_runs = 50

    #The main experiment loop
    print("Training the agents...")
    all_results = []
    for agent in AGENTS:
        print("Training agent: {}".format(agent))
        cur_agent_results = []
        for run in range(num_runs):
            #Set random seeds to ensure replicability of results and the same trajectory of experience across agents for valid comparison
            np.random.seed(run)
            random.seed(run)

            #Send the agent and environment parameters to use for the current run
            agent_params = {"EPSILON": EPSILON, "ALPHA": ALPHA, "GAMMA": GAMMA, "AGENT": agent, "N": N, "IS_STOCHASTIC": IS_STOCHASTIC}
            enviro_params = {"NUM_ACTIONS": NUM_ACTIONS, "IS_STOCHASTIC": IS_STOCHASTIC, "IS_SPARSE": IS_SPARSE}
            RL_agent_message(json.dumps(agent_params))
            RL_env_message(json.dumps(enviro_params))

            run_results = []
            print("Run number: {}".format(str(run)))
            RL_init()
            for episode in range(num_episodes):
                print("Episode number: {}".format(str(episode)))
                RL_episode(max_steps)
                run_results.append(RL_num_steps())
                RL_cleanup()
            cur_agent_results.append(run_results)
        all_results.append(cur_agent_results)

    #Average the results for each parameter setting over all of the runs
    avg_results = []
    for i in range(len(all_results)):
        avg_results.append([np.mean(run) for run in zip(*all_results[i])])

    print("Plotting the results...")
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, max_steps + 1000])
    for i in range(len(avg_results)):
        cur_data = [episode for episode in range(num_episodes)]
        plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="Epsilon Min = {} Alpha = {} Gamma = {} N = {} AGENT = {}".format(EPSILON, ALPHA, GAMMA, N, AGENTS[i]))
    plt.legend(loc='center', bbox_to_anchor=(0.50,0.90))
    if RESULTS_FILE_NAME:
        print("Saving the results...")
        plt.savefig("{}.png".format(RESULTS_FILE_NAME), format="png")
    else:
        print("Displaying the results...")
        plt.show()
    print("Experiment completed!")
