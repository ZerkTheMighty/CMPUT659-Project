#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
  Modified by Cody Rosevear for use in CMPUT659, Winter 2018, University of Alberta
"""

from rl_glue import *  # Required for RL-Glue

import random
import numpy as np

import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import json

VALID_MOVE_SETS = [4, 8, 9]

if __name__ == "__main__":

    #Determine which type of experiment is being run
    parser = argparse.ArgumentParser(description='Solve the windy gridworld problem')
    parser.add_argument('-e', nargs='?', type=float, default=1.0, help='Initial epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 1.0 It decays over time.')
    parser.add_argument('-a', nargs='?', type=float, default=0.50, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.50')
    parser.add_argument('-g', nargs='?', type=float, default=0.9, help='Discount factor, which determines how far ahead from the current state the agent takes into consideraton when updating its values. Default = 1.0')
    parser.add_argument('-n', nargs='?', type=int, default=1, help='Number of steps to be used in n-step sarsa. Default value is n = 1')
    parser.add_argument('-actions', nargs='?', type=int, default=4, help='The number of moves considered valid for the agent must be 4, 8, or 9. Default value is actions = 4')
    parser.add_argument('--algo', action='store_true', help='Specify whether to use a single step or multistep agent.')
    parser.add_argument('--stoch', action='store_true', help='Specify whether to train the agent with a stochastic or deterministic wind.')

    args = parser.parse_args()

    if args.e < 0 or args.e > 1 or args.a < 0 or args.a > 1 or args.g < 0 or args.g > 1:
        exit("Epsilon, Alpha, and parameters must be a value between 0 and 1, inclusive")

    if args.n < 1:
        exit("The number of steps must be >= 1")

    if args.actions not in VALID_MOVE_SETS:
        exit("The valid move sets are 4, 8, and 9. Please choose one of those")

    if args.algo:
        RLGlue("windy_grid_env", "windy_grid_agent")
    else:
        RLGlue("windy_grid_env", "windy_grid_nstep_agent")


    IS_STOCHASTIC = args.stoch
    EPSILON = args.e
    ALPHA = args.a
    N = args.n
    GAMMA = args.g
    NUM_ACTIONS = args.actions

    AGENTS = ['tabularQ', 'neural']
    GRAPH_COLOURS = ['r', 'g', 'b']

    num_episodes = 200
    max_steps = 1000
    num_runs = 10

    all_results = []
    print("Training the agents...")
    for agent in AGENTS:
        print("Training agent: {}".format(agent))

        #To send paramters to the environment and agent files
        agent_params = {"EPSILON": EPSILON, "ALPHA": ALPHA, "N": N, "NUM_ACTIONS": NUM_ACTIONS, "AGENT": agent, "GAMMA": GAMMA}
        enviro_params = {"NUM_ACTIONS": args.actions, "IS_STOCHASTIC": IS_STOCHASTIC}

        cur_agent_results = []
        for run in range(num_runs):
            run_results = []
            random.seed(run)
            np.random.seed(run)
            print("Run: {}".format(run))
            RL_env_message(json.dumps(enviro_params))
            RL_agent_message(json.dumps(agent_params))
            RL_init()

            for episode in range(num_episodes):
                print("Episode: {}".format(episode))
                RL_episode(max_steps)
                run_results.append(RL_num_steps())
                RL_cleanup()
            cur_agent_results.append(run_results)
        all_results.append(cur_agent_results)

    #Average the results for each parameter setting over all of the runs
    avg_results = []
    for i in range(len(all_results)):
        avg_results.append([np.mean(run) for run in zip(*all_results[i])])

    print "\nPlotting the results..."
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, max_steps + 1000])
    for i in range(len(avg_results)):
        plt.plot([episode for episode in range(num_episodes)], avg_results[i], GRAPH_COLOURS[i], label="Epsilon = " + str(EPSILON) + " Alpha = " + str(ALPHA) + " Gamma = " + str(GAMMA) +  " AGENT = " + AGENTS[i])
    plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
    plt.show()
    print "\nFinished!"
