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


if __name__ == "__main__":

    RLGlue("grid_env", "grid_agent")
    GRAPH_COLOURS = ('r', 'g', 'b')
    AGENTS = ['neural']

    parser = argparse.ArgumentParser(description='Solves the gridworld maze problem, as described in Sutton & Barto, 2018')
    parser.add_argument('-e', nargs='?', type=float, default=0.1, help='Epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.1 This represents the minimum value epislon will decay to, since it initially starts at 1')
    parser.add_argument('-a', nargs='?', type=float, default=0.001, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.001')
    parser.add_argument('-g', nargs='?', type=float, default=0.9, help='Discount factor, which determines how far ahead from the current state the agent takes into consideraton when updating its values. Default = 1.0')

    args = parser.parse_args()

    if args.e < 0 or args.e > 1 or args.a < 0 or args.a > 1 or args.g < 0 or args.g > 1:
        exit("Epsilon, Alpha, and Gamma parameters must be a value between 0 and 1, inclusive")

    #Agent parameters
    EPSILON = args.e
    ALPHA = args.a
    GAMMA = args.g

    num_episodes = 200
    max_steps = 1000
    num_runs = 1

    print("Training the agents...")
    all_results = []
    for agent in AGENTS:
        print("Training agent: {}".format(agent))
        agent_params = {"EPSILON": EPSILON, "ALPHA": ALPHA, "GAMMA": GAMMA, "AGENT": agent}
        RL_agent_message(json.dumps(agent_params))
        cur_agent_results = []
        for run in range(num_runs):
            np.random.seed(run)
            random.seed(run)
            run_results = []
            print("Run number: {}".format(str(run)))
            RL_init(run + 1)
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

    print "\nPlotting the results..."
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, max_steps + 1000])
    plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
    for i in range(len(avg_results)):
        cur_data = [episode for episode in range(num_episodes)]
        plt.plot(cur_data, avg_results[i], GRAPH_COLOURS[i], label="Epsilon = " + str(EPSILON) + " Alpha = " + str(ALPHA) + " Gamma = " + str(GAMMA) +  " AGENT = " + AGENTS[i])
        plt.show()
        plt.savefig("{}_results".format(AGENTS[i]), format="png")
    print "\nFinished!"
