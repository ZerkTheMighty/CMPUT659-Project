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

    #Agent parameters
    EPSILON = 1.0
    ALPHA = 0.1
    GAMMA = 1
    AGENTS = ['neural']

    num_episodes = 200
    max_steps = 25
    num_runs = 1

    print("Training the agents...")
    all_results = []
    for agent in AGENTS:
        print("Training agent: {}".format(agent))
        agent_params = {"EPSILON": EPSILON, "ALPHA": ALPHA, "GAMMA": GAMMA, "AGENT": agent}
        RL_agent_message(json.dumps(agent_params))
        cur_agent_results = []
        for run in range(num_runs):
            run_results = []
            print("Run number: {}".format(str(run)))
            RL_init(run)
            for episode in range(num_episodes):
                print("Episode number: {}".format(str(episode)))
                RL_episode(max_steps)
                run_results.append(RL_num_steps())
            #print("Run {} concluded. Starting a new run.".format(run))
            #sleep(5)
                RL_cleanup()
            cur_agent_results.append(run_results)
        all_results.append(cur_agent_results)

    #Averge the results for each parameter setting over the 10 runs
    avg_results = []
    for i in range(len(all_results)):
        avg_results.append([np.mean(run) for run in zip(*all_results[i])])

    print "\nPlotting the results..."
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, 1000])
    for i in range(len(avg_results)):
        plt.plot([episode for episode in range(num_episodes)], avg_results[i], GRAPH_COLOURS[i], label="Epsilon = " + str(EPSILON) + " Alpha = " + str(ALPHA) + " Gamma = " + str(GAMMA) +  " AGENT = " + AGENTS[i])
    plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
    plt.show()
    print "\nFinished!"
