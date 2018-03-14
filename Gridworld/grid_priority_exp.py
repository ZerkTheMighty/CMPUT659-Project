#!/usr/bin/env python

from rl_glue import *  # Required for RL-Glue


import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import json
import random

if __name__ == "__main__":

    GRAPH_COLOURS = ('r', 'g', 'b')

    #Agent parameters
    EPSILON = 0.10
    ALPHA = 0.10
    GAMMA = 0.95
    THETA = 0.00001
    N = 5
    AGENTS = ['Dyna_Q', "Prioritized_Sweeping"]

    num_episodes = 50
    max_steps = 10000
    num_runs = 10

    print("Training the agents...")
    all_results = []
    for agent in AGENTS:
        print("Training the " + agent + " agent...")
        if agent == 'Dyna_Q':
            RLGlue("grid_env", "grid_agent")
        elif agent == 'Prioritized_Sweeping':
            RLGlue("grid_env", "grid_priority_agent")
        else:
            exit("ERROR: VALID AGENT NOT SPECIFIED")
        agent_params = {"EPSILON": EPSILON, "ALPHA": ALPHA, "GAMMA": GAMMA, "N": N, "THETA": THETA}
        RL_agent_message(json.dumps(agent_params))
        cur_n_plan_results = []
        for run in range(num_runs):
            #Different parts of the program use np.random (via utils.py) and others use just random,
            #seeding both with the same seed here to make sure they both start in the same place per run of the program
            np.random.seed(run)
            random.seed(run)
            run_results = []
            print "run number: ", run

            RL_init()
            for episode in range(num_episodes):
                RL_episode(max_steps)
                run_results.append(RL_num_steps())
            RL_cleanup()
            cur_n_plan_results.append(run_results)
        all_results.append(cur_n_plan_results)

    #Averge the results for each parameter setting over the 10 runs
    avg_results = []
    for i in range(len(all_results)):
        avg_results.append([np.mean(run) for run in zip(*all_results[i])])

    print "\nPlotting the results..."
    plt.ylabel('Steps per episode')
    plt.xlabel("Episode")
    plt.axis([0, num_episodes, 0, 800])
    for i in range(len(avg_results)):
        plt.plot([episode for episode in range(num_episodes)], avg_results[i], GRAPH_COLOURS[i], label="Agent = " + AGENTS[i] + " Theta = " + str(THETA) + " Epsilon = " + str(EPSILON) + " Alpha = " + str(ALPHA) + " N = " + str(N))
    plt.legend(loc='center', bbox_to_anchor=(0.50,0.90))
    plt.show()
    print "\nFinished!"
