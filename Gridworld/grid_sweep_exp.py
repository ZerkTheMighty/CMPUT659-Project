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

    parser = argparse.ArgumentParser(description='Experiment that tests various values of alpha for a given epsilon on the obstacles gridworld problem')
    parser.add_argument('-e', nargs='?', type=float, default=0.01, help='Epsilon parameter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.01')
    args = parser.parse_args()

    if args.e < 0:
        exit("Epsilon parameter must be a value between 0 and 1, inclusive")

    RLGlue("grid_env", "grid_agent")

    #Agent parameter sweep space
    ALPHA_VALS = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
    EPSILON = args.e

    #Agent parameters
    GAMMA = 0.95
    N = 5

    num_episodes = 50
    max_steps = 20000
    num_runs = 10

    print("Training the agent...")
    all_results = []
    for alpha in ALPHA_VALS:
        cur_alpha_results = []
        agent_params = {"EPSILON": EPSILON, "ALPHA": alpha, "GAMMA": GAMMA, "N": N}
        RL_agent_message(json.dumps(agent_params))
        print("Training agent with EPSILON = " + str(EPSILON) + " and ALPHA = " + str(alpha))

        for run in range(num_runs):
            #Different parts of the program use np.random (via utils.py) and others use just random,
            #seeding both with the same seed here to make sure they both start in the same place, relative to themselves, per run of the program
            np.random.seed(run)
            random.seed(run)
            run_results = []
            print "run number: ", run
            RL_init()

            for episode in range(num_episodes):
                RL_episode(0)
                run_results.append(RL_num_steps())
            RL_cleanup()
            cur_alpha_results.append(run_results)
        all_results.append(cur_alpha_results)

    #print(all_results)
    avg_results = []
    for i in range(len(all_results)):
        avg_results.append([np.mean(run) for run in zip(*all_results[i])])
        avg_results[i] = np.mean(avg_results[i])

    print "\nPlotting the results..."
    plt.ylabel('Average # steps per episode')
    plt.xlabel("Alpha")
    plt.axis([0, 1, 30, 40])
    plt.plot(ALPHA_VALS, avg_results, label="Epsilon = " + str(EPSILON) + " GAMMA = " + str(GAMMA) + " N = " + str(N))
    plt.legend(loc='center', bbox_to_anchor=(0.60,0.80))
    #plt.savefig("/Users/codyrosevear/Code/RL/Assignments/A5/e_" + str(EPSILON)[0] + str(EPSILON)[2:] + ".png")
    plt.show()
    print "\nFinished!"
