#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue


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
    parser.add_argument('-e', nargs='?', type=float, default=0.1, help='Epsilon paramter value for to be used by the agent when selecting actions epsilon greedy style. Default = 0.1')
    parser.add_argument('-a', nargs='?', type=float, default=0.50, help='Alpha parameter which specifies the step size for the update rule. Default value = 0.50')
    parser.add_argument('-n', nargs='?', type=int, default=1, help='Number of steps to be used in n-step sarsa. Default value is n = 1')
    parser.add_argument('-actions', nargs='?', type=int, default=4, help='The number of moves considered valid for the agent must be 4, 8, or 9. Default value is actions = 4')
    parser.add_argument('-algo', nargs='?', type=str, default='single', help='Specify whether to use a single step or multistep agent. Default value is algo = single. Other option is algo = multi')

    args = parser.parse_args()

    if args.e < 0 or args.e > 1 or args.a < 0 or args.a > 1:
        exit("Epsilon and Alpha parameters must be a value between 0 and 1, inclusive")

    if args.n < 1:
        exit("The number of steps must be >= 1")

    if args.actions not in VALID_MOVE_SETS:
        exit("The valid move sets are 4, 8, and 9. Please choose one of those")

    if args.algo == "single":
        RLGlue("windy_grid_env", "windy_grid_agent")
    elif args.algo == "multi":
        RLGlue("windy_grid_env", "windy_grid_nstep_agent")
    else:
        exit("'single' and 'multi' are the only 2 algorithm options. Please choose one.")

    EPSILON = args.e
    ALPHA = args.a
    N = args.n

    #To send to the environment and agent files
    agent_params = {"EPSILON": args.e, "ALPHA": args.a, "N": args.n, "NUM_ACTIONS": args.actions}
    enviro_params = {"NUM_ACTIONS": args.actions}

    num_episodes = 170
    max_steps = 8000
    num_runs = 1

    episodes_completed = []
    time_steps_completed = []
    cur_num_episodes = 0
    cur_time_step = 0

    print("Training the agent...")
    for run in range(num_runs):
      counter = 0
      print "run number: ", run
      RL_env_message(json.dumps(enviro_params))
      RL_agent_message(json.dumps(agent_params))
      RL_init()
      print "\n"

      for episode in range(num_episodes):
          RL_episode(max_steps)
          episodes_completed.append(RL_num_episodes())
          cur_time_step += RL_num_steps()
          time_steps_completed.append(cur_time_step)
      RL_cleanup()

    print("Printing the results...")
    print("Episode Data")
    print(episodes_completed)
    print("\n")
    print("Time step data")
    print(time_steps_completed)

    print "\nPlotting the results..."
    plt.ylabel('Episodes')
    plt.xlabel("Time Steps")
    plt.axis([0, 8000, 0, 170])
    plt.plot(time_steps_completed, episodes_completed, 'r-', label="Epsilon = " + str(EPSILON) + " Alpha = " + str(ALPHA) + " N = " + str(N))
    plt.legend(loc='center', bbox_to_anchor=(0.70,0.10))
    plt.show()
    print "\nFinished!"
