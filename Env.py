# Import routines

import numpy as np
import math
import random
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        locations = np.arange(1,6)
        time_range = np.arange(0,24)
        days_range = np.arange(0,7)
        
        # Action space is the cartesian product of the locations that the cab can travel to.
        # The pick and drop locations cannot be same so we drop such actions
        self.action_space = [x for x in list(product(locations, locations)) if x[0] != x[1]]
        
        # The state space is the cartesian product of the locations
        self.state_space = list(product(locations, time_range, days_range))
        
        # The initial state is initialized by picking a random state from state space.
        self.state_init = random.sample(self.state_space, 1)[0]

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)








        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
