# Import routines

import numpy as np
import math
import random
from itertools import product
from sklearn.preprocessing import OneHotEncoder

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
        
        # The state space is the cartesian product of the locations, time_range and days_range
        # State is represented as a tuple (Location, time of the day, day)
        # Thus state space is array of the tuple described above
        self.state_space = list(product(locations, time_range, days_range))
        
        # The initial state is initialized by picking a random state from state space.
        self.state_init = random.sample(self.state_space, 1)[0]
        
        # Prepare a one hot encoder for preparing the input vector to use later
        array = np.ones((24, 3))
        array[0:5, 0:1] = locations.reshape((5,1))
        array[0:24, 1:2] = time_range.reshape((24,1))
        array[0:7, 2:3] = days_range.reshape((7,1))
        
        self.enc = OneHotEncoder()
        self.enc.fit(array)
        
        # Load the time matrix
        self.time_matrix = np.load('TM.npy')
        
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # Use one hot encoder to get a one hot encoded format for the state
        state_encod = self.enc.transform([state]).toarray()[0]
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
        if location == 1:
            requests = np.random.poisson(2)
        elif location == 2:
            requests = np.random.poisson(12)
        elif location == 3:
            requests = np.random.poisson(4)
        elif location == 4:
            requests = np.random.poisson(7)
        elif location == 5:
            requests = np.random.poisson(8)



        # We want to cap requests to 15
        if requests >15:
            requests =15

        # We want indexes with range(0,20)
        possible_actions_index = random.sample(range(0, (m-1)*m), requests) 
        actions = [self.action_space[i] for i in possible_actions_idx]

        # Append the action (0,0) meaning the driver chose to ignore requests.
        actions.append((0,0))

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        pickup = action[0]
        drop = action[1]
        
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        
        if pickup == 0 and drop == 0:
            # No action so the reward is negative C
            reward = -C
        else:
            # Indexing starts with 0, and our location values like vary between 1 to 5.
            # So we need to subtract 1 from each curr_loc, pickup and drop values to use for indexing time_matrix
            curr_loc -= 1
            pickup -= 1
            drop -= 1
            
            reward = R * self.time_matrix[pickup][drop] - C * (self.time_matrix[pickup][drop] + self.time_matrix[curr_loc][pickup])
        
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        pickup = action[0]
        drop = action[1]
        
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        
        if pickup == 0 and drop == 0:
            # No action so the next state's location remains the same
            next_loc = curr_loc
            
            # No action so the next state's time is incremented by 1 and then other calculations are done.
            next_time = curr_time + 1
        else:
            # Action taken so next state's location will be the drop location
            next_loc = drop
            
            # Adjustments for lookup in the time_matrix
            curr_loc -= 1
            pickup -= 1
            drop -= 1
            
            # Action is taken. So the next state's time is incremented by time_matrix value for the pick and drop 
            # and hr and day and then other calculations are done.
            next_time = curr_time + self.time_matrix[curr_loc][pickup] + self.time_matrix[pickup][drop]  
            
            
        if next_time > 23:
            # The max time increment is by 11. So the calculation below is sufficient for incrementing the days.
            next_time = next_time % 24
            next_day = curr_day + 1
        else:
            next_day = curr_day
            
        next_state = (next_loc, next_time, next_day)
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
