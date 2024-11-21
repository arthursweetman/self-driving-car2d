"""
This module should have all the moethods for training and tuning the model
to accomplish my predefined goals.
"""

import model
from main import GameState
import random
import numpy as np
from collections import deque

NUM_EPISODES = 10
NUM_TIMESTAMPS = 1000
epsilon = 0.1
REPLAY_MEMORY_SIZE = 1000

def initialize_Q():
    pass

def argmax_Qstar(curr_state):
    pass


def reward(state):
    r = None

    return r


def step(st, action):
    st_next = None


def gradient_descent():
    pass

if __name__ == "__main__":
    D = deque(maxlen = REPLAY_MEMORY_SIZE)
    initialize_Q()
    for episode in range(NUM_EPISODES):
        st = GameState()
        for t in range(NUM_TIMESTAMPS):

            # Epsilon-greedy Policy
            if random.uniform(0,1) < epsilon:
                action = [0,0,0,0]
                action[random.randint(0,3)] = 1
            else:
                action = argmax_Qstar(st)

            # Execute action and observe reward rt and state s_t+1
            st_next = st.game_step(action)
            r = reward(st_next)

            # store transition in D
            D.append((st, action, r, st_next))

            # Set s_t as s_t+1
            st = st_next  # make sure this generates a copy so that they are not pointed at the same object in memory

            # Sample random minibatch of transitions from D
            rand_transition = random.sample(D, 1)  # returns a tuple

            if rand_transition[3] is None:  # next state is terminal
                yj = rand_transition

            # Perform gradient descent step
            gradient_descent()
