import numpy as np

class Environment:
    def __init__(self):
      
        self.num_states = 10  # Number of states
        self.num_actions = 4  # Number of actions

    def reset(self):
     
        return np.random.randint(0, self.num_states)

    def step(self, state, action):

        next_state = (state + action) % self.num_states
        reward = 1 if next_state == self.num_states - 1 else 0  # Reward 1 for reaching the last state
        done = next_state == self.num_states - 1  # Done if last state reached
        return next_state, reward, done
