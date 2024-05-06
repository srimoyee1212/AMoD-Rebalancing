import numpy as np

class SARSA_Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_dim, action_dim))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        td_target = reward + self.discount_factor * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
    
    def get_q_table(self):
        return self.q_table
