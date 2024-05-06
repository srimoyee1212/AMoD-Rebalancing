import numpy as np

class SARSA_Agent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_dim, action_dim))
        self.total_reward = 0  # To calculate total reward per episode
        self.total_exploration = 0  # To count number of times exploration is chosen
        self.total_exploitation = 0  # To count number of times exploitation is chosen
        self.episode_steps = 0  # To calculate total steps per episode

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            self.total_exploration += 1
            return np.random.randint(0, self.action_dim)
        else:
            self.total_exploitation += 1
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        td_target = reward + self.discount_factor * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def train(self, episodes, env):
        for episode in range(episodes):
            state = env.reset()
            action = self.choose_action(state)
            done = False
            self.total_reward = 0
            self.episode_steps = 0
            while not done:
                next_state, reward, done = env.step(state, action)
                next_action = self.choose_action(next_state)
                self.update_q_table(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                self.total_reward += reward
                self.episode_steps += 1

    def get_q_table(self):
        return self.q_table

    def get_avg_reward(self):
        return self.total_reward / self.episode_steps if self.episode_steps > 0 else 0

    def get_exploration_exploitation_ratio(self):
        total_choices = self.total_exploration + self.total_exploitation
        return self.total_exploration / total_choices if total_choices > 0 else 0

    def check_convergence(self, threshold=0.01, window_size=10):
    # Check if the change in Q-values is below a threshold for a certain window size
        if len(self.rewards_history) >= window_size:
            recent_rewards = self.rewards_history[-window_size:]
            average_reward = sum(recent_rewards) / window_size
            if abs(average_reward - self.last_average_reward) < threshold:
                print("Converged!")
                return True
        return False