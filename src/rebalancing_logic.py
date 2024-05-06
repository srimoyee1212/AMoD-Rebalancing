import numpy as np

class RebalancingLogic:
    def __init__(self, q_table):
        self.q_table = q_table

    def choose_actions(self, states):
        actions = []
        for state in states:
            action_probs = self.softmax(self.q_table[state])
            action = np.random.choice(len(action_probs), p=action_probs)
            actions.append(action)
        return actions

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)