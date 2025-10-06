import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.__init__()
    
    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        values = values + [0] #add final bootstrap value

        returns = 0

        return advantages, returns