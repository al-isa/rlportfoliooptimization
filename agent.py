import numpy as np

class MLP:
    """Simple fully connected neural network"""
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(input_size, hidden_size) * 0.1
        self.b2 = np.zeros(hidden_size)
    
    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        return z2 #finalinput
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x)) #for stability
        return exp_x / np.sum(exp_x)
    
    def predict_action(self, x):
        logits = self.forward(x)
        return self.softmax(logits) #interpreted as portfolio weights
    
    def predict_value(self, x):
        return self.forward(x)[0] #scalar output
    
class PPOAgent:
    def __init__(self, n_assets, state_size, hidden_size = 64):
        self.n_assets = n_assets
        self.state_size = state_size
        self.policy_net = MLP(state_size, hidden_size, n_assets)
        self.value_net = MLP(state_size, hidden_size, 1)

    def act(self, state):
        """
        Given a state vector, return portfolio weights (action)
        """
        weights = self.policy_net.predict_action(state)
        return weights #already sums to 1 due to softmax
    
    def evaluate_values(self, state):
        return self.value_net.predict_value(state)
    
        