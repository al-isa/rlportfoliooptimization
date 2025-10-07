import numpy as np
from scipy.stats import dirichlet
from ppo import stable_softplus, dirichlet_log_prob

class MLP:
    """Simple fully connected neural network"""
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def forward(self, x):
        #lets make this a 2d vector to avoid type mismatch
        x = np.array(x).reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        return z2.flatten() #shape (output_size,) finaloutput
    
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
        logits = self.policy_net.forward(state)
        alpha = stable_softplus(logits) + 1e-3
        alpha = np.clip(alpha, 0.5, 50.0)

        weights = np.random.gamma(alpha)
        weights /= np.sum(weights)

        log_prob = dirichlet.logpdf(weights, alpha)
        return weights, log_prob, alpha #already sums to 1 due to softmax
    
    def evaluate_values(self, state):
        return self.value_net.predict_value(state)
    
        