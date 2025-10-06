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

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values [t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def ppo_update(agent, buffer, clip_epsilon=0.2, epochs=5, lr=1e-3):
        """
        Performs PPO updates on polciy and value networks
        """
        states = np.array(buffer.states)
        actions = np.array(buffer.actions)
        rewards = np.array(buffer.rewards)
        values = list(buffer.values)
        log_prob_old = np.array(buffer.log_probs)
        dones = np.array(buffer.dones)

        #compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones)
        advantages = np.array(advantages)
        returns = np.array(returns)

        for epoch in range(epochs):
            for i in range(len(states)):
                state = states[i]
                action_taken = actions[i]
                adv = advantages[i]
                ret = returns[i]
                old_log_prob = log_prob_old[i]

                #policy forward pass
                logits = agent.policy_net.forward(state)
                probs = agent.policy_net.softmax(logits)

                #new log prob of taken action (multivariate softmax)
                log_prob = np.log(np.dot(probs, action_taken) + 1e-8)
                ratio = np.exp(log_prob - old_log_prob)

                #clip ratio
                clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                policy_loss = -min(ratio * adv, clipped_ratio * adv)

                #value forward pass
                value_est = agent.value_net.predict_value(state)
                value_loss = (value_est - ret) ** 2

                #combine losses
                total_loss = policy_loss + 0.5 * value_loss #we can do entropy lateer

                #backpropagation (manual gradient descent)
                #we use finite differences for now (but it is very basic i know)
                for net, loss_grad_fn in [(agent.policy_net, grad_policy),
                                          (agent.value_net, grad_value)]:
                    update_weights(net, state, action_taken, ret, adv, lr, loss_grad_fn)

    def grad_policy(net, state, action, advantage):
        pass

    def grad_value(net, state, target_return):
        pass

    def update_weights(net, state, action, target, adv, lr, grad_fn):
        """
        update weights of net using approximated gradients
        """
        pass
        
