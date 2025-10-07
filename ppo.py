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
    print(f"PPO Update Called: {len(buffer.states)} steps in buffer")
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

    print("Before update w1[0][:3]:", agent.policy_net.w1[0][:3].copy())


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
            policy_loss = -np.minimum(ratio * adv, clipped_ratio * adv)

            #value forward pass
            value_est = agent.value_net.predict_value(state)
            value_loss = (value_est - ret) ** 2

            #combine losses
            total_loss = policy_loss + 0.5 * value_loss #we can do entropy lateer

            print(f"Epoch {epoch} Step {i} | Advantage: {adv:.5f}, Return: {ret:.5f}, "
                  f"Old log prob: {old_log_prob:.5f}, New log prob: {log_prob:.5f}, "
                  f"Ratio: {ratio:.5f}, Policy Loss: {policy_loss:.5f}, Value Loss: {value_loss:.5f}")


            #backpropagation (manual gradient descent)
            #we use finite differences for now (but it is very basic i know)
            for net, loss_grad_fn in [(agent.policy_net, grad_policy),
                                        (agent.value_net, grad_value)]:
                update_weights(net, state, action_taken, ret, adv, lr, loss_grad_fn)
    print("After  update w1[0][:3]:", agent.policy_net.w1[0][:3])


def grad_policy(net, state, action_taken, advantage, lr=1e-3):
    """"
    Backprop for PPO policy net using softmax and advantage
    """

    x = np.array(state).reshape(1, -1)

    #forward pass
    z1 = x @ net.w1 + net.b1
    a1 = np.tanh(z1)
    z2 = a1 @ net.w2 + net.b2
    logits = z2.flatten()
    probs = net.softmax(logits)

    #dL/dlogits = (probs - one_hot(action)) * advantage
    dlogits = (probs - action_taken) * advantage

    #backprop w2, b2
    dL_dw2 = a1.T @ dlogits.reshape(1, -1)
    dL_db2 = dlogits

    #backprop thru tanh
    da1_dz1 = 1 - np.tanh(z1)**2
    dz1 = (dlogits @ net.w2.T) * da1_dz1

    #backprop w1, b1
    dL_dw1 = x.T @ dz1
    dL_db1 = dz1[0]

    print("Gradient norm (policy w1):", np.linalg.norm(dL_dw1))


    #gradient step
    net.w1 -= lr * dL_dw1
    net.b1 -= lr * dL_db1
    net.w2 -= lr * dL_dw2
    net.b2 -= lr * dL_db2

def grad_value(net, state, target_return, lr=1e-3):
    """"
    Backprop for value MLP: MSE loss between value estimate and return
    """
    x = np.array(state).reshape(1, -1)

    #forward pass
    z1 = x @ net.w1 + net.b1
    a1 = np.tanh(z1)
    z2 = a1 @ net.w2 + net.b2
    value_est = z2[0, 0] #scalar

    #Loss: (V - R)^2
    dloss_dz2 = 2 * (value_est - target_return)

    #backprop w2, b2
    dL_dw2 = a1.T * dloss_dz2
    dL_db2 = dloss_dz2

    #backprop a1 -> z1
    da1_dz1 = 1 - np.tanh(z1)**2
    dz1 = (dloss_dz2 * net.w2.T) * da1_dz1

    #backprop w1, b1
    dL_dw1 = x.T @ dz1
    dL_db1 = dz1[0]

    print("Gradient norm (value w1):", np.linalg.norm(dL_dw1))

    #gradient step
    net.w1 -= lr * dL_dw1
    net.b1 -= lr * dL_db1
    net.w2 -= lr * dL_dw2
    net.b2 -= lr * dL_db2


def update_weights(net, state, action, target, adv, lr, grad_fn):
    """
    update weights of net using approximated gradients
    """
    if grad_fn == grad_policy:
        grad_policy(net, state, action, adv, lr)
    elif grad_fn == grad_value:
        grad_value(net, state, target, lr)
    
