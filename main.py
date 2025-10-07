from simulator import MarketSimulator
from plot import plot_asset_prices, plot_portfolio_value, plot_with_regimes
from agent import PPOAgent
from ppo import ppo_update, RolloutBuffer
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    n_assets = 3
    n_days = 50 #sim steps per episode
    n_episodes = 20 #training episodes

    state_size = 3 * n_assets + 1
    agent = PPOAgent(n_assets=n_assets, state_size=state_size)

    episode_returns = []

    for episode in range(n_episodes):
        sim=MarketSimulator(n_assets=n_assets, n_days=n_days)
        buffer = RolloutBuffer()

        print(f"\nEpisode {episode + 1} / {n_episodes}")
        print("Initial prices:", sim.prices)
        print("Initial portfolio value:", sim.portfolio_value)

        for step in range(5):
            #create state vector
            prices = sim.prices
            recent_returns = sim.daily_returns[-1] if sim.daily_returns else np.zeros(n_assets)
            current_weights = sim.weights
            value = np.array([sim.portfolio_value])

            state = np.concatenate([
                prices, recent_returns, current_weights, value
                ])

            #agent action
            action = agent.act(state)
            if step == 0:
                print(f" Agent weights on Day 0: {action.round(3)}")
            sim.set_weights(action)

            #value estimate ( for advantage)
            value_est = agent.evaluate_values(state)

            #log porb of action (softmax)
            logits = agent.policy_net.forward(state)
            probs = agent.policy_net.softmax(logits)
            log_prob = np.log(np.dot(probs, action) + 1e-8)

            #step sim
            prices, returns, regime, reward, p_value = sim.step()

            done = step == (n_days - 1)

            buffer.store(state, action, reward, value_est, log_prob, done)


        # Log episode result
        print("Final Value:", sim.portfolio_value)
        print("Cumulative Return: {:.2f}%".format(sim.cumulative_return() * 100))
        print("Sharpe Ratio: {:.4f}".format(sim.compute_sharpe_ratio()))
        episode_returns.append(sim.cumulative_return())

        #train ppo
        ppo_update(agent, buffer)

    
    # Plot results
    plt.plot(np.cumsum(episode_returns))
    plt.title("Cumulative Return Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

    #visulaization stuff
    # plot_portfolio_value(sim.portfolio_history)
    # plot_asset_prices(sim.price_history)
    # plot_with_regimes(sim.portfolio_history, sim.regime_history)