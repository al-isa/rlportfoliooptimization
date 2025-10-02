from simulator import MarketSimulator
from plot import plot_asset_prices, plot_portfolio_value, plot_with_regimes
from agent import PPOAgent
import numpy as np



if __name__ == "__main__":
    sim = MarketSimulator(n_assets=3, n_days=5)

    n_assets = sim.n_assets
    state_size = 3 * n_assets + 1
    agent = PPOAgent(n_assets=n_assets, state_size=state_size)


    print("Initial prices:", sim.prices)
    print("Initial portfolio value:", sim.portfolio_value)

    for _ in range(5):
        #create state vector
        prices = sim.prices
        recent_returns = sim.daily_returns[-1] if sim.daily_returns else np.zeros(n_assets)
        current_weights = sim.weights
        value = np.array([sim.portfolio_value])

        state = np.concatenate([prices, recent_returns, current_weights, value])

        #agent decides new weights
        action_weights = agent.act(state)

        #apply weights in the simulator (this deducts transaction cost too)
        sim.set_weights(action_weights)

        #step the enviroment forward
        prices, returns, regime, p_return, p_value = sim.step()

        print(f"Day {sim.current_day} | Regime: {regime}")
        print("  Prices:", prices.round(2))
        print("  Returns:", returns.round(4))
        print("  Agent Weights:", action_weights.round(3))
        print("  Portfolio return {:.4f} | Value: {:.4f}".format(p_return, p_value))
    
    print("\nFinal Portfolio Value:", sim.portfolio_value)
    print("Cumulative Return: {:.2f}%".format(sim.cumulative_return() * 100))
    print("Sharpe Ratio: {:.4f}".format(sim.compute_sharpe_ratio()))

    #visulaization stuff
    # plot_portfolio_value(sim.portfolio_history)
    # plot_asset_prices(sim.price_history)
    # plot_with_regimes(sim.portfolio_history, sim.regime_history)