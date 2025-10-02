from simulator import MarketSimulator
from plot import plot_asset_prices, plot_portfolio_value, plot_with_regimes

if __name__ == "__main__":
    sim = MarketSimulator(n_assets=3, n_days=5)

    #equal allocation
    sim.set_weights([0.4, 0.3, 0.3]) #agent can change this dynamically


    print("Initial prices:", sim.prices)
    print("Initial portfolio value:", sim.portfolio_value)

    for _ in range(5):
        prices, returns, regime, p_return, p_value = sim.step()
        print(f"Day {sim.current_day} | Regime: {regime}")
        print("  Prices:", prices.round(2))
        print("  Returns:", returns.round(4))
        print("  Portfolio return {:.4f} | Value: {:.4f}".format(p_return, p_value))
    
    print("\nFinal Portfolio Value:", sim.portfolio_value)
    print("Cumulative Return: {:.2f}%".format(sim.cumulative_return() * 100))
    print("Sharpe Ratio: {:.4f}".format(sim.compute_sharpe_ratio()))

    #visulaization stuff
    # plot_portfolio_value(sim.portfolio_history)
    # plot_asset_prices(sim.price_history)
    # plot_with_regimes(sim.portfolio_history, sim.regime_history)