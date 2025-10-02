import matplotlib.pyplot as plt
import numpy as np

def plot_portfolio_value(portfolio_values):
    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_values, label="Portfolio Value", linewidth=2)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Day")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_asset_prices(price_history):
    prices = np.array(price_history)
    plt.figure(figsize=(10, 4))
    for i in range(prices.shape[1]):
        plt.plot(prices[:, i], label=f"Asset {i}")
    plt.title("Asset Prices Over Time")
    plt.xlabel("Day")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_with_regimes(portfolio_values, regimes):
    plt.figure(figsize=(10, 4))
    x = np.arange(len(portfolio_values))
    values = np.array(portfolio_values)

    for i in range(len(x)-1):
        color = 'green' if regimes[i] == 'bull' else 'red'
        plt.plot(x[i:i+2], values[i:i+2], color=color, linewidth=2)

    plt.title("Portfolio Value with Market Regimes")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()