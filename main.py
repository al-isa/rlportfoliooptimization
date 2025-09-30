from simulator import MarketSimulator

if __name__ == "__main__":
    sim = MarketSimulator(n_assets=3, n_days=5)

    print("Initial prices:", sim.prices)
    for _ in range(5):
        prices, returns = sim.step()
        print(f"Day {sim.current_day}:")
        print("  Prices:", prices)
        print("  Returns:", returns)