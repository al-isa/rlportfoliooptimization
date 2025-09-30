import numpy as np

class MarketSimulator:
    def __init__(self, n_assets=3, n_days=252, seed=42):
        np.random.seed(seed)
        self.n_assets = n_assets
        self.n_days = n_days
        # self.current_day = 0
        self.seed = seed
        self.reset()

        # #simulated log returns for each asset (random for now)
        # self.daily_returns = np.random.normal(loc=0.0005, scale =0.01, size=(n_days, n_assets))

        # #initial prices (eg, $100 per asset)
        # self.prices = np.full(n_assets, 100.0)
        # self.price_history = [self.prices.copy()]

    def reset(self):
        """Reset Simulation"""
        np.random.seed(self.seed)
        self.current_day = 0
        self.prices = np.full(self.n_assets, 100.0)
        self.price_history = [self.prices.copy()]

    def step(self):
        """Advance one day in simulation and update asset prices."""
        if self.current_day >= self.n_days:
            raise Exception("Simulation has ended")
        
        #apply return to price: P_new = P_old * exp(r)
        returns_today = self.daily_returns[self.current_day]
        self.prices *= np.exp(returns_today)
        self.current_day += 1
        self.price_history.append(self.prices.copy())
        
        return self.prices.copy(), returns_today.copy()