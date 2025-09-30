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

        #regime parameters
        self.regime = 'bull' #bull or bear
        self.regime_durations = np.random.randint(30, 90, size=10)
        self.regime_switch_days = np.cumsum(self.regime_durations)
        self.current_regime_index = 0

        #volatility for each asset: initialized at 1%
        self.volatility = np.full(self.n_assets, 0.01)

        #placeholder for transaction cost % per trade
        self.transaction_cost = 0.001 #0.01%

        #simulated daily returns will be created dynamically
        self.daily_returns = []

    def _update_regime(self):
        """Switch between bull and bear regimes"""
        if self.current_day >= self.regime_switch_days[self.current_regime_index]:
            self.current_regime_index += 1
            self.regime = 'bear' if self.regime == 'bull' else 'bull'

    def _generate_returns(self):
        """Create clustered returns with volatility and regime influenece"""
        self._update_regime()

        #volatility clustering (GARCH-like)
        shock = np.random.randn(self.n_assets)
        self.volatility = 0.9 * self.volatility + 0.1 * np.abs(shock)

        #mean return based on regime
        if self.regime == 'bull':
            mu = 0.001 #+0.1% expected daily return
        else: 
            mu = -0.001 #-0.1% in bear market

        #return = mu + shock * volatility
        returns = mu + shock * self.volatility
        return returns

    def step(self):
        """Advance one day in simulation and update asset prices."""
        if self.current_day >= self.n_days:
            raise Exception("Simulation has ended")
        
        #apply return to price: P_new = P_old * exp(r)
        returns_today = self._generate_returns()
        self.daily_returns.append(returns_today.copy())

        self.prices *= np.exp(returns_today)
        self.current_day += 1
        self.price_history.append(self.prices.copy())
        
        return self.prices.copy(), returns_today.copy(), self.regime