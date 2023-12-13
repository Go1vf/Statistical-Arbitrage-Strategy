import numpy as np
import pandas as pd

class BackTesting:
    def __init__(self, trading_strategy, start_time, end_time, initial_capital=10000):
        self.trading_strategy = trading_strategy
        self.start_time = start_time
        self.end_time = end_time
        self.initial_capital = initial_capital
        self.portfolio = {}  # To store the current portfolio status
        self.portfolio_value = []  # To track the portfolio value over time

    def trade(self, token, signal, price):
        """
        Execute a trade based on the given signal.
        """
        if signal == 'buy_open' and token not in self.portfolio:
            self.portfolio[token] = price
        elif signal == 'sell_open' and token not in self.portfolio:
            self.portfolio[token] = -price
        elif signal == 'close_position' and token in self.portfolio:
            self.portfolio.pop(token, None)

    def update_portfolio_value(self, prices):
        """
        Update the portfolio value based on current prices.
        """
        value = self.initial_capital
        for token, entry_price in self.portfolio.items():
            value += (prices[token] - entry_price)
        self.portfolio_value.append(value)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculate the Sharpe ratio for the strategy.
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    def calculate_maximum_drawdown(self, values):
        """
        Calculate the maximum drawdown for the strategy.
        """
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak: 
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def run_backtest(self, price_data):
        """
        Run the backtest of the trading strategy between the start_time and end_time.
        """
        timestamps = pd.date_range(self.start_time, self.end_time, freq='H')
        for timestamp in timestamps:
            self.trading_strategy.time = timestamp
            prices = price_data.loc[timestamp]
            s_scores = self.trading_strategy.calculate_s_scores(prices)
            signals = self.trading_strategy.generate_signals(s_scores)
            for token in signals.index:
                self.trade(token, signals.loc[token, 'trading_signal'], prices[token])
            self.update_portfolio_value(prices)

        # Calculate returns
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        # Calculate performance metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_maximum_drawdown(self.portfolio_value)

        return {
            'portfolio_values': self.portfolio_value,
            'sharpe_ratio': sharpe_ratio,
            'maximum_drawdown': max_drawdown
        }
