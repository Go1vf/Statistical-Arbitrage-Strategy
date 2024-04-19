import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from TradingStrategy import TradingStrategy

class BackTesting:
    def __init__(self, start_time, end_time, M, tokens_largest_cap, tokens_price, initial_capital=10000):
        self.start_time =  datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S%z')
        self.end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S%z')
        self.tokens_largest_cap = tokens_largest_cap
        self.tokens_price = tokens_price
        self.cash_value = initial_capital
        self.window = M
        self.portfolio = defaultdict(lambda: 0)  # To store the current portfolio status
        self.portfolio_value = {}  # To track the portfolio value over time
        self.run_backtest()
        
    def run_backtest(self):
        all_signals = []
        all_eigen_vectors_portfolios = []
        current = self.start_time
        while current <= self.end_time:
            print(current)
            current += timedelta(hours=1)
            strategy = TradingStrategy(self.tokens_price, self.tokens_largest_cap, current, self.window)
            signals = strategy.generate_signals()
            self.trade(signals, current)
            all_signals.append(signals)
            all_eigen_vectors_portfolios.append(strategy.eigen_vectors_portfolios)
        self.portfolio_value = pd.DataFrame(list(self.portfolio_value.items()), columns=['Time', 'PortfolioValue'])
        self.portfolio_value['Return'] = self.portfolio_value['PortfolioValue'].pct_change()
        self.signals_df = pd.concat(all_signals)
        # Concatenate all DataFrames in the all_eigen_vectors_portfolios list
        self.eigen_vectors_portfolios_df = pd.concat(all_eigen_vectors_portfolios)

    def trade(self, signal, time):
        """
        Execute a trade based on all the given signal.
        """
        for index, row in signal.iterrows():
            token = row['Token']
            trade_signal = row['trading_signal']
            curr_share = self.portfolio.get(token, 0)
            curr_prc = self.tokens_price.loc[time, token]
            if trade_signal == 'buy_open':
                # we first need to liquate position
                if curr_share < 0:
                    # It means we are capable of liqudating position
                    min_share = (self.cash_value // curr_prc)
                    if min_share > abs(curr_share):
                        self.cash_value -= curr_prc * abs(curr_share)
                        self.portfolio[token] = 0
                    else:
                    # We can only cover part of the short postion, then we should not buy open
                        self.cash_value -= curr_prc * min_share
                        self.portfolio[token] += min_share
                        continue
                # Check if we have enough money to buy 1 more share
                if self.cash_value >= curr_prc:
                    self.portfolio[token] += 1
                    self.cash_value -= curr_prc
                # If we do not have enough money to buy 1 share, do nothing
                else:
                    continue
            elif trade_signal == 'sell_open':
                # we first need to liqudate position
                if curr_share > 0:
                    self.cash_value += curr_prc * abs(curr_share)
                    self.portfolio[token] = 0
                self.portfolio[token] -= 1
                self.cash_value += curr_prc
            elif trade_signal == 'close_short':
                # If we are in long position
                if curr_share >= 0:
                    continue
                else:
                    min_share = (self.cash_value // curr_prc)
                    if min_share > abs(curr_share):
                        self.cash_value -= curr_prc * abs(curr_share)
                        self.portfolio[token] = 0
                    else:
                    # We can only cover part of the short postion
                        self.cash_value -= curr_prc * min_share
                        self.portfolio[token] += min_share
                        continue
            elif trade_signal == 'close_long':
                # If we are in short position
                if curr_share <= 0:
                    continue
                else:
                    self.cash_value += curr_prc * abs(curr_share)
                    self.portfolio[token] = 0
            else: # close all cases
                # cover short 
                if curr_share < 0:
                    min_share = (self.cash_value // curr_prc)
                    if min_share > abs(curr_share):
                        self.cash_value -= curr_prc * abs(curr_share)
                        self.portfolio[token] = 0
                    else:
                    # We can only cover part of the short postion
                        self.cash_value -= curr_prc * min_share
                        self.portfolio[token] += min_share
                        continue
                # cover long
                if curr_share > 0:
                    self.cash_value += curr_prc * abs(curr_share)
                    self.portfolio[token] = 0
        # Calculating Portfolio Value for the time
        total_value = 0
        for token, num_shares in self.portfolio.items():
            if num_shares != 0:  # Only consider tokens with a non-zero number of shares
                curr_prc = self.tokens_price.loc[time, token]
                total_value += num_shares * curr_prc
        self.portfolio_value[time] = total_value + self.cash_value

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """
        Calculate the Sharpe ratio for the strategy.
        """
        excess_returns = self.portfolio_value['Return'] - risk_free_rate
        return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(8760)

    def calculate_maximum_drawdown(self):
    # Calculate the running maximum
        running_max = self.portfolio_value['PortfolioValue'].cummax()
        
        # Calculate drawdowns
        drawdown = (self.portfolio_value['PortfolioValue'] - running_max) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        return max_drawdown

    # def run_backtest(self, price_data):
    #     """
    #     Run the backtest of the trading strategy between the start_time and end_time.
    #     """
    #     timestamps = pd.date_range(self.start_time, self.end_time, freq='H')
    #     for timestamp in timestamps:
    #         self.trading_strategy.time = timestamp
    #         prices = price_data.loc[timestamp]
    #         s_scores = self.trading_strategy.calculate_s_scores(prices)
    #         signals = self.trading_strategy.generate_signals(s_scores)
    #         for token in signals.index:
    #             self.trade(token, signals.loc[token, 'trading_signal'], prices[token])
    #         self.update_portfolio_value(prices)

    #     # Calculate returns
    #     returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
    #     # Calculate performance metrics
    #     sharpe_ratio = self.calculate_sharpe_ratio(returns)
    #     max_drawdown = self.calculate_maximum_drawdown(self.portfolio_value)

    #     return {
    #         'portfolio_values': self.portfolio_value,
    #         'sharpe_ratio': sharpe_ratio,
    #         'maximum_drawdown': max_drawdown
    #     }
