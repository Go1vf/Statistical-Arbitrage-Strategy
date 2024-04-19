import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from src.trading_strategy import TradingStrategy


class BackTesting:
    def __init__(self, start_time, end_time, M, tokens_largest_cap, tokens_price, initial_capital=10000):
        self.start_time = self._parse_date(start_time)
        self.end_time = self._parse_date(end_time)
        self.tokens_largest_cap = tokens_largest_cap
        self.tokens_price = tokens_price
        self.cash_value = initial_capital
        self.window = M
        self.portfolio = defaultdict(int)  # Use int for default zero
        self.portfolio_value = {}
        self.run_backtest()

    @staticmethod
    def _parse_date(self, date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}") from e

    def run_backtest(self):
        all_signals = []
        all_eigen_vectors_portfolios = []
        current = self.start_time
        while current <= self.end_time:
            strategy = TradingStrategy(self.tokens_price, self.tokens_largest_cap, current, self.window)
            signals = strategy.generate_signals()
            self._trade(signals, current)
            all_signals.append(signals)
            all_eigen_vectors_portfolios.append(strategy.eigen_vectors_portfolios)
            current += timedelta(hours=1)
        self._finalize_data_frames(all_signals, all_eigen_vectors_portfolios)

    def _trade(self, signal, time):
        for index, row in signal.iterrows():
            token, trade_signal = row['Token'], row['trading_signal']
            curr_share = self.portfolio[token]
            curr_prc = self.tokens_price.loc[time, token]

            if trade_signal in ['buy_open', 'sell_open']:
                if trade_signal == 'buy_open' and self.cash_value >= curr_prc:
                    self.portfolio[token] += 1
                    self.cash_value -= curr_prc
                elif trade_signal == 'sell_open':
                    self.portfolio[token] -= 1
                    self.cash_value += curr_prc

            if trade_signal in ['close_long', 'close_short']:
                if (trade_signal == 'close_long' and curr_share > 0) or (trade_signal == 'close_short' and curr_share < 0):
                    self.cash_value += abs(curr_share) * curr_prc
                    self.portfolio[token] = 0

        self._update_portfolio_value(time)

    def _update_portfolio_value(self, time):
        total_value = sum(num_shares * self.tokens_price.loc[time, token] for token, num_shares in self.portfolio.items() if num_shares != 0)
        self.portfolio_value[time] = total_value + self.cash_value

    def _finalize_data_frames(self, all_signals, all_eigen_vectors_portfolios):
        self.portfolio_value = pd.DataFrame(list(self.portfolio_value.items()), columns=['Time', 'PortfolioValue'])
        self.portfolio_value['Return'] = self.portfolio_value['PortfolioValue'].pct_change()
        self.signals_df = pd.concat(all_signals)
        self.eigen_vectors_portfolios_df = pd.concat(all_eigen_vectors_portfolios)

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        excess_returns = self.portfolio_value['Return'] - risk_free_rate
        return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(8760)

    def calculate_maximum_draw_down(self):
        running_max = self.portfolio_value['PortfolioValue'].cummax()
        draw_down = (self.portfolio_value['PortfolioValue'] - running_max) / running_max
        return draw_down.min()

