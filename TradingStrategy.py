import numpy as np
import pandas as pd
from Utility import Utility

class TradingStrategy:
    def __init__(self, tokens_price, tokens_largest_cap, time, M, sbo=1.25, sso=1.25, sbc=0.75, ssc=0.5):
        self.sbo = sbo  # s-score buy open threshold
        self.sso = sso  # s-score sell open threshold
        self.sbc = sbc  # s-score buy close threshold
        self.ssc = ssc  # s-score sell close threshold
        self.time = time
        self.tokens_price = tokens_price
        self.tokens_largest_cap = tokens_largest_cap
        self.M = M
        self.calculate_factor_returns()
        self.calculate_s_score()

    def calculate_factor_returns(self):
        self.tokens_price_window = Utility.select_window(self.tokens_price, self.M, self.time)
        self.tokens_largest_cap_window = Utility.select_window(self.tokens_largest_cap, self.M, self.time)
        self.common_tokens = Utility.find_tokens(self.tokens_largest_cap_window, self.M)
        self.common_token_prices = self.tokens_price_window[self.common_tokens]
        self.common_token_prices = self.common_token_prices.fillna(method='ffill', axis=0)
        self.hourly_returns = Utility.get_hourly_returns(self.common_token_prices)
        self.correlation_matrix = Utility.calculate_correlation_matrix(self.hourly_returns)
        self.eigenvectors, self.eigenvalues = Utility.principal_component_analysis(self.correlation_matrix)
        self.eigen_vectors_portfolios = Utility.compute_eigenportfolios(self.eigenvectors, self.hourly_returns, self.time)
        self.factors_return = Utility.calculate_factor_returns(self.hourly_returns, self.eigen_vectors_portfolios)
        self.regression_coeffs_df, self.residuals_df = Utility.estimate_residual_return(self.hourly_returns, self.factors_return)


    def calculate_s_score(self):
        """
        Calculate the s-score for a token.
        """
        self.ou_parameters = Utility.calculate_ou_parameters(self.residuals_df)
        self.s_score = - self.ou_parameters['m'] / self.ou_parameters['sigma_eq']

    def generate_signals(self):
        """
        Generate buy/sell signals based on s-scores and return as a DataFrame.
        """
        signals_data = []
        for token, s_score in self.s_score.items():
            # Implement the logic based on the paper's rules
            if s_score < -self.sbo:
                signal = 'buy_open'
            elif s_score > self.sso:
                signal = 'sell_open'
            elif s_score < self.sbc and s_score > -self.ssc:  # Close short if s_score is less than s_bc
                signal = 'close_all'
            elif s_score > -self.ssc:  # Close long if s_score is greater than -s_sc
                signal = 'close_long'
            elif s_score < self.sbc:
                signal = 'close_short'
            else:
                signal = 'hold'

            # Create a dictionary for each token and its signal, including the time and s-score
            signals_data.append({
                'time': self.time,
                'Token': token,
                'trading_signal': signal,
                's_score': s_score
            })

        # Convert the list of dictionaries to a DataFrame
        signals_df = pd.DataFrame(signals_data)
        return signals_df
