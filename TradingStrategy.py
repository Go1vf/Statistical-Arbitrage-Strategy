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
        self.tokens_price_window = Utility.select_window(tokens_price, M, time)
        self.tokens_largest_cap_window = Utility.select_window(tokens_largest_cap, M, time)
        self.common_tokens = Utility.find_tokens(tokens_largest_cap, M)

    def calculate_factor_returns(self):
        self.common_token_prices = self.tokens_price_window[self.common_tokens]
        self.common_token_prices.fillna(method='ffill', axis=0, inplace=True)
        self.hourly_returns = Utility.get_hourly_returns(self.common_token_prices)
        self.correlation_matrix = Utility.calculate_correlation_matrix(self.hourly_returns)
        self.eigenvectors, self.eigenvalues = Utility.principal_component_analysis(self.correlation_matrix)
        self.eigen_vectors_portfolios = Utility.compute_eigenportfolios(self.eigenvectors, self.hourly_returns, self.time)
        self.factors_return_1 = Utility.calculate_factor_returns(self.hourly_returns, self.eigen_vectors_portfolios['eigenportfolio1'])
        self.factors_return_2 = Utility.calculate_factor_returns(self.hourly_returns, self.eigen_vectors_portfolios['eigenportfolio2'])
        self.regression_coeffs_df, self.residuals_df = Utility.estimate_residual_return(self.hourly_returns, factors_return)
        


    def calculate_s_score(self):
        """
        Calculate the s-score for a token.
        """

        self.s_score = - ou_parameters['m'] / ou_parameters['sigma_eq']

    def generate_signals(self, s_scores):
        """
        Generate buy/sell signals based on s-scores and return as a DataFrame.
        """
        signals_data = []
        for token, s_score in s_scores.items():
            if s_score > self.sbo:
                signal = 'buy_open'
            elif s_score < -self.sso:
                signal = 'sell_open'
            elif self.sbc < s_score < -self.ssc:
                signal = 'close_position'
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
