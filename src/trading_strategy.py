import pandas as pd
import numpy as np
from src.utility import Utility
from sklearn.linear_model import LinearRegression


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

        # Attributes waiting to be calculated
        self._tokens_price_window = None
        self._tokens_largest_cap_window = None
        self._common_tokens = None
        self._common_token_prices = None
        self._hourly_returns = None
        self._correlation_matrix = None
        self._eigenvectors = None
        self._eigenvalues = None
        self._eigen_vectors_portfolios = None
        self._factors_return = None
        self._residual_return = None
        self._ou_parameters = None
        self._s_score = None

    @property
    def tokens_price_window(self):
        if self._tokens_price_window is None:
            self._tokens_price_window = Utility.select_window(self.tokens_price, self.M, self.time)
        return self._tokens_price_window

    @property
    def tokens_largest_cap_window(self):
        if self._tokens_largest_cap_window is None:
            self._tokens_largest_cap_window = Utility.select_window(self.tokens_largest_cap, self.M, self.time)
        return self._tokens_largest_cap_window

    @property
    def common_tokens(self):
        if self._common_tokens is None:
            self._common_tokens = Utility.find_tokens(self.tokens_largest_cap_window, self.M)
        return self._common_tokens

    @property
    def common_token_prices(self):
        if self._common_token_prices is None:
            self._common_token_prices = self.tokens_price_window.loc[:, self.common_tokens].fillna(method='ffill',
                                                                                                   axis=0)
        return self._common_token_prices

    @property
    def hourly_returns(self):
        if self._hourly_returns is None:
            self._hourly_returns = Utility.get_hourly_returns(self.common_token_prices)
        return self._hourly_returns

    @property
    def correlation_matrix(self):
        if self._correlation_matrix is None:
            self._correlation_matrix = Utility.calculate_correlation_matrix(self.hourly_returns)
        return self._correlation_matrix

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            self._eigenvectors, _ = Utility.principal_component_analysis(self.correlation_matrix)
        return self._eigenvectors

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            _, self._eigenvalues = Utility.principal_component_analysis(self.correlation_matrix)
        return self._eigenvalues

    @property
    def eigen_vectors_portfolios(self):
        if self._eigen_vectors_portfolios is None:
            self._eigen_vectors_portfolios = Utility.compute_eigenportfolios(self.eigenvectors, self.hourly_returns,
                                                                             self.time)
        return self._eigen_vectors_portfolios

    @property
    def factors_return(self):
        if self._factors_return is None:
            self._factors_return = Utility.calculate_factor_returns(self.hourly_returns, self.eigen_vectors_portfolios)
        return self._factors_return

    @property
    def residual_return(self):
        if self._residual_return is None:
            _, self._residual_return = Utility.estimate_residual_return(self.hourly_returns, self.factors_return)
        return self._residual_return

    @property
    def ou_parameters(self):
        if self._ou_parameters is None:
            temp = self.residual_return.columns
            self._ou_parameters = pd.DataFrame(index=temp, columns=['kappa', 'm', 'sigma', 'sigma_eq'])
            model = LinearRegression()

            for token in temp:
                X_k = self.residual_return[token].cumsum()
                # Define X_n and X_n_plus_1 for the token
                X_n = X_k[:-1]
                X_n_plus_1 = X_k[1:]

                # Reshape X_n for sklearn compatibility
                X_n = X_n.values.reshape(-1, 1)
                X_n_plus_1 = X_n_plus_1.values

                # Fit the linear regression model for the token
                model.fit(X_n, X_n_plus_1)

                # Get the regression parameters 'a' and 'b'
                a = model.intercept_  # Intercept is a scalar
                b = model.coef_[0]  # b parameter is a 1D array, get the first element

                # Calculate variance of residuals
                residuals = X_n_plus_1 - model.predict(X_n)
                residuals_variance = np.var(residuals, ddof=1)

                # Calculate OU process parameters for the token
                kappa = -np.log(b) * 8760  # Assuming 8760 hours in a year
                m = a / (1 - b)
                sigma = np.sqrt(residuals_variance * 2 * kappa / (1 - b ** 2))
                sigma_eq = np.sqrt(residuals_variance / (1 - b ** 2))

                # Store results in the DataFrame
                self._ou_parameters.loc[token, :] = [kappa, m, sigma, sigma_eq]

        return self._ou_parameters

    @property
    def s_score(self):
        """
        Calculate the s-score for a token.
        """
        if self._s_score is None:
            self._s_score = - self.ou_parameters['m'] / self.ou_parameters['sigma_eq']
        return self._s_score

    def generate_signals(self):
        """
        Generate buy/sell signals based on s-scores and return as a DataFrame.
        """
        signals_data = []
        for token, s_score in self.s_score.items():
            # Implement the logic based on the paper's rules
            # Open short means we buy one dollar of the corresponding stock and sell beta dollars of BTC
            if s_score > self.sso:
                signal = 'open_short'
            elif s_score < -self.sbo:
                signal = 'open_long'
            elif self.sbc > s_score > -self.ssc:
                signal = 'close_short/long'
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

#%%
