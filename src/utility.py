import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class Utility:
    """A utility class with static methods."""

    def __init__(self):
        """Private constructor to prevent instantiation."""
        raise NotImplementedError("This class is not meant to be instantiated")

    @staticmethod
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}") from e

    @staticmethod
    def select_window(df, M: int, time):
        return df.loc[time - timedelta(hours=M):time]

    @staticmethod
    def find_tokens(df, M: int):
        """Find tokens that appear more than M*0.8 times"""
        all_tokens = df.iloc[:, 1:].values.flatten()
        token_counts = pd.Series(all_tokens).value_counts()
        threshold = M * 0.8
        frequent_tokens = token_counts[token_counts >= threshold].index.tolist()
        return frequent_tokens

    @staticmethod
    def get_common_token_prices(tokens_price_window, common_tokens: list):
        """Get the price of tokens_price_window"""
        common_token_prices = tokens_price_window[common_tokens]
        common_token_prices.fillna(method='ffill', axis=0, inplace=True)
        return common_token_prices

    @staticmethod
    def get_hourly_returns(df):
        hourly_returns = df.pct_change()
        # Drop the first row as it will be NaN after pct_change()
        hourly_returns = hourly_returns.iloc[1:]
        hourly_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        hourly_returns = hourly_returns.fillna(0)
        return hourly_returns

    @staticmethod
    def calculate_correlation_matrix(df):
        # Z-score normalization
        normalized_df = (df - df.mean()) / df.std()
        # Compute the empirical correlation matrix
        correlation_matrix = normalized_df.corr()
        return correlation_matrix

    @staticmethod
    def principal_component_analysis(df, n_components=2):
        pca = PCA(n_components=2)
        # Fit the PCA on the hourly returns data
        pca.fit(df)
        # Get the principal components (eigenvectors)
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_
        return eigenvectors, eigenvalues

    @staticmethod
    def compute_eigenportfolios(eigenvectors, hourly_returns, time):
        # Calculate standard deviation of hourly returns
        ret_std = hourly_returns.std()
        
        # Normalize each eigenvector by the standard deviation of returns
        eigen_portfolio1 = eigenvectors[0] / ret_std
        eigen_portfolio2 = eigenvectors[1] / ret_std
        
        # Create a DataFrame for the eigen_portfolios
        df = pd.DataFrame({
            'time': time,
            'common_tokens': hourly_returns.columns,
            'eigenvectors1': eigenvectors[0],
            'eigenvectors2': eigenvectors[1],
            'eigen_portfolio1': eigen_portfolio1,
            'eigen_portfolio2': eigen_portfolio2
        })
        df = df.reset_index(drop=True)
        return df
    
    @staticmethod
    def calculate_factor_returns(hourly_returns, eigen_portfolios):
        df = np.dot(hourly_returns, eigen_portfolios[['eigen_portfolio1', 'eigen_portfolio2']])
        df = pd.DataFrame(df, columns=['RiskFactor1', 'RiskFactor2'])
        return df

    @staticmethod
    def estimate_residual_return(hourly_returns, factors_return):
        model = LinearRegression()
        # Initialize dictionaries to store the regression coefficients for each token
        regression_coefficients = {'beta_0': {}, 'beta_1': {}, 'beta_2': {}}
        # Initialize DataFrame for residuals
        residuals_df = pd.DataFrame(index=hourly_returns.index)

        # Loop through each token (column) in the hourly returns DataFrame
        for token in hourly_returns.columns:
            # Prepare the independent variables (X) and dependent variable (y)
            X = factors_return[['RiskFactor1', 'RiskFactor2']]  # the two principal components as independent variables
            y = hourly_returns[token]  # the hourly returns for the token as the dependent variable
            
            # Fit the linear regression model
            model.fit(X, y)
            
            # Store the coefficients
            regression_coefficients['beta_0'][token] = model.intercept_
            regression_coefficients['beta_1'][token] = model.coef_[0]
            regression_coefficients['beta_2'][token] = model.coef_[1]
            
            # Calculate and store the residuals
            residuals_df[token] = y - model.predict(X)

        # Create a DataFrame for the regression coefficients
        regression_coeffs_df = pd.DataFrame(regression_coefficients)

        return regression_coeffs_df, residuals_df

