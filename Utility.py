import pandas as pd
import sklearn
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class Utility:
    """A utility class with static methods."""

    def __init__(self):
        """Private constructor to prevent instantiation."""
        raise NotImplementedError("This class is not meant to be instantiated")

    @staticmethod
    def select_window(df, M: int, time):
        return df.loc[time - timedelta(hours=M):time]

    @staticmethod
    def find_tokens(df, M: int):
        """Find tokens that appear more than M*0.8 times"""
        all_tokens = df.iloc[:, 1:].values.flatten()
        token_counts = pd.Series(all_tokens).value_counts()
        threshold = M * 0.8
        frequent_tokens = token_counts[token_counts > threshold].index.tolist()
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
        hourly_returns.dropna(inplace=True)
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
        eigenportfolio1 = eigenvectors[0] / ret_std
        eigenportfolio2 = eigenvectors[1] / ret_std
        
        # Create a DataFrame for the eigenportfolios
        df = pd.DataFrame({
            'time': time,
            'common_tokens': hourly_returns.columns,
            'eigenvectors1': eigenvectors[0],
            'eigenvectors2': eigenvectors[1],
            'eigenportfolio1': eigenportfolio1,
            'eigenportfolio2': eigenportfolio2
        })
        df = df.reset_index(drop=True)
        return df
    
    @staticmethod
    def calculate_factor_returns(hourly_returns, eigenportfolios):
        return hourly_returns.dot(eigenportfolios)

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
            X = factors_return[['PC1', 'PC2']]  # the two principal components as independent variables
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

    
    def calculate_ou_parameters(df):
        model = LinearRegression()
        ou_parameters = pd.DataFrame(index=df.columns, columns=['kappa', 'm', 'sigma', 'sigma_eq'])

        for token in df:
            X_k = df[token].cumsum()

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
            b = model.coef_[0]    # Coef is a 1D array, get the first element

            # Calculate variance of residuals
            residuals = X_n_plus_1 - model.predict(X_n)
            residuals_variance = np.var(residuals, ddof=1)

            # Calculate OU process parameters for the token
            kappa = -np.log(b) * 8760  # Assuming 8760 hours in a year
            m = a / (1 - b)
            sigma = np.sqrt(residuals_variance * 2 * kappa / (1 - b**2))
            sigma_eq = np.sqrt(residuals_variance / (1 - b**2))

            # Store results in the DataFrame
            ou_parameters.loc[token] = [kappa, m, sigma, sigma_eq]
        
        return ou_parameters
