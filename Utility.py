import pandas as pd
import sklearn
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
from sklearn.decomposition import PCA

class Utility:
    """A utility class with static methods."""

    def __init__(self):
        """Private constructor to prevent instantiation."""
        raise NotImplementedError("This class is not meant to be instantiated")

    @staticmethod
    def select_window(df, M: int, start_time):
        return df.loc[start_time:start_time + timedelta(hours=M-1)]

    @staticmethod
    def find_tokens(df, M: int):
        """Find tokens that appear more than M*0.8 times"""
        all_tokens = df.iloc[:, 1:].values.flatten()
        token_counts = pd.Series(all_tokens).value_counts()
        threshold = M * 0.8
        frequent_tokens = token_counts[token_counts > threshold].index.tolist()
        return frequent_tokens

    @staticmethod
    def principal_component_analysis(df, n_components=2):
        pca = PCA(n_components=2)
        # Fit the PCA on the hourly returns data
        pca.fit(df)
        # Get the principal components (eigenvectors)
        principal_components = pca.components_
        explained_variance = pca.explained_variance_
        # Construct the eigenportfolios
        eigenportfolios = pd.DataFrame(principal_components.T, index=df.columns, columns=['PC1', 'PC2'])
        return principal_components, explained_variance, eigenportfolios
    
    @staticmethod
    def calculate_factor_returns(hourly_returns, eigenportfolios):
        return hourly_returns.dot(eigenportfolios)



