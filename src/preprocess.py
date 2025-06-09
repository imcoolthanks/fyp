# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Data Analysis Functions
# -----------------------------

def compute_log_return(df):
    """
    Compute the logarithmic return of each asset in the DataFrame.

    Parameters:
        df (pd.DataFrame): A DataFrame where each column represents an asset's price over time.

    Returns:
        log_return_df (pd.DataFrame): Logarithmic returns of the input price data.
    """
    log_return_df = pd.DataFrame()

    for asset in df.columns:
        log_return_df[asset] = np.log(df[asset] / df[asset].shift(1))

    # Drop the first row which contains NaN values
    log_return_df.dropna(inplace=True)

    return log_return_df

def standardize_df(df):
    """
    Standardize the DataFrame using zero mean and unit variance.

    Parameters:
        df (pd.DataFrame): Input DataFrame to standardize.

    Returns:
        standardized_data (np.ndarray): Standardized data.
    """
    return StandardScaler().fit_transform(df)

def compute_pca(df, percentage_variance=0.99):
    """
    Apply PCA on the standardized data.

    Parameters:
        df (np.ndarray): Standardized data.
        percentage_variance (float): Desired explained variance to retain.

    Returns:
        pca (PCA object): Fitted PCA object.
        pca_components_df (np.ndarray): Transformed PCA components.
    """
    pca = PCA(percentage_variance)
    pca_components_df = pca.fit_transform(df)

    return pca, pca_components_df

def explain_pca(pca, pca_components, asset=""):
    """
    Print PCA explanation and show a scatter plot of the first two PCA components.

    Parameters:
        pca (PCA object): Fitted PCA object.
        pca_components (np.ndarray): Transformed data using PCA.
        asset (str): Optional label for the asset.
    """
    if asset:
        print(f"Asset: {asset}")

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Scatter plot of the first two principal components
    plt.title("PCA Components")
    plt.scatter(pca_components[:, 0], pca_components[:, 1], label=asset, marker='o')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# Binning Functions
# -----------------------------

def bin_data_with_quartiles(data, num_bins):
    """
    Bin the data into quantile-based bins.

    Parameters:
        data (np.ndarray or pd.DataFrame): Input data to bin.
        num_bins (int): Number of quantile bins to divide the data into.

    Returns:
        binned_data (np.ndarray): Quantile-based binned data.
    """
    data = np.array(data)
    _, d = data.shape

    binned_data = np.zeros_like(data)

    for i in range(d):
        # Use pandas qcut to bin into equal-sized quantile bins
        binned_data[:, i] = pd.qcut(data[:, i], q=num_bins, labels=False)

    return binned_data
