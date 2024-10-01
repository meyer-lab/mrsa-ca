"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Make this prettier by starting out with fully concatenated datasets
    from import data.

"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd


def perform_PCA(data: pd.DataFrame):
    """
    Scale and perform principle component analysis on either provided
    data or on the default dataset returned by concat_dataset().

    Parameters:
        data (pd.DataFrame): the data to perform PCA on

    Returns:
        scores (pd.DataFrame): the scores matrix of the data as a result of PCA
        loadings (pd.DataFrame): the loadings matrix of the data as a result of PCA
        pca (object): the PCA object for further use in the code.
    """

    components = 70
    pca = PCA(n_components=components)
    scaler: StandardScaler = StandardScaler().set_output(transform="pandas")

    scaled_rna = scaler.fit_transform(data)
    rna_decomp = pca.fit_transform(scaled_rna)

    pc_labels = [f"PC{i}" for i in range(1, components + 1)]

    scores = pd.DataFrame(rna_decomp, index=data.index, columns=pc_labels)
    loadings = pd.DataFrame(pca.components_, index=pc_labels, columns=data.columns)

    return scores, loadings, pca
