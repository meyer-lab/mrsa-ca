"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Make this prettier by starting out with fully concatenated datasets
    from import data.

"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca(
    data: pd.DataFrame, components: int = 70
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Column-wise z-score the data, then perform PCA on it.

    Args:
        data (pd.DataFrame): The input data for PCA.
        components (int): The number of components to use for PCA.

    Returns:
        scores (pd.DataFrame): The scores matrix of the data as a result of PCA.
        loadings (pd.DataFrame): The loadings matrix of the data as a result of PCA.
        pca (PCA): The PCA object used to perform the analysis.
    """

    pca = PCA(n_components=components)
    scaler: StandardScaler = StandardScaler().set_output(transform="pandas")

    scaled_rna = scaler.fit_transform(data)
    rna_decomp = pca.fit_transform(scaled_rna)

    pc_labels_index = pd.Index(range(1, components + 1), name="PC")

    scores = pd.DataFrame(rna_decomp, index=data.index, columns=pc_labels_index)
    loadings = pd.DataFrame(
        pca.components_, index=pc_labels_index, columns=data.columns
    )

    return scores, loadings, pca
