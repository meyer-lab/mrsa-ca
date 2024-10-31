"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Make this prettier by starting out with fully concatenated datasets
    from import data.

"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_pca(data: pd.DataFrame, scale: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Perform PCA on the given data.

    Args:
        data (pd.DataFrame): The input data for PCA.

    Returns:
        scores (pd.DataFrame): The scores matrix of the data as a result of PCA.
        loadings (pd.DataFrame): The loadings matrix of the data as a result of PCA.
        pca (PCA): The PCA object for further use in the code.
    """
    components: int = 50
    pca = PCA(n_components=components)
    if scale:
        scaler: StandardScaler = StandardScaler().set_output(transform="pandas")
        scaled_rna = scaler.fit_transform(data)
        rna_decomp = pca.fit_transform(scaled_rna)
    else:
        rna_decomp = pca.fit_transform(data)

    pc_labels = [f"PC{i}" for i in range(1, components + 1)]
    pc_labels_index = pd.Index(pc_labels)

    scores = pd.DataFrame(rna_decomp, index=data.index, columns=pc_labels_index)
    loadings = pd.DataFrame(
        pca.components_, index=pc_labels_index, columns=data.columns
    )

    return scores, loadings, pca
