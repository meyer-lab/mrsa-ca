"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Make this prettier by starting out with fully concatenated datasets
    from import data.

"""

from typing import List
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Perform PCA on the given data.

    Args:
        data (pd.DataFrame): The input data for PCA.

    Returns:
        scores (pd.DataFrame): The scores matrix of the data as a result of PCA.
        loadings (pd.DataFrame): The loadings matrix of the data as a result of PCA.
        pca (PCA): The PCA object for further use in the code.
    """
    components: int = 70
    pca = PCA(n_components=components)
    scaler: StandardScaler = StandardScaler().set_output(transform="pandas")

    scaled_rna = scaler.fit_transform(data)
    rna_decomp = pca.fit_transform(scaled_rna)

    # Create column labels for the scores DataFrame
    column_labels: List[str] = [f"PC{i}" for i in range(1, components + 1)]

    # Explicitly cast column_labels to pd.Index for pyright
    column_labels_index = pd.Index(column_labels)

    # Create the scores DataFrame
    scores = pd.DataFrame(rna_decomp, index=data.index, columns=column_labels_index)

    # Create row labels for the loadings DataFrame
    rows_labels: List[str] = [f"PC{i + 1}" for i in range(pca.n_components_)]

    # Explicitly cast rows_labels to pd.Index for pyright
    rows_labels_index = pd.Index(rows_labels)
    
    # Create the loadings DataFrame
    loadings = pd.DataFrame(pca.components_, index=rows_labels_index, columns=data.columns)

    return scores, loadings, pca
