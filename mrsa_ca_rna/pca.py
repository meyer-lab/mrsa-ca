"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Make this prettier by starting out with fully concatenated datasets
    from import data.

"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mrsa_ca_rna.import_data import concat_datasets

import pandas as pd
import numpy as np


def perform_PCA(data: pd.DataFrame = None):
    """
    Perform pca decomposition on supplied data, or performs pca on data from concatenated
    databases if none given.

    Returns:
        scores (pd.DataFrame): the scores matrix of the data as a result of PCA
        loadings (pd.DataFrame): the loadings matrix of the data as a result of PCA
        pca (object): the PCA object for further use in the code.
    """

    if data is None:
        rna_dfmi = concat_datasets()
        rna_mat = rna_dfmi["rna"]
        meta = rna_dfmi["meta"]
        specific = True
    else:
        rna_mat = data
        specific = False

    components = 70
    pca = PCA(n_components=components)
    scaler = StandardScaler().set_output(transform="pandas")

    scaled_rna = scaler.fit_transform(rna_mat)
    rna_decomp = pca.fit_transform(scaled_rna)

    column_labels = []
    for i in range(1, components + 1):
        column_labels.append("PC" + str(i))

    scores = pd.DataFrame(rna_decomp, index=rna_mat.index, columns=column_labels)

    rows = []
    for i in range(pca.n_components_):
        rows.append("PC" + str(i + 1))

    loadings = pd.DataFrame(pca.components_, index=rows, columns=rna_mat.columns)

    return scores, loadings, pca
