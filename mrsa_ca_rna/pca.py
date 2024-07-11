"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:


"""

from sklearn.decomposition import PCA
from mrsa_ca_rna.import_data import concat_datasets, validation_data
import pandas as pd


def perform_PCA():
    """
    Perform pca analysis on concatenated rna matrix, then attach corresponding patient metadata

    Returns:
        scores (pd.DataFrame): the scores matrix of the concatenated datasets as a result of PCA
        loadings (pd.DataFrame): the loadings matrix of the concatenated datasets as a result of PCA
        pca (object): the PCA object for further use in the code.
    """

    rna_mat, meta_mat = concat_datasets()
    components = 100
    pca = PCA(n_components=components)
    rna_decomp = pca.fit_transform(rna_mat)

    column_labels = []
    for i in range(1, components + 1):
        column_labels.append("PC" + str(i))

    scores = pd.DataFrame(rna_decomp, rna_mat.index, column_labels)

    # add disease type (mrsa, ca, healthy) and persistance metadata to scores
    scores = pd.concat([meta_mat, scores], axis=1)
    scores.dropna(axis=0, inplace=True)  # some mrsa patients did not have rna data

    rows = []
    for i in range(pca.n_components_):
        rows.append("PC" + str(i + 1))

    loadings = pd.DataFrame(pca.components_, index=rows, columns=rna_mat.columns)

    return scores, loadings, pca


def perform_PCA_validation():
    """
    Performs PCA on the validation data

    Returns:
        scores (pandas.DataFrame): the scores matrix as a result of PCA
        loadings (pandas.DataFrame): the loadings matrix as a result of PCA
        pca (object): the PCA object for further access
    """

    val_rna = validation_data()

    components = 60
    pca = PCA(n_components=components)
    val_decomp = pca.fit_transform(val_rna.iloc[:, 2:])

    column_labels = []
    for i in range(1, components + 1):
        column_labels.append("PC" + str(i))
    rows = []
    for i in range(pca.n_components_):
        rows.append("PC" + str(i + 1))

    scores = pd.DataFrame(val_decomp, val_rna.index, column_labels)
    scores = pd.concat([val_rna.iloc[:, :2], scores], axis=1)
    loadings = pd.DataFrame(pca.components_, index=rows, columns=val_rna.columns[2:])

    return scores, loadings, pca
