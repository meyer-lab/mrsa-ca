"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    After performing PCA, when creating scores and loadings df's,
        add relevant metadata into these df's because they'll be
        used for graphing all sorts of things and it will just be
        easier to have the data available in the df's.
"""

from sklearn.decomposition import PCA
from mrsa_ca_rna.import_data import concat_datasets
import pandas as pd


def perform_PCA():
    """
    Perform pca analysis on concatenated rna matrix, then attach corresponding patient metadata

    Returns:
        scores (pd.DataFrame): the scores matrix of the concatenated datasets as a result of PCA
        loadings (pd.DataFrame): the loadings matrix of the concatenated datasets as a result of PCA
        pca (object): the PCA object for further use in the code. Might remove once I finish changing perform_PCA
    """

    rna_mat, meta_mat = concat_datasets()
    components = 100  # delta percent explained drops below 0.1% @ ~component 70
    pca = PCA(n_components=components)
    rna_decomp = pca.fit_transform(rna_mat)

    column_labels = []
    for i in range(1, 101):
        column_labels.append("PC" + str(i))

    scores = pd.DataFrame(rna_decomp, rna_mat.index, column_labels)

    # add disease type (mrsa, ca, healthy) and persistance metadata to scores
    scores = pd.concat(
        [meta_mat, scores], axis=1
    )
    scores.dropna(axis=0, inplace=True) # some mrsa patients did not have rna data

    rows = []
    for i in range(pca.n_components_):
        rows.append("PC" + str(i + 1))

    loadings = pd.DataFrame(pca.components_, index=rows, columns=rna_mat.columns)

    return scores, loadings, pca
