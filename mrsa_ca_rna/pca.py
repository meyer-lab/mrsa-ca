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


def perform_PCA(data :pd.DataFrame=None):
    """
    Perform pca analysis on concatenated rna matrix, then attach corresponding patient metadata

    Returns:
        scores (pd.DataFrame): the scores matrix of the concatenated datasets as a result of PCA
        loadings (pd.DataFrame): the loadings matrix of the concatenated datasets as a result of PCA
        pca (object): the PCA object for further use in the code.
    """

    if data is None:
        rna_mat = concat_datasets()
    else:
        rna_mat = data

    components = 70
    pca = PCA(n_components=components)
    scaler = StandardScaler().set_output(transform="pandas")

    scaled_rna = scaler.fit_transform(
        rna_mat.loc[:, ~rna_mat.columns.str.contains("status|disease")]
    )
    rna_decomp = pca.fit_transform(scaled_rna)

    column_labels = []
    for i in range(1, components + 1):
        column_labels.append("PC" + str(i))

    scores = pd.DataFrame(rna_decomp, rna_mat.index, column_labels)

    # add disease type (mrsa, ca, healthy) and persistance metadata to scores
    scores = pd.concat([rna_mat.loc[:, "status":"disease"], scores], axis=1)

    rows = []
    for i in range(pca.n_components_):
        rows.append("PC" + str(i + 1))

    loadings = pd.DataFrame(pca.components_, index=rows, columns=rna_mat.columns[2:])

    return scores, loadings, pca