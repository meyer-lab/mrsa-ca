"""
This file is the start of analyzing the loadings of the
mrsa+ca+healthy data, based on previous scores analysis.
This file may become obsolete post scores heatmap analysis currently
planned.

To-do:
    homogenize structure with figure01 once I figure that one out.
"""

import seaborn as sns

from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.import_data import concat_datasets


def genFig():
    adata = concat_datasets(scale=True, tpm=True)
    df = adata.to_df()

    _, loadings, _ = perform_pca(df)

    desired_components = range(4)

    data = loadings.iloc[desired_components]
    data = data.T

    # pairplot of loadings
    f = sns.pairplot(data)

    return f
