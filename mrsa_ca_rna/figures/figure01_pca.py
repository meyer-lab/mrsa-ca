"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:

    Get pairplot to work with base.py and general plotting procedure

"""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca


def genFig():
    """
    Used to contain the regular plotting skeleton but I am
    struggling to make it compatible with seaborn.pairplot
    so it's being removed for now so I can quickly graph
    scores.
    """

    # bring in the rna anndata objects and push them to dataframes for perform_pca()
    adata = concat_datasets(scale=False, tpm=True)

    df = adata.to_df()

    scores, _, _ = perform_pca(df)

    desired_components = range(9)

    data: pd.DataFrame = scores.iloc[:, desired_components].copy()
    data["disease"] = adata.obs["disease"].values

    # seaborn only uses the top level so we need to remove our current one
    # so it can see our lower one.
    f = sns.pairplot(data, hue="disease", palette="viridis")
    return f
