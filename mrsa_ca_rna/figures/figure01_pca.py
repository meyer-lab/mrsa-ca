"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:

    Get pairplot to work with base.py and general plotting procedure

"""

import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase


def genFig():
    """
    Used to contain the regular plotting skeleton but I am
    struggling to make it compatible with seaborn.pairplot
    so it's being removed for now so I can quickly graph
    scores.
    """

    # bring in the rna anndata objects and push them to dataframes for perform_PCA()
    adata = concat_datasets(scaled=False, tpm=True)

    df = adata.to_df()

    scores, _, _ = perform_PCA(df)

    desired_components = pd.IndexSlice["components", scores["components"].columns[0:6]]

    data: pd.DataFrame = scores.loc[:, desired_components]
    data[("meta", "disease")] = scores.index.get_level_values(
        0
    )  # seaborn likes long-form data, make a disease column

    # seaborn only uses the top level so we need to remove our current one so it can see our lower one.
    f = sns.pairplot(data.droplevel(0, 1), hue="disease", palette="viridis")
    return f
