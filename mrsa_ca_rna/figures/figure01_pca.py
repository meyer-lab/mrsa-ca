"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:

    Get pairplot to work with base.py and general plotting procedure

"""

import numpy as np
import pandas as pd

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns


def genFig():
    """
    Used to contain the regular plotting skeleton but I am
    struggling to make it compatible with seaborn.pairplot
    so it's being removed for now so I can quickly graph
    scores.
    """

    scores, _, _ = perform_PCA()

    desired_components = pd.IndexSlice["components", scores["components"].columns[0:6]]

    data: pd.DataFrame = scores.loc[:, desired_components]
    data[("meta", "disease")] = scores.index.get_level_values(
        0
    )  # seaborn likes long-form data, make a disease column

    # seaborn only uses the top level so we need to remove our current one so it can see our lower one.
    f = sns.pairplot(data.droplevel(0, 1), hue="disease", palette="viridis")
    return f


genFig()
