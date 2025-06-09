"""The figures produced by this file will attempt to show the central hypothesis of the paper:
Data decomposition can be used to identify the directionality of a disease. This
directionality describes important features of the disease when related back to 
the original data in gene space. When comparing these features across diseases,
the shared directionality of two diseases can be used to identify commonalities
in the diseases. To find this shared directionality, we can concatenate the
two disease datasets and perform PCA on the combined dataset. We should observe
components that not only separate the two diseases, but also show a clear similarity
in the directionalyity of the two diseases.

To accomplish this, we will perform PCA on the combined dataset (MRSA and Candidemia)
and then perform logistic regression on the PCA components to identify the most
important components for determining the disease status. The mosty important component
will likely be one that shows shared directionality between the two diseases, while 
the first component will likely be one that shows the most separation between the two.
From here, we can identify the genes that are most important for each component. We can
do a pairwise comparison of the genes that are most important for each component to
identify the a gene pair that provide a good example of both shared and non-shared
directionality. We can then plot the the two diseases separately in this gene pair space
with PCA components overlaid. This will allow us to see the directionality of the two
diseases serparately. Then we can concatenate the two diseases and make the same scatter
plot with the PCA components overlaid. This will allow us to see the directionality that
is both shared and non-shared between the two diseases."""

import pandas as pd
import anndata as ad
import seaborn as sns

from mrsa_ca_rna.utils import concat_datasets
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.regression import perform_LR
from mrsa_ca_rna.figures.base import setupBase


 # Placeholder plotting functions
def plot_coefficients():
    # Placeholder for plotting coefficients
    pass

def plot_pairwise_comparison():
    # Placeholder for plotting pairwise comparison
    pass

def plot_directionality():
    # Placeholder for plotting directionality
    pass


def genFig():

    datasets = ["mrsa", "ca"]
    diseases = ["MRSA", "Candidemia"]
    combined_ad = concat_datasets(
        datasets,
        diseases,
        scale=False,
    )

    mrsa_data = combined_ad[combined_ad.obs["disease"] == "MRSA"].copy()
    ca_data = combined_ad[combined_ad.obs["disease"] == "Candidemia"].copy()

    # Setup and plot coefficients
    size_coef = (10, 6)
    layout_coef = {"ncols": 1, "nrows": 2}
    ax_coef, f_coef, _ = setupBase(size_coef, layout_coef)
    plot_coefficients(ax_coef, combined_ad)

    # Setup and plot pairwise comparison
    size_pair = (10, 6)
    layout_pair = {"ncols": 1, "nrows": 2}
    ax_pair, f_pair, _ = setupBase(size_pair, layout_pair)
    plot_pairwise_comparison(ax_pair, combined_ad)

    # Setup and plot directionality
    size_dir = (10, 6)
    layout_dir = {"ncols": 1, "nrows": 2}
    ax_dir, f_dir, _ = setupBase(size_dir, layout_dir)
    plot_directionality(ax_dir, combined_ad, mrsa_data, ca_data)


    return f_coef, f_pair, f_dir
