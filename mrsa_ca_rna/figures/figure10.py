"""This file will plot the factor matrices of the CA time data"""

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import concat_datasets, extract_time_data
from mrsa_ca_rna.figures.base import setupBase

import numpy as np
import pandas as pd
import seaborn as sns


def figure10_setup():
    """Set up the data for the tensor factorization and return the results"""

    time_data = extract_time_data()

    time_xr = prepare_data(time_data, expansion_dim="subject_id")

    tensor_decomp, _ = perform_parafac2(time_xr, rank=2)
    time_factors = tensor_decomp[1]

    return time_factors


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 8)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    time_data = extract_time_data()

    time_factors, rec_errors = figure10_setup()

    time_ranks = range(1, 3)
    # x axis label: rank
    x_ax_label = "Rank"

    # y axis labels: subject_id, eigen, genes
    t_ax_labels = ["Subject ID", "Rank", "Genes"]

    # tick labels: subject_id, rank, genes
    time_labels = [time_data.obs["subject_id"].unique(), time_ranks, 500]

    # plot heatmap of disease factors
    for i, factor in enumerate(time_factors):
        a = sns.heatmap(
            factor,
            ax=ax[i],
            cmap="viridis",
            xticklabels=time_ranks,
            yticklabels=time_labels[i],
        )
        a.set_title(f"Disease Factor Matrix {i+1}")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(t_ax_labels[i])
        # a.set_xticklabels(disease_ranks)
        # a.set_yticklabels(disease_labels[i])

    return f


# d_f = [1, 2, 3]
# t_f = [4, 5]
# d_l = ["d1", "d2", "d3"]
# t_l = ["t1", "t2"]

# zip1, zip2 = zip([d_f, t_f], [d_l, t_l])
# print(zip1, zip2)
