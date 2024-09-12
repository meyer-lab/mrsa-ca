"""This file plots the data pf2 reconstruction for the disease and time datasets
This must be updated with all the currently used disease data! (Everything included in figure12)"""

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import concat_datasets, ca_data_split
from mrsa_ca_rna.figures.base import setupBase

import seaborn as sns


def figure11_setup():
    """Set up the data for the tensor factorization of both disease and time datasets
    and return the reconstruction errors to make R2X plots"""

    disease_data = concat_datasets(scale=True, tpm=True)
    time_data, _, _ = ca_data_split(scale=True, tpm=True)

    disease_xr = prepare_data(disease_data, expansion_dim="disease")
    time_xr = prepare_data(time_data, expansion_dim="subject_id")

    ranks_d = range(1, 21)
    ranks_t = range(1, 3)

    r2x_d = []
    r2x_t = []

    for rank_d in ranks_d:
        _, rec_errors_d = perform_parafac2(disease_xr, rank=rank_d)
        r2x_d.append(1 - min(rec_errors_d))

    for rank_t in ranks_t:
        _, rec_errors_t = perform_parafac2(time_xr, rank=rank_t)
        r2x_t.append(1 - min(rec_errors_t))

    return r2x_d, r2x_t


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (8, 4)
    layout = {"ncols": 2, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    r2x_d, r2x_t = figure11_setup()

    x_ax_label = "Rank"
    y_ax_label = "R2X"

    a = sns.barplot(x=range(1, 21), y=r2x_d, ax=ax[0])
    a.set_title("Disease Data R2X\nn_max_iter=100")
    a.set_xlabel(x_ax_label)
    a.set_ylabel(y_ax_label)

    b = sns.barplot(x=range(1, 3), y=r2x_t, ax=ax[1])
    b.set_title("Time Data R2X\nn_max_iter=100")
    b.set_xlabel(x_ax_label)
    b.set_ylabel(y_ax_label)

    return f
