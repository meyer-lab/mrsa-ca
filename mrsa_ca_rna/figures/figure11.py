"""This file plots the data pf2 reconstruction for the disease and time datasets"""

import seaborn as sns
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import (
    ca_data_split,
    concat_datasets,
)


def figure11_setup():
    """Set up the data for the tensor factorization of both disease and time datasets
    and return the reconstruction errors to make R2X plots"""

    # data import, concatenation, scaling, and preparation
    # same as figure12_setup
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    # import time dataset (CA)
    time_data, _, _ = ca_data_split()
    time_data.X = StandardScaler().fit_transform(time_data.X)

    # split and organize into xarray datasets along corresponding expansion dimensions
    # (disease->disease, time->subject_id)
    disease_xr = prepare_data(disease_data, expansion_dim="disease")
    time_xr = prepare_data(time_data, expansion_dim="subject_id")

    # change ranks_d back to range(1, 11) when running the full dataset!
    ranks_d = range(1, 21)
    ranks_t = range(1, 21)

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

    # change range(1, 4) back to range(1, 11) when running the full dataset!
    a = sns.barplot(x=range(1, len(r2x_d) + 1), y=r2x_d, ax=ax[0])
    a.set_title("Disease Data R2X")
    a.set_xlabel(x_ax_label)
    a.set_ylabel(y_ax_label)

    b = sns.barplot(x=range(1, len(r2x_t) + 1), y=r2x_t, ax=ax[1])
    b.set_title("Time Data R2X")
    b.set_xlabel(x_ax_label)
    b.set_ylabel(y_ax_label)

    return f
