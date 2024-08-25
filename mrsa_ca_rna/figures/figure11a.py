"""This file will plot the variance explained by the factors of the tensor factorization"""

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.import_data import concat_datasets, extract_time_data
from mrsa_ca_rna.figures.base import setupBase

import numpy as np
import pandas as pd
import seaborn as sns


def figure11a_setup(disease_data=None, time_data=None):
    """Set up the data for the tensor factorization of both disease and time datasets
    and return the reconstruction errors to make R2X plots"""

    if disease_data is None:
        disease_data = concat_datasets(scaled=(True, 0), tpm=True)
    if time_data is None:
        time_data = extract_time_data(scaled=(True, 0))

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

    fig_size = (8, 16)
    layout = {"ncols": 2, "nrows": 4}
    ax, f, _ = setupBase(fig_size, layout)

    cases = [0, 1, "f", "s"]

    for j, case in enumerate(cases):
        disease_data = concat_datasets(scaled=(True, case), tpm=True)
        time_data = extract_time_data(scaled=(True, case))

        r2x_d, r2x_t = figure11a_setup()

        x_ax_label = "Rank"
        y_ax_label = "R2X"

        a = sns.barplot(x=range(1, 21), y=r2x_d, ax=ax[0 + j * 2])
        a.set_title("Disease Data R2X\nn_max_iter=2000")
        a.set_xlabel(x_ax_label)
        a.set_ylabel(y_ax_label)

        b = sns.barplot(x=range(1, 3), y=r2x_t, ax=ax[1 + j * 2])
        b.set_title("Time Data R2X\nn_max_iter=2000")
        b.set_xlabel(x_ax_label)
        b.set_ylabel(y_ax_label)

    return f
