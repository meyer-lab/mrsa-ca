"""File plots the pacmap of the projections and weighted projections of the pf2
disease data"""

# main module imports
import numpy as np
import pandas as pd
import seaborn as sns

# secondary module imports
# local module imports
from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.map import perform_pacmap


def figure13_setup():
    """Set up the data for the pacmap and return the results"""

    """data import, concatenation, scaling, and preparation"""
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    disease_xr = prepare_data(disease_data, expansion_dim="disease")

    # Perform parafac2 factorization, pull out the factors and projections
    tensor_decomp, _ = perform_parafac2(disease_xr, rank=50, l1=0.5)
    disease_factors = tensor_decomp[1]
    disease_projections = tensor_decomp[2]

    # Make the weighted projections and perform the pacmap
    weighted_projections = [x @ disease_factors[1] for x in disease_projections]

    mapped_p = perform_pacmap(disease_projections)
    mapped_wp = perform_pacmap(weighted_projections)

    return mapped_p, mapped_wp, disease_data


def genFig():
    fig_size = (8, 4)
    layout = {"ncols": 2, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    mapped_p, mapped_wp, disease_data = figure13_setup()

    # Ensure the disease array is the correct shape for hstack
    disease_array = disease_data.obs["disease"].to_numpy().reshape(-1, 1)

    # Create a dataframe to label the diseases for scatterplot
    data_p = pd.DataFrame(
        np.hstack((disease_array, mapped_p)),
        columns=pd.Index(["Disease", "PacMAP 1", "PacMAP 2"]),
    )
    data_wp = pd.DataFrame(
        np.hstack((disease_array, mapped_wp)),
        columns=pd.Index(["Disease", "PacMAP 1", "PacMAP 2"]),
    )
    data = [data_p, data_wp]

    for i, d in enumerate(data):
        a = sns.scatterplot(d, x="PacMAP 1", y="PacMAP 2", hue="Disease", ax=ax[i])
        a.set_title("Weighted Projections" if i == 1 else "Projections")
        a.set_xlabel("PacMAP 1")
        a.set_ylabel("PacMAP 2")

    return f


genFig()
