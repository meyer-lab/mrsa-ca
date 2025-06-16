"""This file will plot the projections of the PF2 model
to observe patient distributions in the latent space.
We suspect there might be a few outliers in the latent space."""

import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    X = concat_datasets()

    rank = 5

    X, _ = perform_parafac2(X, slice_col="disease", rank=rank)

    # Make a weighted projection DataFrame for easier plotting and disease labeling
    wp_df = pd.DataFrame(
        X.obsm["Pf2_projections"],
        index=X.obs.index,
        columns=pd.Index([x for x in range(1, X.obsm["Pf2_projections"].shape[1] + 1)]),
    )
    wp_df["disease"] = X.obs["disease"].values

    return wp_df


def genFig():
    """Generate the figure for the projections of the PF2 model"""

    # Get the weighted projections
    weighted_projections = figure_setup()

    layout = {"ncols": 4, "nrows": 4}
    fig_size = (16, 16)
    ax, f, _ = setupBase(fig_size, layout)

    cmap = sns.diverging_palette(145, 300, as_cmap=True)

    # For each disease, plot the weighted projections
    for i, disease in enumerate(weighted_projections["disease"].unique()):
        # Subset the DataFrame for the current disease
        projection: pd.DataFrame = weighted_projections.loc[
            weighted_projections["disease"] == disease
        ].drop(columns=["disease"])

        a = sns.heatmap(
            projection,
            ax=ax[i],
            cmap=cmap,
            center=0,
            cbar=True,
            xticklabels=projection.columns.to_list(),
            yticklabels=False,
        )
        a.set_title(f"Weighted Projections for {disease}")
        a.set_xlabel("Components")

    return f
