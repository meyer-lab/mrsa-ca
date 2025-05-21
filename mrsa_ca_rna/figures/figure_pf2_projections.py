"""This file will plot the projections of the PF2 model
to observe patient distributions in the latent space.
We suspect there might be a few outliers in the latent space."""

import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    datasets = "all"
    data = concat_datasets(datasets, scale=True)

    rank = 5

    _, factors, projections, r2x = perform_parafac2(
        data, condition_name="disease", rank=rank
    )

    disease_map = dict(
        zip(data.obs["condition_unique_idxs"], data.obs["disease"], strict=True)
    )

    # Get weighted projections for each disease
    weighted_projections = {}
    for i, projection in enumerate(projections):
        weighted_projections[disease_map[i]] = projection @ factors[1]

    return weighted_projections


def genFig():
    """Generate the figure for the projections of the PF2 model"""

    # Get the weighted projections
    weighted_projections = figure_setup()

    layout = {"ncols": 5, "nrows": 4}
    fig_size = (20, 16)
    ax, f, _ = setupBase(fig_size, layout)

    cmap = sns.diverging_palette(145, 300, as_cmap=True)

    component_labels = [
        str(x) for x in range(1, weighted_projections["MRSA"].shape[1] + 1)
    ]

    for i, (disease, projection) in enumerate(weighted_projections.items()):
        a = sns.heatmap(
            projection,
            ax=ax[i],
            cmap=cmap,
            center=0,
            cbar=True,
            xticklabels=component_labels,
            yticklabels=False,
        )
        a.set_title(f"Weighted Projections for {disease}")
        a.set_xlabel("Components")

    return f
