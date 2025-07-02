"""This file will plot the projections of the PF2 model
to observe patient distributions in the latent space.
We suspect there might be a few outliers in the latent space."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    X = concat_datasets()

    # # Remove Breast Cancer TCR data and reZ the data
    # X = X[X.obs["disease"] != "Breast Cancer TCR"].copy()
    # from sklearn.preprocessing import StandardScaler
    # X.X = StandardScaler().fit_transform(X.X)

    rank = 1

    X, _ = perform_parafac2(X, slice_col="disease", rank=rank)

    # Make a weighted projection DataFrame for easier plotting and disease labeling
    p_df = pd.DataFrame(
        X.obsm["Pf2_projections"],
        index=X.obs.index,
        columns=pd.Index([x for x in range(1, X.obsm["Pf2_projections"].shape[1] + 1)]),
    )
    p_df["disease"] = X.obs["disease"].values

    return p_df


def genFig():
    """Generate the figure for the projections of the PF2 model"""

    # Get the weighted projections
    projections = figure_setup()

    # Setup the projections figure
    layout = {"ncols": 4, "nrows": 4}
    fig_size = (16, 16)
    ax, f, _ = setupBase(fig_size, layout)

    # Find the absolute maximum value across all projections
    max_abs_value = projections.drop(columns=["disease"]).abs().max().max()

    # Normalize the projections by the maximum absolute value
    projections.iloc[:, :-1] /= max_abs_value

    # For each disease, plot the projections
    for i, disease in enumerate(projections["disease"].unique()):
        # Subset the DataFrame for the current disease
        projection: pd.DataFrame = projections.loc[
            projections["disease"] == disease
        ].drop(columns=["disease"])

        # We normalized the projections so we could directly compare them
        # across diseases, so we can use the same vmax and vmin for all heatmaps
        a = sns.heatmap(
            projection,
            ax=ax[i],
            cmap="coolwarm",
            vmax=1,
            vmin=-1,
            center=0,
            cbar=True,
            xticklabels=projection.columns.to_list(),
            yticklabels=False,
        )
        a.set_title(f"Weighted Projections for {disease}")
        a.set_xlabel("Eigenstate")

    # Setup a new figure for the overall distribution of samples
    layout = {"ncols": 1, "nrows": 1}
    fig_size = (4, 4)
    ax, g, _ = setupBase(fig_size, layout)

    # Calculate the percentiles for the first projection column
    column_name = projections.columns[0]
    p50 = projections[column_name].quantile(0.5)
    p75 = projections[column_name].quantile(0.75)
    p95 = projections[column_name].quantile(0.95)

    # Plot a stripplot of the samples across all diseases
    b = sns.stripplot(
        data=projections,
        x="disease",
        y=column_name,
        ax=ax[0],
        jitter=True,
        alpha=0.5,
        color="black",
    )

    # Add horizontal lines for the percentiles
    b.axhline(y=p50, color="red", linestyle="--", alpha=0.7, label="50th percentile")
    b.axhline(y=p75, color="orange", linestyle="--", alpha=0.7, label="75th percentile")
    b.axhline(y=p95, color="green", linestyle="--", alpha=0.7, label="95th percentile")

    # Add a legend
    b.legend(loc="best", frameon=True, framealpha=0.7)

    # Add sample counts to disease labels and rotate them for better visibility
    disease_counts = projections["disease"].value_counts().to_dict()
    new_labels = [
        f"{disease} (n={disease_counts[disease]})"
        for disease in projections["disease"].unique()
    ]
    plt.setp(b.get_xticklabels(), labels=new_labels, rotation=45, ha="right")

    b.set_title("Distribution of Samples Across Diseases")
    b.set_xlabel("Disease")
    b.set_ylabel("Projection Value")

    return f, g
