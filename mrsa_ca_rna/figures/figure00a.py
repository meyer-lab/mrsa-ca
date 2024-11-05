# main module imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# secondary module imports
from kneed import KneeLocator
from matplotlib.lines import Line2D

# local module imports
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca


def setup_figure00a():
    """
    Set up the data for the PCA explained variance figure.
    Goal is to explore the variance within each data set.
    """

    # data import, concatenation, scaling, and preparation
    concat_diseases = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    data_labels = concat_diseases.obs["disease"].unique()
    dataset_list = [
        concat_diseases[concat_diseases.obs["disease"] == label]
        for label in data_labels
    ]

    for i, dataset in enumerate(dataset_list):
        _, _, pca = perform_pca(dataset.to_df(), scale=False)

        components = np.arange(1, pca.n_components_ + 1, dtype=int)
        svd_val = pca.singular_values_
        total_explained = np.cumsum(pca.explained_variance_ratio_)

        data = pd.DataFrame(components, columns=pd.Index(["components"]))
        data["singular values"] = svd_val
        data["total explained variance"] = total_explained

        dataset_list[i] = data

    variance_data = dict(zip(data_labels, dataset_list, strict=False))

    return variance_data


def genFig():
    """
    Generate the PCA explained variance figure.
    """

    fig_size = (9, 6)
    layout = {"ncols": 3, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    datasets = setup_figure00a()

    # plot the per component explained variance
    for i, keys in enumerate(datasets):
        cmap = plt.get_cmap("viridis")

        kneedle = KneeLocator(
            datasets[keys]["components"],
            datasets[keys]["singular values"],
            S=1.0,
            curve="convex",
            direction="decreasing",
        )

        a = sns.lineplot(
            data=datasets[keys],
            x="components",
            y="singular values",
            ax=ax[i],
            label="Singular values",
        )
        # Add the axvline without adding it to the legend
        a.axvline(kneedle.knee, color="red", linestyle="--")
        a.set_xlabel("# of Components")
        a.set_ylabel("Singular values")
        a.set_title(f"PCA performance of {keys} dataset")

        ax2 = a.twinx()
        sns.lineplot(
            data=datasets[keys],
            x="components",
            y="total explained variance",
            ax=ax2,
            label="total explained variance",
            color=cmap(0.5),
            legend=False,
        )

        # set the y-axis ticks to be approx same spacing
        a.set_yticks(np.linspace(*a.get_ybound(), 5).round(2))
        ax2.set_yticks(np.linspace(0, 1.0, 5).round(2))
        a.set_xticks(np.arange(0, 51, 10))

        # Get handles and labels from the primary y-axis
        handles, labels = a.get_legend_handles_labels()

        # Get handles and labels from the secondary y-axis
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Create a custom handle for the axvline
        knee_legend = Line2D(
            [0], [0], color="red", linestyle="--", label=f"Knee Point ({kneedle.knee})"
        )

        # Combine all handles and labels
        handles = handles + handles2 + [knee_legend]
        labels = labels + labels2 + [f"Knee Point ({kneedle.knee})"]

        # Update the legend with all handles and labels
        a.legend(handles=handles, labels=labels, loc="best")

    return f
