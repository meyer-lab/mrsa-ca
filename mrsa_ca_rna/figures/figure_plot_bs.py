"""Thie file will plot the pf2 b factor matrix for different subsets
of the disease datasets to identify heavy eigen-1 loading issues."""

import anndata as ad
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import concat_datasets


def subset_data(by: str = "disease") -> list[ad.AnnData]:
    """Get all the data and subset it for the factorization sets."""

    X = concat_datasets()

    data_list = []
    if by == "disease":
        # Subset by disease
        diseases = X.obs["disease"].unique()
        for disease in diseases:
            subset = X[X.obs["disease"] == disease].copy()
            subset, _ = perform_parafac2(
                subset,
                slice_col="disease",
                rank=10,
            )
            data_list.append(subset)
    elif by == "rank":
        # Subset by rank
        ranks = [5, 10, 15, 20]
        for rank in ranks:
            for rank in ranks:
                subset = X.copy()
                subset, _ = perform_parafac2(
                    subset,
                    slice_col="disease",
                    rank=rank,
                )
                data_list.append(subset)

    return data_list


def genFig():
    data_list = subset_data(by="rank")
    layout, fig_size = calculate_layout(num_plots=len(data_list))
    ax, f, _ = setupBase(fig_size, layout)

    for i, X in enumerate(data_list):
        # Plot the factor matrix for each subset
        ranks_labels = [f"Rank {j + 1}" for j in range(X.uns["Pf2_B"].shape[1])]

        b = sns.heatmap(
            X.uns["Pf2_B"],
            ax=ax[i],
            cmap="coolwarm",
            center=0,
            xticklabels=ranks_labels,
            yticklabels=ranks_labels,
        )
        b.set_title("Eigenstate Factor Matrix")
        b.set_xlabel("Rank")
        b.set_ylabel("Eigenstate")

        return f
