"""Thie file will plot the pf2 b factor matrix for different subsets
of the disease datasets to identify heavy eigen-1 loading issues."""

import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import concat_datasets


def subset_data(by: str = "disease") -> dict[str, np.ndarray]:
    """Get all the data and subset it for the factorization sets."""

    X = concat_datasets()

    data_list = {}
    if by == "disease":
        # Subset by disease
        diseases = X.obs["disease"].unique()
        for disease in diseases:
            subset = X[X.obs["disease"] != disease].copy()
            subset.X = StandardScaler().fit_transform(subset.X)
            subset, _ = perform_parafac2(
                subset,
                slice_col="disease",
                rank=10,
            )
            data_list[disease] = subset.uns["Pf2_B"]

    elif by == "rank":
        # Subset by rank
        ranks = [5, 10, 15, 20]
        for rank in ranks:
            subset = X.copy()
            subset, _ = perform_parafac2(
                subset,
                slice_col="disease",
                rank=rank,
            )
            data_list[str(rank)] = subset.uns["Pf2_B"]

    elif by == "size":
        sizes = [0.25, 0.5, 0.75, 1.0]
        for size in sizes:
            rng = np.random.default_rng(42)
            sub_ind = rng.choice(X.shape[0], int(X.shape[0] * size), replace=False)
            subset = X[sub_ind].copy()
            subset.X = StandardScaler().fit_transform(subset.X)
            subset, _ = perform_parafac2(
                subset,
                slice_col="disease",
                rank=10,
            )
            data_list[str(size)] = subset.uns["Pf2_B"]

    return data_list


def genFig():

    sub_type = "rank"
    data_list = subset_data(by=sub_type)
    layout, fig_size = calculate_layout(num_plots=len(data_list))
    ax, f, _ = setupBase(fig_size, layout)

    for i, (key, X) in enumerate(data_list.items()):
        # Plot the factor matrix for each subset
        ranks_labels = [f"{j + 1}" for j in range(X.shape[1])]

        b = sns.heatmap(
            X,
            ax=ax[i],
            cmap="coolwarm",
            center=0,
            xticklabels=ranks_labels,
            yticklabels=ranks_labels,
        )
        b.set_title(f"Eigenstate Factor Matrix Case: {key}")
        b.set_xlabel("Rank")
        b.set_ylabel("Eigenstate")

        f.suptitle(f"{sub_type} Dependence: Normalized Factors", fontsize=16)

    return f
