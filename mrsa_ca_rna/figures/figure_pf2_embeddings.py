"""This file will plot embeddings of the pf2 factorization for the disease datasets"""

import anndata as ad
import seaborn as sns
from matplotlib import pyplot as plt

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import concat_datasets


def get_data() -> tuple[ad.AnnData, float]:
    """Concatenate the data and perform the factorization"""
    X = concat_datasets()
    X, r2x = perform_parafac2(
        X,
        slice_col="disease",
        rank=5,
    )
    return X, r2x


def genFig():
    """Generate the figure with PaCMAP and UMAP embeddings"""
    X, r2x = get_data()

    layout, fig_size = calculate_layout(num_plots=2, scale_factor=4)
    ax, f, _ = setupBase(fig_size, layout)

    a = sns.scatterplot(
        x=X.obsm["Pf2_PaCMAP"][:, 0],
        y=X.obsm["Pf2_PaCMAP"][:, 1],
        hue=X.obs["disease"],
        ax=ax[0],
        palette="tab20",
        s=10,
    )
    a.set_title("PaCMAP Embedding organized by Disease")
    a.set_xlabel("PaCMAP 1")
    a.set_ylabel("PaCMAP 2")

    # Make a centered normalized value for the hue
    vmin = X.obsm["Pf2_projections"][:, 0].min()
    vmax = X.obsm["Pf2_projections"][:, 0].max()
    abs_max = max(abs(vmin), abs(vmax))
    norm = plt.Normalize(-abs_max, abs_max)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax[1], label="Eigen-1 Value")

    b = sns.scatterplot(
        x=X.obsm["Pf2_PaCMAP"][:, 0],
        y=X.obsm["Pf2_PaCMAP"][:, 1],
        hue=X.obsm["Pf2_projections"][:, 0],
        ax=ax[1],
        hue_norm=norm,
        palette="coolwarm",
        s=10,
    )
    b.set_title("PaCMAP Embedding organized by Eigen-1 Value")
    b.set_xlabel("PaCMAP 1")
    b.set_ylabel("PaCMAP 2")

    f.suptitle(
        f"Pf2 Factorization with R2X: {r2x:.3f}\nEmbeddings of Disease Projections",
        fontsize=16,
    )

    return f
