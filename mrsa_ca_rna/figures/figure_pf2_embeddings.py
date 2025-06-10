"""This file will plot embeddings of the pf2 factorization for the disease datasets"""

import anndata as ad

import seaborn as sns
import umap
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
    )
    a.set_title("PaCMAP Embedding of Disease Projections")
    a.set_xlabel("PaCMAP 1")
    a.set_ylabel("PaCMAP 2")

    # Make a normalized UMAP value for the hue
    norm = plt.Normalize(X.obsm["Pf2_projections"][:, 0].min(), X.obsm["Pf2_projections"][:, 0].max())
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
    )
    b.set_title("PaCMAP Embedding of Disease Projections")
    b.set_xlabel("PaCMAP 1")
    b.set_ylabel("PaCMAP 2")

    f.suptitle(
        f"Pf2 Factorization with R2X: {r2x:.3f}\n"
        "Embeddings of Disease Projections",
        fontsize=16
        )

    return f
