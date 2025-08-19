"""This file will plot embeddings of the pf2 factorization for the disease datasets"""

import anndata as ad
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import prepare_data


def get_data() -> tuple[ad.AnnData, float]:
    """Concatenate the data and perform the factorization"""
    X = prepare_data()
    X, r2x = perform_parafac2(X)
    return X, r2x


def genFig():
    """Generate the figure with PaCMAP and UMAP embeddings"""
    X, r2x = get_data()

    layout = {"ncols": 2, "nrows": 1}
    fig_size = (10, 4)
    ax, f, _ = setupBase(fig_size, layout)

    # Explicitly cast the data to avoid spmatrix issues
    pacmap_coords: np.ndarray = np.asarray(X.obsm["Pf2_PaCMAP"])
    projections: np.ndarray = np.asarray(X.obsm["Pf2_projections"])

    a = sns.scatterplot(
        x=pacmap_coords[:, 0],
        y=pacmap_coords[:, 1],
        hue=X.obs["disease"],
        ax=ax[0],
        palette="tab20",
        s=5,
    )
    a.set_title(
        f"PaCMAP Embedding of Disease Projections {X.varm['Pf2_C'].shape[1]} components"
    )
    a.set_xlabel("PaCMAP 1")
    a.set_ylabel("PaCMAP 2")
    a.legend(markerscale=2)

    # Identify the strongest eigen state (row with largest sum across columns)
    strongest_eigenstate = np.sum(np.abs(np.asarray(X.uns["Pf2_B"])), axis=1).argmax()

    # Make a centered normalized value for the hue of the second plot
    vmin = float(projections[:, strongest_eigenstate].min())
    vmax = float(projections[:, strongest_eigenstate].max())
    abs_max = max(abs(vmin), abs(vmax))
    norm = Normalize(-abs_max, abs_max)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm, ax=ax[1], label=f"Eigen-{strongest_eigenstate + 1} Value"
    ).set_label(f"Eigen-{strongest_eigenstate + 1} Value", rotation=270)

    b = sns.scatterplot(
        x=pacmap_coords[:, 0],
        y=pacmap_coords[:, 1],
        hue=projections[:, strongest_eigenstate],
        ax=ax[1],
        hue_norm=norm,
        palette="coolwarm",
        s=5,
    )
    b.set_title(f"PaCMAP Embedding\norganized by Eigen-{strongest_eigenstate + 1} Value")
    b.set_xlabel("PaCMAP 1")
    b.set_ylabel("PaCMAP 2")
    b.legend(markerscale=2)

    return f
