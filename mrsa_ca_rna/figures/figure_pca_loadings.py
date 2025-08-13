"""
This file is the start of analyzing the loadings of the
mrsa+ca+healthy data, based on previous scores analysis.
This file may become obsolete post scores heatmap analysis currently
planned.

"""

import numpy as np
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.utils import prepare_data, prepare_mrsa_ca


def figure_setup():
    adata = prepare_data(filter_threshold=-1)
    _, _, adata = prepare_mrsa_ca(adata)

    # Convert the AnnData object to a DataFrame for PCA
    df = adata.to_df()

    _, loadings, _ = perform_pca(df, components=10)

    return loadings.T


def genFig():
    fig_size = (8, 12)
    layout = {"ncols": 2, "nrows": 3}
    ax, f, _ = setupBase(fig_size, layout)

    data = figure_setup()

    # plot a heatmap of the loadings
    a = sns.heatmap(data, cmap="vlag", ax=ax[0])
    a.set_title("PCA Loadings")
    a.set_xlabel("Principal Components")
    a.set_ylabel("Genes")
    a.set_yticklabels([])

    # add hue labels to the top and bottom 3 genes in each of the first 5 components
    for i in range(5):
        # subset the data to the current component
        component_data = data.iloc[:, i]

        # get the top and bottom 3 genes locations
        top_genes = component_data.nlargest(3).index
        bottom_genes = component_data.nsmallest(3).index
        top_genes_text = "\n-".join(top_genes)
        bottom_genes_text = "\n-".join(bottom_genes)

        # add a status column to the data
        component_data = component_data.to_frame()
        component_data["status"] = "mids"
        component_data.loc[top_genes, "status"] = "tops"
        component_data.loc[bottom_genes, "status"] = "bottoms"

        # plot the stripplot
        b = sns.stripplot(
            x=[i] * len(component_data),
            y=component_data.iloc[:, 0],
            hue=component_data["status"],
            hue_order=["tops", "mids", "bottoms"],
            palette=sns.color_palette("icefire", 3),
            ax=ax[i + 1],
            dodge=False,
        )
        handles, labels = b.get_legend_handles_labels()
        labels = [
            f"Bottoms:\n-{bottom_genes_text}"
            if label == "bottoms"
            else f"Tops:\n-{top_genes_text}"
            if label == "tops"
            else "Mids"
            for label in labels
        ]
        b.legend(handles, labels)

        # get the x positions of the points
        offsets = b.collections[0].get_offsets()
        x_positions = np.array(offsets)[:, 0]

        # label the top and bottom 3 genes
        for gene in top_genes:
            gene_index: int = component_data.index.get_loc(gene)
            b.text(
                x_positions[gene_index],
                component_data.iloc[gene_index, 0],
                gene,
                horizontalalignment="right",
                size="small",
                color="black",
                weight="semibold",
            )
        for gene in bottom_genes:
            gene_index = component_data.index.get_loc(gene)
            b.text(
                x_positions[gene_index],
                component_data.iloc[gene_index, 0],
                gene,
                horizontalalignment="right",
                size="small",
                color="black",
                weight="semibold",
            )

    return f
