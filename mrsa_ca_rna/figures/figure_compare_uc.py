"""This file will compare the expression data of the three sources for UC: h5, EXP, and GEO."""

import numpy as np
import seaborn as sns
import pandas as pd

from mrsa_ca_rna.import_data import import_uc, import_uc_exp, import_uc_geo
from mrsa_ca_rna.utils import calculate_cpm
from mrsa_ca_rna.figures.base import setupBase, calculate_layout


def get_data():

    # Bring in the three datasets for UC
    uc_h5_adata = import_uc()
    uc_exp_adata = import_uc_exp()
    uc_geo_adata = import_uc_geo()

    # Rename disease in h5 to match the others
    uc_h5_adata.obs["disease"] = "UC_H5"

    # CPM normalize each dataset
    uc_h5_adata.layers["cpm"] = calculate_cpm(uc_h5_adata.X)
    uc_exp_adata.layers["cpm"] = calculate_cpm(uc_exp_adata.X)
    uc_geo_adata.layers["cpm"] = calculate_cpm(uc_geo_adata.X)

    return uc_h5_adata, uc_exp_adata, uc_geo_adata

def genFig():
    """Generate the figure comparing the three datasets for UC."""

    # Get our three datasets to compare
    uc_h5, uc_exp, uc_geo = get_data()

    # Randomly select 20 genes shared between the three datasets
    rng = np.random.default_rng(420)
    common_genes = set(uc_h5.var_names).intersection(set(uc_exp.var_names)).intersection(set(uc_geo.var_names))
    selected_genes = common_genes if len(common_genes) <= 20 else set(
        rng.choice(list(common_genes), size=20, replace=False)
    )

    # Plot the expression of these genes in all three datasets
    num_plots = len(selected_genes)
    layout, fig_size = calculate_layout(num_plots, 4)
    ax1, f, _ = setupBase(fig_size, layout)
    ax2, g, _ = setupBase(fig_size, layout)

    # Plot each gene's expression distribution in all datasets
    for i, gene in enumerate(selected_genes):
        # Get data for this gene from all datasets
        uc_h5_data = uc_h5[:, gene].layers["cpm"].flatten()
        uc_exp_data = uc_exp[:, gene].layers["cpm"].flatten()
        uc_geo_data = uc_geo[:, gene].layers["cpm"].flatten()

        # Combine into a single dataframe for seaborn
        combined_data = pd.DataFrame({
            'CPM': np.concatenate([uc_h5_data, uc_exp_data, uc_geo_data]),
            'Dataset': ['UC_H5'] * len(uc_h5_data) + ['UC_EXP'] * len(uc_exp_data) + ['UC_GEO'] * len(uc_geo_data)
        })

        # Create a boxplot for this gene
        a = sns.boxplot(x='Dataset', y='CPM', data=combined_data, ax=ax1[i])
        a.set_title(gene)
        a.set_xlabel("Dataset")
        a.set_ylabel("Expression (CPM)")

        # Create a histogram for this gene
        b = sns.histplot(data=combined_data,
                         x='CPM',
                         hue='Dataset',
                         ax=ax2[i],
                         bins=20,
                         palette="Set1",
                         edgecolor='black')
        b.set_title(gene)
        b.set_xlabel("Expression (CPM)")
        b.set_ylabel("Frequency")

    return f, g
