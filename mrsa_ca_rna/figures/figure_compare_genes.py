"""This file will investigate the ARCHS4 data by comparing it to the MRSA data we have
on-hand. The goal is to see if bimodal expression patterns are present in both"""

import numpy as np
import seaborn as sns
import pandas as pd

from mrsa_ca_rna.utils import map_genes, concat_datasets, calculate_cpm
from mrsa_ca_rna.import_data import import_mrsa_tfac
from mrsa_ca_rna.figures.base import setupBase, calculate_layout

def get_data():

    # Import MRSA archs4 data
    X = concat_datasets(filter_threshold=-1)
    X = X[X.obs["disease"] == "MRSA", :]
    
    X.layers["cpm"] = calculate_cpm(X.layers["raw"])

    # Import MRSA TFAC data
    tfac_mrsa = import_mrsa_tfac()
    tfac_mrsa.layers["cpm"] = calculate_cpm(tfac_mrsa.X)

    # Map tfac genes from ensembl to symbol
    mapping = map_genes(
        tfac_mrsa.var_names.to_list(),
        gtf_path="mrsa_ca_rna/data/gencode.v48.comprehensive.annotation.gtf.gz",
        from_type="ensembl",
        to_type="symbol",
    )

    tfac_mrsa.var_names = tfac_mrsa.var_names.map(mapping)

    return X, tfac_mrsa

def genFig():

    # Get our two datasets to compare
    archs4, tfac = get_data()

    # Randomly select 20 genes shared between the two datasets
    rng = np.random.default_rng(420)
    common_genes = set(archs4.var_names).intersection(set(tfac.var_names))
    selected_genes = common_genes if len(common_genes) <= 20 else set(
        rng.choice(list(common_genes), size=20, replace=False)
    )

    # Plot the expression of these genes in both datasets
    num_plots = len(selected_genes)
    layout, fig_size = calculate_layout(num_plots, 4)
    ax, f, _ = setupBase(fig_size, layout)

    # Plot each gene's expression distribution in both datasets
    # ARCHS4 in blue, TFAC in red
    for i, gene in enumerate(selected_genes):
        # Get data for this gene from both datasets
        archs4_data = archs4[:, gene].layers["cpm"].flatten()
        tfac_data = tfac[:, gene].layers["cpm"].flatten()
        
        # Combine into a single dataframe for seaborn
        combined_data = pd.DataFrame({
            'CPM': np.concatenate([archs4_data, tfac_data]),
            'Dataset': ['ARCHS4'] * len(archs4_data) + ['TFAC'] * len(tfac_data)
        })
        
        # Plot with hue parameter
        a = sns.histplot(
            data=combined_data,
            x='CPM',
            hue='Dataset',
            bins=50,
            alpha=0.6,
            ax=ax[i],
            palette={'ARCHS4': 'blue', 'TFAC': 'red'}
        )
        
        a.set_title(f"{gene} Distribution")
        a.set_xlabel("Counts (CPM)")
        a.set_ylabel("Frequency")

    f.suptitle("Gene Expression Comparison: ARCHS4 vs TFAC MRSA", fontsize=16)

    return f