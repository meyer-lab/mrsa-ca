"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:
    Make a function that takes in PCA component data and automatically determines the
        appropriate size and layout of the graph. Can remove assert once
        completed.
    Regenerate plots using recently added gse_healthy data
"""

import numpy as np
import pandas as pd

from mrsa_ca_rna.import_data import import_mrsa_rna, import_ca_rna, import_GSE_rna, form_matrix
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns
import matplotlib.pyplot as plt


def figure01_setup():
    
    # import mrsa_rna data to generate mrsa labeling
    mrsa_rna = import_mrsa_rna()
    mrsa_label = np.full(len(mrsa_rna.index), "mrsa").T

    # import mrsa_rna data to generate ca labeling
    ca_pos_rna, ca_neg_rna = import_ca_rna()
    ca_label = np.full(len(ca_pos_rna.index), "ca").T

    # use previous ca_neg import and any other healthy import to make healthy label
    gse_rna = import_GSE_rna()
    healthy_ca = np.full(len(ca_neg_rna.index), "healthy_ca").T
    healthy_gse = np.full(len(gse_rna.index), "healthy_gse").T

    # make state column for graphing across disease type and healthy
    state = np.concatenate((mrsa_label, ca_label, healthy_ca, healthy_gse))

    rna_mat = form_matrix()
    rna_decomp, pca = perform_PCA(rna_mat)
    
    columns = []
    for i in range(pca.n_components_):
        columns.append("PC" + str(i+1))
    
    rna_decomp.insert(loc=0, column="state", value=state)

    return rna_decomp


def genFig():

    fig_size = (12, 9)
    layout = {
        "ncols": 4,
        "nrows": 3,
    }
    ax, f, _ = setupBase(
        fig_size,
        layout
    )

    data = figure01_setup()

    # modify what components you want to compare to one another:
    component_pairs = np.array([[1,2],
                                [1,3],
                                [2,3],
                                [2,4],
                                [3,4],
                                [3,5],
                                [4,5],
                                [4,6],
                                [5,6],
                                [5,7],
                                [6,7],
                                [7,8]],
                                dtype=int)
    
    assert component_pairs.shape[0] == layout["ncols"]*layout["nrows"], "component pairs to be graphed do not match figure layout size"

    # coincidentally, columns in DataFrames start at 1
    for i, (j, k) in enumerate(component_pairs):
        a = sns.scatterplot(data=data.loc[:,(data.columns[j],data.columns[k])], x=data.columns[j], y=data.columns[k], hue=data.loc[:,"state"], ax=ax[i])
        a.set_xlabel(data.columns[j])
        a.set_ylabel(data.columns[k])
        a.set_title(f"Var Comp {data.columns[j]} vs {data.columns[k]}")


    return f

"""Debug function call section"""
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig01_Large_Sequential_compare.png")