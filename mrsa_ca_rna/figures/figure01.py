"""
This is a copy of figure01_pca.py that was generated when I was trying
to figure out what had happened to the original (see commit message before
the commit accompanying this comment section)

This is to be deleted once I can ensure the previous file generates plots correctly
"""

import numpy as np
import pandas as pd

from mrsa_ca_rna.import_data import import_mrsa_rna, import_ca_rna, form_matrix
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns
import matplotlib.pyplot as plt


def figure01_setup():
    
    mrsa_rna = import_mrsa_rna()
    mrsa_label = np.full(len(mrsa_rna.index), "mrsa").T

    ca_pos_rna, ca_neg_rna = import_ca_rna()
    ca_label = np.full(len(ca_pos_rna.index), "ca").T
    healthy_label = np.full(len(ca_neg_rna.index), "healthy").T

    state = np.concatenate((mrsa_label, ca_label, healthy_label))

    rna_mat = form_matrix()
    rna_decomp, pca = perform_PCA(rna_mat)
    
    columns = []
    for i in range(pca.n_components_):
        columns.append("PC" + str(i+1))
    
    rna_decomp.insert(loc=0, column="state", value=state)

    return rna_decomp


def genFig():

    fig_size = (6, 6)
    layout = {
        "ncols": 2,
        "nrows": 2,
    }
    ax, f, _ = setupBase(
        fig_size,
        layout
    )

    data = figure01_setup()

    # modify what components you want to compare to one another:
    component_pairs = np.array([[1,2],
                                [3,4],
                                [5,6],
                                [7,8]],
                                dtype=int)
    
    assert component_pairs.shape[0] == layout["ncols"]*layout["nrows"], "component pairs to be graphed do not match figure layout size"

    for i, (j, k) in enumerate(component_pairs):
        a = sns.scatterplot(data=data.loc[:,(data.columns[j],data.columns[k])], x=data.columns[j], y=data.columns[k], hue=data.loc[:,"state"], ax=ax[i])
        a.set_xlabel(data.columns[j])
        a.set_ylabel(data.columns[k])
        a.set_title(f"Var Comp {data.columns[j]} vs {data.columns[k]}")


    return f

"""Debug function call section"""
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig01_Healthy+.png")