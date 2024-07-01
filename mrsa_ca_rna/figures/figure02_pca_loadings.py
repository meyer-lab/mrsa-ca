"""
This file is the start of analyzing the loadings of the
mrsa+ca+healthy data, based on previous scores analysis.
This file may become obsolete post scores heatmap analysis currently
planned.

To-do:
    change file to make loadings DataFrame
"""


import numpy as np
import pandas as pd

from mrsa_ca_rna.import_data import form_matrix
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns
import matplotlib.pyplot as plt


def figure01_setup():
    
    """code-block is useless for loadings because you cannot use disease state
    in a feature-based context. The features (genes) are not unique but shared
    across disease types. We can only determine their significance to PC's, which
    we then relate to disease type."""
    # # import mrsa_rna data to generate mrsa labeling, using columns for features
    # mrsa_rna = import_mrsa_rna()
    # mrsa_label = np.full(len(mrsa_rna.columns), "mrsa").T

    # # import mrsa_rna data to generate ca labeling, using columns for features
    # ca_pos_rna, ca_neg_rna = import_ca_rna()
    # ca_label = np.full(len(ca_pos_rna.columns), "ca").T

    # # use previous ca_neg import and any other healthy import to make healthy label
    # gse_rna = import_GSE_rna()
    # healthy_ca = np.full(len(ca_neg_rna.columns), "healthy_ca").T
    # healthy_gse = np.full(len(gse_rna.columns), "healthy_gse").T

    # # make state column for graphing across disease type and healthy
    # state = np.concatenate((mrsa_label, ca_label, healthy_ca, healthy_gse))

    rna_mat = form_matrix()
    rna_decomp, pca = perform_PCA(rna_mat)
    
    rows = []
    for i in range(pca.n_components_):
        rows.append("PC" + str(i+1))
    
    loadings = pd.DataFrame(pca.components_, index=rows, columns=rna_mat.columns)
    # loadings.insert(loc=0, column="state", value=state)

    return loadings


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

    # reduce all readable component pair values by 1 since indexes start at 0
    component_pairs -= 1
    # removed hue=data.loc[:, "state"] as it cannot be used in a feature-based context
    for i, (j, k) in enumerate(component_pairs):
        a = sns.scatterplot(data=data.loc[(data.index[j],data.index[k]),:].T, x=data.index[j], y=data.index[k], ax=ax[i])
        a.set_xlabel(data.index[j])
        a.set_ylabel(data.index[k])
        a.set_title(f"Feature Var by {data.index[j]} vs {data.index[k]}")


    return f

"""Debug function call section"""
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig02_LoadingsExp.png")