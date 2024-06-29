"""
Generate a figure depicting the explained variance of the PCA
analysis. We're looking to see diminishing returns after several
components.

To-do:
    Cleanup after writing more of these figure files
    Regenerate plots using recently added gse_healthy data
"""

import matplotlib.pyplot as plt
from mrsa_ca_rna.import_data import form_matrix
import numpy as np
import pandas as pd
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns


from mrsa_ca_rna.pca import perform_PCA

def figure_00_setup():
    """ Make and organize the data to be used in genFig"""
    rna_mat = form_matrix()
    rna_decomp, pca = perform_PCA(rna_mat)

    components = range(1, pca.n_components_+1)
    total_explained = np.cumsum(pca.explained_variance_ratio_)

    data = np.stack([components, total_explained]).T
    data = pd.DataFrame(data, columns=["components", "explained"])
    data["components"] = data["components"].astype("int32")

    return data

    # fig00 = plt.figure()
    # plt.plot(components, explained)
    # plt.title("'Completeness' of PCA Decomposition" )
    # plt.xlabel("# of Principle Components")
    # plt.ylabel("Fraction of variance explained")
    # fig00.savefig("./output/fig00")

def genFig():
    """
    Start making the figure.
    """
    fig_size = (3,3)
    layout = {
        "ncols": 1,
        "nrows": 1
    }
    ax, f, _ = setupBase(
        fig_size,
        layout
    )

    data = figure_00_setup()
    a = sns.lineplot(data=data, x="components", y="explained", ax=ax[0])
    a.set_xlabel("# of Components")
    a.set_ylabel("Fraction of explained variance")
    a.set_title("PCA performance")

    return f

# #debug
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig00_Healthy+.png")