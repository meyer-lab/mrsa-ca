"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:
    Make a nice loop for graphing all the different components.
    Refactor using new filesystem setup (where to outout figures,
        what files to import, what arguments the functions take, etc.)
    Color scatters based on 'mrsa' and 'ca'
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from mrsa_ca_rna.import_data import import_mrsa_rna, import_ca_rna, form_matrix
from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns
import matplotlib.pyplot as plt


def figure01_setup():
    
    mrsa_rna = import_mrsa_rna()
    mrsa_label = np.full(len(mrsa_rna.index), "mrsa").T

    ca_rna = import_ca_rna()
    ca_label = np.full(len(ca_rna.index), "ca").T

    state = np.concatenate((mrsa_label, ca_label))

    rna_mat = form_matrix()
    rna_decomp, pca = perform_PCA(rna_mat)
    
    columns = []
    for i in range(pca.n_components_):
        columns.append("PC" + str(i+1))
    
    rna_decomp.insert(loc=0, column="state", value=state)

    return rna_decomp

    # use as reference for what I want the loop to accomplish.
    # fig01 = plt.figure()
    # plt.scatter(rna_decomp.loc["mrsa","PC1"], rna_decomp.loc["mrsa", "PC2"], color="red")
    # plt.scatter(rna_decomp.loc["ca","PC1"], rna_decomp.loc["ca", "PC2"], color="blue")
    # fig01.savefig("fig01")

    # fig02 = plt.figure()
    # plt.scatter(rna_decomp.loc["mrsa","PC3"], rna_decomp.loc["mrsa", "PC4"], color="red")
    # plt.scatter(rna_decomp.loc["ca","PC3"], rna_decomp.loc["ca", "PC4"], color="blue")
    # fig02.savefig("fig02")

    # fig03 = plt.figure()
    # plt.scatter(rna_decomp.loc["mrsa","PC5"], rna_decomp.loc["mrsa", "PC6"], color="red")
    # plt.scatter(rna_decomp.loc["ca","PC5"], rna_decomp.loc["ca", "PC6"], color="blue")
    # fig03.savefig("fig03")

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

    

    a = sns.scatterplot(data=data.loc[:,"PC2":"PC3"], x="PC2", y="PC3", hue=data.loc[:,"state"], ax=ax[0])
    a.set_xlabel("PC2")
    a.set_ylabel("PC3")
    a.set_title("Var Comp 1")

    a = sns.scatterplot(data=data.loc[:,("PC2","PC4")], x="PC2", y="PC4", hue=data.loc[:,"state"], ax=ax[1])
    a.set_xlabel("PC2")
    a.set_ylabel("PC4")
    a.set_title("Var Comp 2")

    a = sns.scatterplot(data=data.loc[:,("PC2","PC5")], x="PC2", y="PC5", hue=data.loc[:,"state"], ax=ax[2])
    a.set_xlabel("PC2")
    a.set_ylabel("PC5")
    a.set_title("Var Comp 3")

    a = sns.scatterplot(data=data.loc[:,("PC2","PC6")], x="PC2", y="PC6", hue=data.loc[:,"state"], ax=ax[3])
    a.set_xlabel("PC2")
    a.set_ylabel("PC6")
    a.set_title("Var Comp 4")

    return f

# debug
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig01_PC2Comp.png")