"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:
    Make a nice loop for graphing all the different components.
    Implement seaborn.
    Refactor using new filesystem setup (where to outout figures,
        what files to import, what arguments the functions take, etc.)
    Change everything to be Fig_Setup -> MakeFig (subplots) like in
        tfac-mrsa code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mrsa_ca_rna.import_data import import_mrsa_rna, import_ca_rna
from mrsa_ca_rna.pca import perform_PCA


def plot_pca():
    n_components = 6
    
    mrsa_rna = import_mrsa_rna()
    new_mrsa = np.full((len(mrsa_rna.index),), "mrsa")
    # mrsa_rna.set_index(new_mrsa, inplace=True)

    ca_rna = import_ca_rna()
    new_ca = np.full((len(ca_rna.index),), "ca")
    # ca_rna.set_index(new_ca, inplace=True)

    indeces = np.concatenate((new_mrsa, new_ca))
    columns = []
    for i in range(n_components):
        columns.append("PC" + str(i+1))


    rna_decomp = pd.DataFrame(perform_PCA(n_components), indeces, columns)
    

    """
    figuring out a nicer loop...
    """
    disease = ["mrsa", "ca"]
    colors = ["red", "blue"]
    fig = plt.figure()
    for i, color in zip(disease, colors):
        plt.scatter(rna_decomp.loc[i, "PC1"], rna_decomp.loc[i, "PC2"], color=color, label=i)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of MRSA and CA RNAseq")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    fig.savefig("loopTest")

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