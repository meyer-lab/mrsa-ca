"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Move all graphing elements out.
    Figure out nice loop for graphing.
    Relearn PCA and SVD to confirm I know what I'm graphing
        and why. Also, get confirmation about what I'm
        hoping to show (differences diseases across genes?).
    Add r2x analysis to double check performance of PCA.
        Can simply use explained_variance_, looping to higher component #s
"""

from sklearn.decomposition import PCA
from import_data import form_matrix, import_mrsa_rna, import_ca_rna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def perform_PCA(components=8):

    rna_mat = form_matrix()

    pca = PCA(n_components=components)

    rna_decomp = pca.fit_transform(rna_mat)

    # print(f"shape of loadings: {pca.components_.shape}, (components, gene)")
    # print(f"Shape of scores: {rna_decomp.shape} (patients, components)")
    print(rna_decomp)

    return rna_decomp

"""
Move all plotting and figure generation to another file after making sure
I'm on the right track with these.
"""
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


    

    


    

# perform_PCA(2)
plot_pca()
