"""
Generate a figure depicting the explained variance of the PCA
analysis. We're looking to see diminishing returns after several
components.

To-do:
    Make graph look pretty?
        Implement seaborn.
"""

import matplotlib.pyplot as plt
from mrsa_ca_rna.import_data import form_matrix
import numpy as np


from mrsa_ca_rna.pca import perform_PCA

def figure00():

    rna_mat = form_matrix()

    rna_decomp, components, explained = perform_PCA(rna_mat)
    components = np.arange(1, components+1)
    fig00 = plt.figure()
    plt.plot(components, explained)
    plt.title("'Completeness' of PCA Decomposition" )
    plt.xlabel("# of Principle Components")
    plt.ylabel("Fraction of variance explained")
    fig00.savefig("./output/fig00")

# debug
figure00()