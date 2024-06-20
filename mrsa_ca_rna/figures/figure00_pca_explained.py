"""
Generate a figure depicting the explained variance of the PCA
analysis. We're looking to see diminishing returns after several
components.

To-do:
    Populate file with skeleton.
    Make graph look pretty?
        Implement seaborn.
"""

import matplotlib.pyplot as plt
from mrsa_ca_rna.import_data import form_matrix


from mrsa_ca_rna.pca import perform_PCA

def figure00():

    rna_mat = form_matrix()

    rna_decomp, components, explained = perform_PCA(rna_mat)

    fig00 = plt.figure()
    plt.plot([explained, enumerate(explained)])
    fig00.savefig("fig00")

