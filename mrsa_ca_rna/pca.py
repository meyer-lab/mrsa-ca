"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Relearn PCA and SVD to confirm I know what I'm graphing
        and why. Also, get confirmation about what I'm
        hoping to show (differences diseases across genes?).
    Perform PCA only once since it is ordered (PC1 is the same at
        1 or 36 components) and then take subsets for graphs and
        calcs.
    Run PCA on longitudinal data that I wither find here or in
        import_data.
"""

from sklearn.decomposition import PCA
from mrsa_ca_rna.import_data import form_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def perform_PCA(rna_mat:pd.DataFrame):
    """
    Perform pca analysis on concatenated rna matrix
    
    Accepts: rna_mat (pd.DataFrame)
    
    Returns: rna_decomp (pd.DataFrame), pca (object)
    """

    
    components = 100 # delta percent explained drops below 0.1% @ ~component 70
    pca = PCA(n_components=components)
    rna_decomp = pca.fit_transform(rna_mat)

    column_labels = []
    for i in range(1,101):
        column_labels.append("PC" + str(i))

    rna_decomp = pd.DataFrame(rna_decomp, rna_mat.index, column_labels)


    return rna_decomp, pca

# debug calls
# rna_mat = form_matrix()
# perform_PCA(rna_mat)
