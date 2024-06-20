"""
Perform the PCA analysis on the formed matrix from import_data.py

To-do:
    Relearn PCA and SVD to confirm I know what I'm graphing
        and why. Also, get confirmation about what I'm
        hoping to show (differences diseases across genes?).
    Arrange rna_comp into a pandas DataFrame for ease of use before
        returning it.
            Index labels: mrsa or ca
            Column labels: PC1, PC2, PC3, etc.
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
    
    Returns: rna_decomp (ndarray), opt_comp (int), total_explained (ndarray)
    """


    """
    Cycle through a an increasing amount of components to perform pca.
    Perform the fit_transform of the data in-situ once optimal components
    are found, or the max # of components tested if threshold is not reached.
    """
    components = np.arange(1, 101)
    total_explained = []
    threshold = 0.95
    change_threshold = 0.005
    for component in components:
        pca = PCA(n_components=component)
        rna_decomp = pca.fit_transform(rna_mat)
        total_explained.append(pca.explained_variance_ratio_.sum())
        try:
            delta = (total_explained[-1]-total_explained[-2])/total_explained[-2]
        except:
            print("Skipping delta calculation with 1 component")
        else:
            print(f"Trying {component} components with {delta*100}% more variance explained...")
            if pca.explained_variance_ratio_.sum() >= threshold:
                opt_comp = component
                print(f"Explained variance matches or exceeds threshold at {component} components")
                break
            elif (component > 1 and delta < change_threshold):
                opt_comp = component
                print(f"Failed to reach threshold before explained variance delta dopped below 0.1% at {component} components")
                break
    

    # print(f"shape of loadings: {pca.components_.shape}, (components, gene)")
    # print(f"Shape of scores: {rna_decomp.shape} (patients, components)")
    # print(rna_decomp)

    return rna_decomp, int(opt_comp), total_explained

# debug calls
# perform_PCA()
