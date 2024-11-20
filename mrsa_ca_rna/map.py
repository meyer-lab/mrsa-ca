"""This file contains the map function for the disease datasets"""

# main module imports
import numpy as np

# secondary module imports
from pacmap import PaCMAP

# local module imports

def perform_pacmap(matrix_list:list):
    """
    Perform the pacmap function on the list of matrices
    
    Accepts:
        matrix_list (list): list of matrices to perform pacmap on

    Returns:
        pacmap_list (list): list of pacmap results
    """
    # Define a pacmap object for us to use
    pacmap = PaCMAP(n_components=2, n_neighbors=None)

    # Perform the pacmap function on the list of matrices
    concat_mat = np.concatenate(matrix_list, axis=0)

    mapped_mat = pacmap.fit_transform(concat_mat)

    return mapped_mat