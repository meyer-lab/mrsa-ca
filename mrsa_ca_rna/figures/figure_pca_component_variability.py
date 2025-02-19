"""This file will plot the norms of the PCA components for the MRSA CA RNA data to
assess the variability of the components across runs with resampling."""

import numpy as np
import pandas as pd

from sklearn.utils import resample

from mrsa_ca_rna.pca import perform_pca
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():

    mrsa_ca = concat_datasets(["mrsa", "ca"], scale=True)

    # resample the data 1000 times
    resampled_data = []
    for _ in range(1000):
        resampled_data.append(resample(mrsa_ca, replace=True))

    # for each resampled dataset, perform PCA and store the norms of the components
    pca_norms = pd.DataFrame(index=[f"PC{i}" for i in range(1, 71)])
    for i, data in enumerate(resampled_data):
        _, _, pca = perform_pca(data, n_components=70)
        pca_norms[f"Resample {i+1}"] = (np.linalg.norm(pca.components_, axis=1))
        
    return pca_norms
figure_setup()