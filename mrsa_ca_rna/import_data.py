"""
Import mrsa and ca rna data from tfac-mrsa and ca-rna repos respectively.

To-do:
    Change inputs & outputs of functions for ease-of-use when further code is developed.
    Include more pre-processing as I learn what I'm going to need.
    Find longitudinal data in ca data since we know it should be there.
        They will either have the same label if they denote patient or 
        they may just be offset by a constant value if they are samples
        and come from the same patient. Take a look and see what's there.
    Get more data from healthy patients to include in this analysis to
        increase the power of the PCA analysis for finding real, not batch,
        differences in the data.
        Maybe use: https://www.sciencedirect.com/science/article/pii/S2666379122004062 for
        dataset suggestions and perhaps thier methods for identifying viral vs. non-vrial
        infection.
"""

from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

BASE_DIR = dirname(dirname(abspath(__file__)))

# print(f"The base directory is: {BASE_DIR}")

def import_mrsa_rna():
    """
    reads mrsa rna data from rna_combat_tpm_mrsa
    
    Reurns: mrsa_rna (pandas.DataFram)
    """
    mrsa_rna = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "rna_combat_tpm_mrsa.txt.zip"),
        delimiter=",",
        index_col=0,
        engine="c",
        dtype="float64"
    )

    # patient # needs to be converted to int32
    mrsa_rna.index = mrsa_rna.index.astype("int32")


    # # always scale (demean and divide by variance) rna data. scale() expects array or matrix-like, so we numpy
    # mrsa_rna.loc[:,:] = scale(mrsa_rna.to_numpy())

    return mrsa_rna

def import_ca_meta():
    """
    reads ca metadata from discovery_data_meta.txt and trims to Candida phenotype

    Returns:
        ca_pos_meta (pandas.Index): patients positive with CA
        ca_neg_meta (pandas.Index): patients negative with CA
    """
    ca_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "discovery_metadata_ca.txt"),
        delimiter=",",
        index_col=0
    )

    # # drop anything not CA
    # ca_meta.drop(ca_meta.loc[ca_meta["characteristics_ch1.4.phenotype"] != "Candidemia"].index, inplace=True)

    ca_pos_meta = ca_meta.loc[ca_meta["characteristics_ch1.4.phenotype"] == "Candidemia"].index
    ca_neg_meta = ca_meta.loc[ca_meta["characteristics_ch1.4.phenotype"] == "Healthy"].index

    return ca_pos_meta, ca_neg_meta

def import_ca_rna():
    """
    reads ca data from discovery_data_ca
    
    Returns:
        ca_pos_rna (pandas.DataFrame): patient rna positive with ca
        ca_neg_rna (pandas.DataFrame): patient rna negative with ca
    """
    ca_rna = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "discovery_data_ca.txt.gz"),
        delimiter=",",
        # converters={0: lambda x: int(x,16)},
        index_col=0
    )

    # ca_rna.index = ca_rna.index.map(lambda x: int(x, 16))

    # TPM the data across the rows
    ca_rna = ca_rna.mul(1000000/ca_rna.sum(axis=1), axis=0)

    # Trim ca dataset to only those patients with ca or healthy phenotype
    ca_pos_meta, ca_neg_meta = import_ca_meta()
    ca_pos_rna = ca_rna.loc[ca_pos_meta]
    ca_neg_rna = ca_rna.loc[ca_neg_meta]

    

    # Let's check for repeat indices
    matching = 0
    for i, id in enumerate(ca_pos_rna.index):
        for ii in range(len(ca_pos_rna.index)):
            if id == ca_rna.index[ii] and i != ii:
                matching += 1

    # # we want to replace each member of the dataframe, not the index or column labels
    # ca_rna.loc[:,:] = scale(ca_rna.to_numpy())

    return ca_pos_rna, ca_neg_rna


def form_matrix():
    """
    concatenate the two datasets while trimming to shared genes (columns)
    
    Returns:
        rna_combined (pandas.DataFrame): Concatenated MRSA and CA positive + CA negative rna data
    """
    mrsa_rna = import_mrsa_rna()
    # new_mrsa_ind = np.full((len(mrsa_rna.index),), "mrsa")
    # mrsa_rna.set_index(new_mrsa_ind, inplace=True)
    
    ca_pos_rna, ca_neg_rna = import_ca_rna()
    # new_ca_ind = np.full((len(ca_rna.index),), "ca")
    # ca_rna.set_index(new_ca_ind, inplace=True)

    # print(f"mrsa and ca rna matrices are shape: {mrsa_rna.shape} and {ca_rna.shape} respectively")
    rna_combined = pd.concat([mrsa_rna, ca_pos_rna, ca_neg_rna])
    # print(f"size of concatenated matrix: {rna_combined.shape}")
    rna_combined = rna_combined.dropna(axis=1)
    # print(f"shape of the rna_combined after dropping NaN genes: {rna_combined.shape}")
    # print(rna_combined.index)

    # scale the matrix after all the data is added to it
    rna_combined.loc[:,:] = scale(rna_combined.to_numpy())

    return rna_combined

#debug calls
# mrsaImportTest = import_mrsa_rna()
# caImportTest = import_ca_rna()
rna_combined = form_matrix()
# print(mrsaImportTest.columns)
# print(caImportTest.columns)
# import_ca_meta()