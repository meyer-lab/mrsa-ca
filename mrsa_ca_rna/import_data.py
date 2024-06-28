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


    return ca_pos_rna, ca_neg_rna

def import_GSE_metadata():
    """
    Read metadata file to determine patient characteristics.

    Returns:
        gse_healthy_pats (pandas.Index): index of ID's for just healthy patients
    """
    gse_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "GSE114192_modified_metadata.txt.gz"),
        delimiter="\t",
        index_col=0
    )

    gse_annot = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "Human_GRCh38_p13_annot.tsv.gz"),
        delimiter="\t",
        index_col=0
    )

    gse_meta = gse_meta.T
    gse_meta = gse_meta.reset_index(names="!Sample_title").set_index("ID_REF").dropna(axis="columns")
    gse_meta.drop("GSM3137557", inplace=True) # This patient isn't in the rna data for some reason?


    gene_conversion = pd.DataFrame(gse_annot.loc[:,"EnsemblGeneID"], index=gse_annot.index, columns=["EnsemblGeneID"]).dropna(axis=0)
    gene_conversion = dict(zip(gene_conversion.index, gene_conversion["EnsemblGeneID"]))

    
    gse_healthy_pat = gse_meta.loc[gse_meta["!Sample_title"].str.contains("Healthy_Control")].index

    return gse_healthy_pat, gene_conversion


def import_GSE_rna():
    """
    Read GSE rna data. We are looking for healthy samples to augment our power with mrsa-ca
    
    Returns:
        gse_rna (pandas.DataFrame): A DataFrame of shape (patient x gene)
    """
    gse_rna = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "GSE114192_norm_counts_TPM.tsv.gz"),
        delimiter="\t",
        index_col=0
    )
    # transpose to get DataFrame into (patient x gene) form
    gse_rna = gse_rna.T

    gse_healthy_index, gene_conversion = import_GSE_metadata()

    gse_rna = gse_rna.loc[gse_healthy_index]
    gse_rna.rename(gene_conversion, axis=1, inplace=True)

    # ca_pos_meta = ca_meta.loc[ca_meta["characteristics_ch1.4.phenotype"] == "Candidemia"].index

    gse_rna = gse_rna.loc[:,gse_rna.columns.str.contains("ENSG")==True]

    return gse_rna


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

    gse_healthy = import_GSE_rna()

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
# import_GSE_rna()
# import_GSE_metadata()