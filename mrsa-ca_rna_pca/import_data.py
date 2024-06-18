"""
Import mrsa and ca rna data from tfac-mrsa and ca-rna repos respectively.
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
        join(BASE_DIR, "mrsa-ca_rna_pca", "data", "rna_combat_tpm_mrsa.txt.zip"),
        delimiter=",",
        index_col=0,
        engine="c",
        dtype="float64"
    )

    # patient # needs to be converted to int32
    mrsa_rna.index = mrsa_rna.index.astype("int32")
    print(mrsa_rna)

    # always scale (demean and divide by variance) rna data
    mrsa_rna.loc[:,:] = scale(mrsa_rna.to_numpy())

    return mrsa_rna


mrsaImportTest = import_mrsa_rna()
# print(mrsaImportTest)
