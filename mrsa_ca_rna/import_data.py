"""
Import mrsa and ca rna data from tfac-mrsa and ca-rna repos respectively.

To-do:
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

    CA data does not contain resolving/persisting distinctions, but does contain mortality
        rates. Consider using that for regression.
"""

from os.path import join, dirname, abspath
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

BASE_DIR = dirname(dirname(abspath(__file__)))

# print(f"The base directory is: {BASE_DIR}")

def import_mrsa_meta():
    """
    reads mrsa metadata from patient_metadata_mrsa.txt
    
    Returns:
        mrsa_meta (pandas.DataFrame)
    """

    mrsa_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "patient_metadata_mrsa.txt"),
        delimiter=",",
        index_col=0
    )


    return mrsa_meta


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
        dtype="float64",
    )

    # patient # needs to be converted to int32
    mrsa_rna.index = mrsa_rna.index.astype("int32")

    return mrsa_rna


def import_ca_meta():
    """
    reads ca metadata from discovery_data_meta.txt and trims to Candida phenotype

    Returns:
        ca_meta (pandas.DataFrame): candidemia patient metadata containing only candidemia and healthy phenotypes
    """
    ca_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "discovery_metadata_ca.txt"),
        delimiter=",",
        index_col=0,
    )

    # trim CA metadata down to only candidemia and healthy patients
    ca_meta = ca_meta.loc[
        ca_meta["characteristics_ch1.4.phenotype"].str.contains("Candidemia|Healthy")
    ]

    return ca_meta


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
        index_col=0,
    )

    # ca_rna.index = ca_rna.index.map(lambda x: int(x, 16))

    # TPM the data across the rows
    ca_rna = ca_rna.mul(1000000 / ca_rna.sum(axis=1), axis=0)

    # Trim ca dataset to only those patients with ca or healthy phenotype
    healthy_pats = import_ca_meta()
    ca_rna = ca_rna.loc[healthy_pats.index]

    return ca_rna


def import_ca_val_meta():
    """import validation ca meta data to find longitudinal data"""
    ca_val_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "validation_metadata_ca.txt"),
        delimiter=",",
        index_col=0,
    )

    return ca_val_meta


def import_ca_val_rna():
    """import rna data from the validation group"""
    ca_val_rna = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "validation_data_ca.txt.gz"),
        delimiter=",",
        index_col=0,
    )

    return ca_val_rna


def import_GSE_metadata():
    """
    Read metadata file to determine patient characteristics.

    Returns:
        gse_healthy_pats (pandas.DataFrame): DataFrame of just healthy patients
        gene_conversion (python dict): mapping of GeneID to EmsemblGeneID provided by annot
    """
    # patient metadata for cross checking healthy patient data
    gse_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "GSE114192_modified_metadata.txt.gz"),
        delimiter="\t",
        index_col=0,
    )
    # gene annotation for converting between GeneID and EnsemblGeneID
    gse_annot = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "Human_GRCh38_p13_annot.tsv.gz"),
        delimiter="\t",
        index_col=0,
    )

    # patients moved to index and reindexed to patient ID
    gse_meta = gse_meta.T
    gse_meta = (
        gse_meta.reset_index(names="!Sample_title")
        .set_index("ID_REF")
        .dropna(axis="columns")
    )
    gse_meta.drop(
        "GSM3137557", inplace=True
    )  # This patient isn't in the rna data for some reason?

    # make a gene conversion mapping to use for converting geneID to EnsemblGeneID
    gene_conversion = pd.DataFrame(
        gse_annot.loc[:, "EnsemblGeneID"],
        index=gse_annot.index,
        columns=["EnsemblGeneID"],
    ).dropna(axis=0)
    gene_conversion = dict(
        zip(gene_conversion.index, gene_conversion["EnsemblGeneID"])
    )  # multiple GeneID are mapped to the same EnsemblGeneID

    # we only want healthy patients
    gse_healthy_pat = gse_meta.loc[
        gse_meta["!Sample_title"].str.contains("Healthy_Control")
    ]

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
        index_col=0,
    )
    # transpose to get DataFrame into (patient x gene) form
    gse_rna = gse_rna.T

    gse_healthy_pat, gene_conversion = import_GSE_metadata()
    # just need index of gse_healthy_pat
    gse_healthy_index = gse_healthy_pat.index

    # trim to healthy data only
    gse_rna = gse_rna.loc[gse_healthy_index]
    gse_rna.rename(
        gene_conversion, axis=1, inplace=True
    )  # this results in duplicate column labels

    # must remove all newly generated duplicate columns because of the gene_conversion mapping (itself generated from their annotation file)
    gse_rna = gse_rna.loc[:, ~gse_rna.columns.duplicated()]

    # keep all the successfully mapped genes and discard anything using original convention
    gse_rna = gse_rna.loc[:, gse_rna.columns.str.contains("ENSG", na=False)]
    gse_rna.index.name, gse_rna.columns.name = (
        None,
        None,
    )  # remove any possibly conflicting values before concat

    return gse_rna


def concat_datasets():
    """
    concatenate the two datasets while trimming to shared genes (columns)

    Returns:
        rna_combined (pandas.DataFrame): Concatenated MRSA, CA+, CA-, and extra healthy datasets
        meta_combined (pandas.DataFrame): Concatenated MRSA, CA, and GSE datasets
    """
    # import relevant mrsa and ca data
    mrsa_meta = import_mrsa_meta()
    mrsa_rna = import_mrsa_rna()
    ca_meta = import_ca_meta()
    ca_rna = import_ca_rna()

    # import any addition healthy data to help with analysis power
    gse_meta, _ = import_GSE_metadata()
    gse_healthy = import_GSE_rna()

    # concatenate all the matrices to along patient axis, keeping only the overlapping columns (join="inner")
    rna_combined = pd.concat(
        [mrsa_rna, ca_rna, gse_healthy], axis=0, join="inner"
    )
    rna_combined = rna_combined.dropna(axis=1)

    # normalize and concatenate all the metadata 
    mrsa_meta.drop(mrsa_meta.columns.difference(["status"]), axis=1, inplace=True)
    mrsa_meta["disease"]= "mrsa"

    ca_meta.drop(ca_meta.columns.difference(["characteristics_ch1.4.phenotype"]), axis=1, inplace=True)
    ca_meta.rename(columns={"characteristics_ch1.4.phenotype": "disease"}, inplace=True)
    ca_meta.insert(0, "status", np.full(len(ca_meta.index), "N/A"))
    ca_meta.loc[ca_meta["disease"].str.contains("Candidemia"),"status"] = "unknown"

    gse_meta.drop(gse_meta.columns.difference(["!Sample_title"]), axis=1, inplace=True)
    gse_meta.rename(columns={"!Sample_title": "disease"}, inplace=True)
    gse_meta["disease"] = "Healthy"
    gse_meta.insert(0, "status", np.full(len(gse_meta.index), "N/A"))

    # CA data does not have persistant or resolving types
    meta_combined = pd.concat(
        [mrsa_meta, ca_meta, gse_meta], axis=0, join="inner"
        )

    # scale the matrix after all the data is added to it
    rna_combined.loc[:, :] = scale(rna_combined.to_numpy())
    
    return rna_combined, meta_combined


# debug calls
# import_mrsa_meta()
# import_mrsa_rna()
# import_ca_meta()
# import_ca_rna()
# import_ca_val_meta()
# import_ca_val_rna()
# concat_datasets()
# import_GSE_rna()
# import_GSE_metadata()

