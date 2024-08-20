"""
Import mrsa and ca rna data from tfac-mrsa and ca-rna repos respectively.

To-do:

    Redo CA data imports.
        Get CA directly from the GEO database. Import and modify as desired.
        Time course data should be intact.


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
import anndata as ad
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
        index_col=0,
    )

    return mrsa_meta


def import_mrsa_val_meta():
    mrsa_val_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "validation_patient_metadata_mrsa.txt"),
        delimiter=",",
        index_col=0,
    )

    return mrsa_val_meta


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


def import_ca_disc_meta():
    """
    reads ca metadata from ca_discovery_meta_GSE176260.txt and trims to Candida phenotype

    Returns:
        ca_meta (pandas.DataFrame): candidemia patient metadata containing only candidemia and healthy phenotypes
    """
    ca_disc_meta_full = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "ca_discovery_meta_GSE176260.txt"),
        delimiter="\t",
        index_col=0,
    )
    ca_disc_meta_full.index.name = None  # remove the name to reduce complication later

    # trim to just what we need
    ca_disc_meta_trimmed = ca_disc_meta_full.iloc[[0, 8, 9, 10, 11, 12, 13, 14], :]
    ca_disc_meta_trimmed = ca_disc_meta_trimmed.reset_index(drop=True)
    index_dict = {
        0: "sample_id",
        1: "subject_id",
        2: "QC",
        3: "analysis",
        4: "time",
        5: "disease",
        6: "gender",
        7: "age",
    }
    ca_disc_meta_relabeled = ca_disc_meta_trimmed.rename(index=index_dict)

    # remove the labels from the values themselves by taking only what is after the ": "
    for i in ca_disc_meta_relabeled.index:
        for j in ca_disc_meta_relabeled.columns:
            if len(ca_disc_meta_relabeled.loc[i, j].split(": ")) > 1:
                ca_disc_meta_relabeled.loc[i, j] = ca_disc_meta_relabeled.loc[
                    i, j
                ].split(": ")[1]

    ca_disc_meta_cleaned = (
        ca_disc_meta_relabeled.T
    )  # index by sample instead of by label

    ca_disc_meta_cleaned = ca_disc_meta_cleaned.loc[
        ca_disc_meta_cleaned["QC"].str.contains("Pass"), :
    ].drop("QC", axis=1)

    # index by sample_id instead of title. We may never need the title again from here
    ca_disc_meta_untitled = ca_disc_meta_cleaned.reset_index(
        names=["sample_title"]
    )  # remove this line if we no longer need the title
    ca_disc_meta_reindexed = ca_disc_meta_untitled.set_index("sample_id")

    # sorting by sibject_id and time
    for i in ca_disc_meta_reindexed.index:
        ca_disc_meta_reindexed.loc[i, "time"] = ca_disc_meta_reindexed.loc[
            i, "time"
        ].zfill(2)
    ca_disc_meta_sorted = ca_disc_meta_reindexed.sort_values(by=["subject_id", "time"])

    return ca_disc_meta_sorted


def import_ca_disc_rna():
    """
    reads ca data from discovery_data_ca

    Returns:

    """
    ca_disc_rna_GeneID = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "ca_discovery_TPM_RNA_GSE176260.tsv.gz"),
        delimiter="\t",
        index_col=0,
    )

    ca_disc_rna_annot = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "Human_GRCh38_p13_annot.tsv.gz"),
        delimiter="\t",
        index_col=0,
        dtype=str,
    )

    gene_conversion = dict(
        zip(ca_disc_rna_annot.index, ca_disc_rna_annot["EnsemblGeneID"])
    )

    ca_disc_rna = ca_disc_rna_GeneID.rename(
        gene_conversion, axis=0
    )  # convert from GeneID to EnsemblGeneID
    ca_disc_rna.rename(
        index=str, inplace=True
    )  # convert nans to strings for subsequent compare
    ca_disc_rna = ca_disc_rna.loc[
        ca_disc_rna.index.str.contains("ENSG"), :
    ]  # drop all unmapped (nan) genes
    ca_disc_rna = ca_disc_rna.groupby(
        ca_disc_rna.index
    ).last()  # drop all but the last duplicate row indices (non-uniquely mapped EnsemblGeneIDs)
    ca_disc_rna.index.name = None

    ca_disc_rna = ca_disc_rna.T  # index by sample instead of by gene

    return ca_disc_rna


def import_ca_val_meta():
    """import validation ca meta data to find longitudinal data"""
    ca_val_meta_full = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "ca_validation_meta_GSE176261.txt"),
        delimiter="\t",
        index_col=0,
    )

    ca_val_meta_full.index.name = None  # remove the name to reduce complication later

    # trim to just what we need
    ca_val_meta_trimmed = ca_val_meta_full.iloc[[0, 8, 9, 10, 11, 12, 13, 14], :]
    ca_val_meta_trimmed = ca_val_meta_trimmed.reset_index(drop=True)
    index_dict = {
        0: "sample_id",
        1: "subject_id",
        2: "QC",
        3: "analysis",
        4: "time",
        5: "disease",
        6: "gender",
        7: "age",
    }
    ca_val_meta_relabeled = ca_val_meta_trimmed.rename(index=index_dict)

    # remove the labels from the values themselves by taking only what is after the ": "
    for i in ca_val_meta_relabeled.index:
        for j in ca_val_meta_relabeled.columns:
            if len(ca_val_meta_relabeled.loc[i, j].split(": ")) > 1:
                ca_val_meta_relabeled.loc[i, j] = ca_val_meta_relabeled.loc[i, j].split(
                    ": "
                )[1]

    ca_val_meta_cleaned = ca_val_meta_relabeled.T  # index by sample instead of by label

    ca_val_meta_cleaned = ca_val_meta_cleaned.loc[
        ca_val_meta_cleaned["QC"].str.contains("Pass"), :
    ].drop("QC", axis=1)

    # index by sample_id instead of title. We may never need the title again from here
    ca_val_meta_untitled = ca_val_meta_cleaned.reset_index(
        names=["sample_title"]
    )  # remove this line if we no longer need the title
    ca_val_meta_reindexed = ca_val_meta_untitled.set_index("sample_id")

    # sorting by sibject_id and time
    for i in ca_val_meta_reindexed.index:
        ca_val_meta_reindexed.loc[i, "time"] = ca_val_meta_reindexed.loc[
            i, "time"
        ].zfill(2)
    ca_val_meta_sorted = ca_val_meta_reindexed.sort_values(by=["subject_id", "time"])

    return ca_val_meta_sorted


def import_ca_val_rna():
    """import rna data from the validation group"""
    ca_val_rna_GeneID = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "ca_validation_TPM_RNA_GSE176261.tsv.gz"),
        delimiter="\t",
        index_col=0,
    )

    ca_val_rna_annot = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "Human_GRCh38_p13_annot.tsv.gz"),
        delimiter="\t",
        index_col=0,
        dtype=str,
    )

    gene_conversion = dict(
        zip(ca_val_rna_annot.index, ca_val_rna_annot["EnsemblGeneID"])
    )

    ca_val_rna = ca_val_rna_GeneID.rename(
        gene_conversion, axis=0
    )  # convert from GeneID to EnsemblGeneID
    ca_val_rna.rename(
        index=str, inplace=True
    )  # convert nans to strings for subsequent compare
    ca_val_rna = ca_val_rna.loc[
        ca_val_rna.index.str.contains("ENSG"), :
    ]  # drop all unmapped (nan) genes
    ca_val_rna = ca_val_rna.groupby(
        ca_val_rna.index
    ).last()  # drop all but the last duplicate row indices (non-uniquely mapped EnsemblGeneIDs)
    ca_val_rna.index.name = None

    ca_val_rna = ca_val_rna.T  # index by sample instead of by gene

    return ca_val_rna


def extract_time_data():
    ca_disc_meta = import_ca_disc_meta()
    ca_val_meta = import_ca_val_meta()
    ca_disc_rna = import_ca_disc_rna()
    ca_val_rna = import_ca_val_rna()

    ca_rna = pd.concat([ca_disc_rna, ca_val_rna], axis=0, join="inner")
    ca_meta = pd.concat([ca_disc_meta, ca_val_meta], axis=0, join="inner")

    ca_meta_ch = ca_meta.loc[
        ca_meta["disease"].str.contains("Candidemia|Healthy"), :
    ]  # grab just data marked as healthy or candidemia

    # extract the time data by looking for duplicate subject_id
    ca_meta_ch_t = ca_meta_ch.loc[ca_meta_ch["subject_id"].duplicated(keep=False), :]
    ca_meta_ch_t["status"] = "Unknown"

    ca_rna_timed = pd.concat(
        [
            ca_meta_ch_t.loc[:, ["subject_id", "gender", "age", "time", "disease", "status"]],
            ca_rna,
        ],
        axis=1,
        keys=["meta", "rna"],
        join="inner",
    )

    # put the time data into an anndata object using metadata as obs and rna as var
    ca_rna_timed_ad = ad.AnnData(ca_rna_timed["rna"], obs=ca_rna_timed["meta"])

    return ca_rna_timed_ad


def concat_datasets(scaled: bool=True, tpm: bool=True):
    """
    concatenate rna datasets of interest into a single dataframe for analysis

    Returns:
        rna_df (pandas.DataFrame): single annotated (status, disease) dataframe of rna data from the imported datasets
    """

    # start a list of rna datasets to be concatenated at the end of this function
    rna_list = list()

    """import mrsa data and set up mrsa_rna df with all required annotations. Includes 'validation' dataset"""
    mrsa_meta = import_mrsa_meta()
    mrsa_val_meta = import_mrsa_val_meta()
    mrsa_rna = import_mrsa_rna()

    # insert a disease and status column, keeping status as strings to avoid data type mixing with CA status: "Unknown"
    mrsa_rna.insert(0, column="status", value=mrsa_meta["status"].astype(str))
    mrsa_rna.insert(0, column="disease", value="MRSA")
    mrsa_rna.insert(0, column="time", value="NA")
    mrsa_rna.insert(0, column="age", value=mrsa_meta["age"])
    mrsa_rna.insert(0, column="gender", value=mrsa_meta["gender"])
    mrsa_rna.loc[mrsa_rna["status"].str.contains("Unknown"), "status"] = mrsa_val_meta[
        "status"
    ].astype(str)
    mrsa_rna = mrsa_rna.reset_index(names=["subject_id"]).set_index(
        "subject_id", drop=False
    )
    mrsa_rna.index.name = None

    # send the mrsa_rna pd.DataFrame to an Anndata object with ENSG as var names and all other columns as obs
    mrsa_ad = ad.AnnData(mrsa_rna.loc[:, mrsa_rna.columns.str.contains("ENSG")], obs=mrsa_rna.loc[:, ~mrsa_rna.columns.str.contains("ENSG")])
    rna_list.append(mrsa_ad)

    """import ca data and set up ca_rna df with all required annotations. Includes 'validation' dataset"""
    ca_disc_meta = import_ca_disc_meta()
    ca_val_meta = import_ca_val_meta()
    ca_disc_rna = import_ca_disc_rna()
    ca_val_rna = import_ca_val_rna()

    # combine the discovery and validation data together, then make the ca_rna df with annotations: Healthy->NA, CA->Unknown
    cah_rna = pd.concat([ca_disc_rna, ca_val_rna], axis=0, join="inner")
    ca_meta = pd.concat([ca_disc_meta, ca_val_meta], axis=0, join="inner")

    ca_meta_ch = ca_meta.loc[
        ca_meta["disease"].str.contains("Candidemia|Healthy"), :
    ]  # grab just data marked as healthy or candidemia

    # remove duplicated patients (ones with time points) so we're not doubling up when we add them back in.
    ca_meta_ch_nt = ca_meta_ch.loc[~ca_meta_ch["subject_id"].duplicated(keep=False), :]
    # remove any additional patients that the paper also exlcuded from their analysis.
    ca_meta_ch_nt = ca_meta_ch_nt.loc[ca_meta_ch_nt["analysis"] == "Yes", :]
    ca_meta_ch_nt["status"] = "Unknown"

    ca_rna = pd.concat(
        [
            ca_meta_ch_nt.loc[
                ca_meta_ch_nt["disease"] == "Candidemia",
                ["subject_id", "gender", "age", "time", "disease", "status"],
            ],
            cah_rna,
        ],
        axis=1,
        join="inner",
        keys=["meta", "rna"]
    )
    ca_rna_ad = ad.AnnData(ca_rna["rna"], obs=ca_rna["meta"])
    rna_list.append(ca_rna_ad)

    ca_timed_ad = extract_time_data()
    rna_list.append(ca_timed_ad)

    healthy_rna = pd.concat(
        [
            ca_meta_ch_nt.loc[
                ca_meta_ch_nt["disease"] == "Healthy",
                ["subject_id", "gender", "age", "time", "disease", "status"],
            ],
            cah_rna,
        ],
        axis=1,
        join="inner",
        keys=["meta", "rna"]
    )
    healthy_rna_ad = ad.AnnData(healthy_rna["rna"], obs=healthy_rna["meta"])
    rna_list.append(healthy_rna_ad)

    # concat all anndata objects together keeping only the vars in common and expanding the obs to include all
    rna_ad = ad.concat(rna_list, axis=0, join="inner")

    # re-TPM the RNA data by default by normalizing each row to 1,000,000
    if tpm:
        desired_value = 1000000

        X = rna_ad.X
        row_sums = X.sum(axis=1)

        scaling_factors = desired_value / row_sums

        X_normalized = X * scaling_factors[:, np.newaxis]

        rna_ad.X = X_normalized

    if scaled:
        rna_ad.X = scale(rna_ad.X)

    # # re-TPM the RNA data by default. This is a per-row action and leaks nothing between datasets
    # if tpm:
        
    #     rna_dfmi.iloc[:, 2:] = rna_dfmi.iloc[:, 2:].div(
    #         rna_dfmi.iloc[:, 2:].sum(axis=1) / 1000000, axis=0
    #     )

    # # scale the rna values before returning, this forever entangles datasets together and must not be done if separately considering sections
    # if scaled:
        
    #     rna_dfmi["rna"] = scale(rna_dfmi["rna"].to_numpy())

    return rna_ad