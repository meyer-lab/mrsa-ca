"""
Import data from the mrsa_ca_rna project for analysis.
Each dataset is imported along with its metadata.
"""

from os.path import abspath, dirname, join
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd

BASE_DIR = dirname(dirname(abspath(__file__)))


# annotations for the human genome
def import_human_annot():
    human_annot = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "Human_GRCh38_p13_annot.tsv.gz"),
        delimiter="\t",
        index_col=0,
        dtype=str,
    )

    return human_annot


# start with metadata for MRSA
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


# import validation metadata for MRSA
def import_mrsa_val_meta():
    mrsa_val_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "validation_patient_metadata_mrsa.txt"),
        delimiter=",",
        index_col=0,
    )

    return mrsa_val_meta


# this dataset contains all mrsa rna data
def import_mrsa_rna():
    """
    reads mrsa rna data from rna_combat_tpm_mrsa

    Reurns:
        mrsa_rna (anndata.AnnData): mrsa rna data with all required annotations
    """
    mrsa_rna = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "rna_combat_tpm_mrsa.txt.zip"),
        delimiter=",",
        index_col=0,
        engine="c",
    )

    mrsa_meta = import_mrsa_meta()

    # trim the metadata to just those shared with the rna data
    mrsa_meta = mrsa_meta.loc[mrsa_rna.index, :]

    mrsa_val_meta = import_mrsa_val_meta()

    # add the metadata to the rna data. value columns should be converted to list-like
    mrsa_rna.insert(
        0, column="status", value=mrsa_meta["status"].astype(str).to_numpy()
    )
    mrsa_rna.insert(0, column="disease", value="MRSA")
    mrsa_rna.insert(0, column="time", value="NA")
    mrsa_rna.insert(0, column="age", value=mrsa_meta["age"].astype(str).to_numpy())
    mrsa_rna.insert(
        0, column="gender", value=mrsa_meta["gender"].astype(str).to_numpy()
    )
    mrsa_rna.loc[mrsa_rna["status"].str.contains("Unknown"), "status"] = mrsa_val_meta[
        "status"
    ].astype(str)
    mrsa_rna = mrsa_rna.reset_index(names=["subject_id"]).set_index(
        "subject_id", drop=False
    )
    mrsa_rna.index.name = None
    mrsa_rna.index = mrsa_rna.index.astype("str")  # anndata requires string index

    # send the mrsa_rna pd.DataFrame to an Anndata object with
    # ENSG as var names and all other columns as obs
    mrsa_ad = ad.AnnData(
        mrsa_rna.loc[:, mrsa_rna.columns.str.contains("ENSG")],
        obs=mrsa_rna.loc[:, ~mrsa_rna.columns.str.contains("ENSG")],
    )

    return mrsa_ad


def import_ca_disc_meta():
    """
    reads ca metadata from ca_discovery_meta_GSE176260.txt
    and trims to Candida phenotype

    Returns:
        ca_meta (pandas.DataFrame): candidemia patient metadata containing
                                    only candidemia and healthy phenotypes
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

    ca_disc_rna_annot = import_human_annot()

    gene_conversion = dict(
        zip(ca_disc_rna_annot.index, ca_disc_rna_annot["EnsemblGeneID"], strict=False)
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
    ).last()  # drop all but the last duplicate row indices
    # (non-uniquely mapped EnsemblGeneIDs)
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

    ca_val_rna_annot = import_human_annot()

    gene_conversion = dict(
        zip(ca_val_rna_annot.index, ca_val_rna_annot["EnsemblGeneID"], strict=False)
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
    ).last()  # drop all but the last duplicate row indices
    # (non-uniquely mapped EnsemblGeneIDs)
    ca_val_rna.index.name = None

    ca_val_rna = ca_val_rna.T  # index by sample instead of by gene

    return ca_val_rna


def ca_data_split():
    ca_disc_meta = import_ca_disc_meta()
    ca_val_meta = import_ca_val_meta()
    ca_disc_rna = import_ca_disc_rna()
    ca_val_rna = import_ca_val_rna()

    ca_rna = pd.concat([ca_disc_rna, ca_val_rna], axis=0, join="inner")
    ca_meta = pd.concat([ca_disc_meta, ca_val_meta], axis=0, join="inner")

    # add a status column to all meta data
    ca_meta["status"] = "Unknown"

    # seperate out the candidemia and healthy data
    ca_meta_c = ca_meta.loc[ca_meta["disease"] == "Candidemia", :]
    ca_meta_h = ca_meta.loc[ca_meta["disease"] == "Healthy", :]

    # extract the time data by looking for duplicate subject_id.
    # Duplicates = time points
    ca_meta_c_t = ca_meta_c.loc[ca_meta_c["subject_id"].duplicated(keep=False), :]
    ca_meta_c_nt = ca_meta_c.loc[~ca_meta_c["subject_id"].duplicated(keep=False), :]

    gene_labels = list(ca_rna.columns)
    meta_labels = ["subject_id", "gender", "age", "time", "disease", "status"]

    # make dataframes of the time, non-time, and healthy data
    # easily keep only shared samples by using the join="inner" argument
    ca_rna_timed = pd.concat(
        [
            ca_meta_c_t.loc[:, meta_labels],
            ca_rna,
        ],
        axis=1,
        keys=["meta", "rna"],
        join="inner",
    )

    ca_rna_nontimed = pd.concat(
        [
            ca_meta_c_nt.loc[:, meta_labels],
            ca_rna,
        ],
        axis=1,
        keys=["meta", "rna"],
        join="inner",
    )

    healthy_rna = pd.concat(
        [
            ca_meta_h.loc[:, meta_labels],
            ca_rna,
        ],
        axis=1,
        join="inner",
        keys=["meta", "rna"],
    )

    # put the time and non-timed data into anndata objects
    # use the lables with loc to allow pyright to know they are dataframes
    # because we have to send dataframes, we have to drop the top level for anndata
    ca_rna_timed_ad = ad.AnnData(
        ca_rna_timed.loc[:, ("rna", gene_labels)].droplevel(0, axis=1),
        obs=ca_rna_timed.loc[:, ("meta", meta_labels)].droplevel(0, axis=1),
    )
    ca_rna_nontimed_ad = ad.AnnData(
        ca_rna_nontimed.loc[:, ("rna", gene_labels)].droplevel(0, axis=1),
        obs=ca_rna_nontimed.loc[:, ("meta", meta_labels)].droplevel(0, axis=1),
    )
    healthy_rna_ad = ad.AnnData(
        healthy_rna.loc[:, ("rna", gene_labels)].droplevel(0, axis=1),
        obs=healthy_rna.loc[:, ("meta", meta_labels)].droplevel(0, axis=1),
    )

    ca_list = [ca_rna_timed_ad, ca_rna_nontimed_ad, healthy_rna_ad]

    # re-TPM the RNA data by default by normalizing each row to 1,000,000
    desired_value = 1000000

    for ca_ad in ca_list:
        # I know ca_ad.X is an ndarray, but pyright doesn't
        # replace this with proper type gating to avoid the cast
        X = cast(np.ndarray, ca_ad.X)
        row_sums = X.sum(axis=1)

        scaling_factors = desired_value / row_sums

        X_normalized = X * scaling_factors[:, np.newaxis]

        ca_ad.X = X_normalized

    return ca_rna_timed_ad, ca_rna_nontimed_ad, healthy_rna_ad


# concat the ca datasets
def concat_ca():
    """
    Concatenate the CA data along the patient axis and
    return an annotated AnnData object.

    Returns:
        ca_ad (AnnData): concatenated CA data with all required annotations.
    """

    ca_timed, ca_nontimed, _ = ca_data_split()

    # concat all anndata objects together keeping only the vars in common
    #  and expanding the obs to include all
    ca_ad = ad.concat([ca_timed, ca_nontimed], axis=0, join="inner")

    return ca_ad


def import_breast_cancer_meta():
    """import breast cancer metadata from the GEO file"""
    breast_cancer_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "BreastCancer_GSE201085_meta.csv.gz"),
        delimiter=",",
        index_col=0,
    )

    breast_cancer_meta = breast_cancer_meta.loc[:, ["ER", "PR", "HER2", "Recur"]]

    # pandas currently performs a silent casting during replace,
    # altering the column types
    # in the future, this will no longer occur, but it currently throws a warning
    # to opt-in to the future behavior, we use the following context manager
    with pd.option_context("future.no_silent_downcasting", True):
        # change all instances of "neg" to 0 and "pos" to 1
        breast_cancer_meta = breast_cancer_meta.replace("neg", 0)
        breast_cancer_meta = breast_cancer_meta.replace("pos", 1)
        # change all instance of "NO" to 0 and "YES" or Nan to 1
        breast_cancer_meta = breast_cancer_meta.replace("NO", 0)
        breast_cancer_meta = breast_cancer_meta.replace("YES", 1)
        breast_cancer_meta = breast_cancer_meta.fillna(
            1
        )  # labeling all non-treated people as having recurred for now

    # add subject_id and disease to the metadata
    breast_cancer_meta["subject_id"] = breast_cancer_meta.index
    breast_cancer_meta["disease"] = "BreastCancer"

    return breast_cancer_meta


def import_breast_cancer(tpm: bool = True):
    """import breast cancer data from the GEO file"""
    breast_cancer = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "BreastCancer_GSE201085_tpm.csv.gz"),
        delimiter=",",
        index_col=0,
    )

    # import metadata for making an anndata object
    breast_cancer_meta = import_breast_cancer_meta()

    # import the human genome annotation file and make a gene conversion dictionary
    human_annot = import_human_annot()
    gene_conversion = dict(
        zip(human_annot["Symbol"], human_annot["EnsemblGeneID"], strict=False)
    )

    # re-map the gene symbol to EnsemblGeneID
    breast_cancer = breast_cancer.rename(gene_conversion, axis=0)

    # drop all unmapped (NaN and non-ENSG) genes and
    # drop all but the last duplicate row indices
    breast_cancer = breast_cancer.loc[
        breast_cancer.index.str.contains("ENSG", na=False), :
    ]
    breast_cancer = breast_cancer.groupby(breast_cancer.index).last()

    # swap genes to columns and samples to rows
    breast_cancer = breast_cancer.T

    # make an anndata object with the rna data and the metadata
    breast_cancer_ad = ad.AnnData(breast_cancer, obs=breast_cancer_meta)

    if tpm:
        desired_value = 1000000

        # I know breast_cancer_ad.X is an ndarray, but pyright doesn't
        # replace this with proper type gating to avoid the cast
        X = cast(np.ndarray, breast_cancer_ad.X)

        row_sums = X.sum(axis=1)

        scaling_factors = desired_value / row_sums

        X_normalized = X * scaling_factors[:, np.newaxis]

        breast_cancer_ad.X = X_normalized

    return breast_cancer_ad


def import_healthy(tpm: bool = True):
    """import and extract healthy data from healthy_source_GSE177044.csv.gz

    Parameters:
        tpm (bool): whether to normalize the data to TPM

    Returns:
        healthy_ad (AnnData): healthy data in an AnnData object
    """
    healthy = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "healthy_source_GSE177044.csv.gz"),
        delimiter=",",
        index_col=0,
    )

    # drop all the columns that are not controls, this also drops gene_name column
    healthy = healthy.loc[:, healthy.columns.str.contains("Control")]

    # # drop the gene name column
    # healthy = healthy.drop("gene_name", axis=1)

    # swap genes to columns and samples to rows
    healthy = healthy.T

    # make a metadata dataframe for the healthy data containing subject_id
    #  made from the index and disease as "Healthy"
    healthy_meta = pd.DataFrame(index=healthy.index)

    healthy_meta["subject_id"] = healthy_meta.index
    healthy_meta["disease"] = "Healthy"

    # make an anndata object with the rna data and the metadata
    healthy_ad = ad.AnnData(healthy, obs=healthy_meta)

    if tpm:
        desired_value = 1000000

        # I know healthy_ad.X is an ndarray, but pyright doesn't
        # replace this with proper type gating to avoid the cast
        X = cast(np.ndarray, healthy_ad.X)
        row_sums = X.sum(axis=1)

        scaling_factors = desired_value / row_sums

        X_normalized = X * scaling_factors[:, np.newaxis]

        healthy_ad.X = X_normalized

    return healthy_ad


def import_covid_meta():
    """import covid metadata from the GEO file"""
    covid_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "covid_metadata.txt"),
        delimiter=",",
        index_col=0,
    )

    covid_meta_trimmed = covid_meta.loc[:, ["sex", "disease", "disease_staging"]]

    return covid_meta_trimmed


def import_covid():
    covid_raw = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "covid_all_counts.txt"),
        delimiter="\t",
        index_col=0,
    )
    covid_raw = covid_raw.dropna(axis=1, how="all")

    covid_meta = import_covid_meta()

    covid_raw.reset_index(inplace=True, names="sample_id")
    covid_raw.loc[:, ["sample_id"]] = covid_raw["sample_id"].str.split(
        ".", expand=True
    )[0]
    covid_raw.set_index("sample_id", inplace=True)

    covid_ad = ad.AnnData(covid_raw.T, obs=covid_meta)

    # tpm normalize the data
    desired_value = 1000000
    X = cast(np.ndarray, covid_ad.X)
    row_sums = X.sum(axis=1)
    scaling_factors = desired_value / row_sums
    X_normalized = X * scaling_factors[:, np.newaxis]
    covid_ad.X = X_normalized

    return covid_ad
