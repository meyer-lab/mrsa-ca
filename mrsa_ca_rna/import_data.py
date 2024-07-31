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


# further digesting the full CA_series_matrix from the primary source required
# def import_ca_series():
#     ca_val_meta = pd.read_csv(
#         join(BASE_DIR, "mrsa_ca_rna", "data", "CA_series_matrix_compat.txt"),
#         delimiter="\t",
#         index_col=0,
#     )

#     ca_val_meta.reset_index(inplace=True)
#     j = 0
#     for i in range(len(ca_val_meta.index)):
#         if ca_val_meta["!Sample_title"].duplicated()[i]:
#             j += 1
#             ca_val_meta.iloc[i, 0] = (
#                 ca_val_meta.iloc[i, 0]
#                 + "."
#                 + str(j)
#                 + "."
#                 + ca_val_meta.iloc[i, 1].split(": ")[0]
#             )
#         elif j != 0:
#             ca_val_meta.iloc[i - j - 1, 0] = (
#                 ca_val_meta.iloc[i - j - 1, 0]
#                 + ".0."
#                 + ca_val_meta.iloc[i - j - 1, 1].split(": ")[0]
#             )
#             j = 0

#     ca_val_meta.set_index("!Sample_title", inplace=True)
#     ca_val_meta = ca_val_meta.T

#     for label in ca_val_meta.columns[8:21]:
#         ca_val_meta[label] = ca_val_meta[label].str.split(": ").str[1]

#     ca_val_meta.reset_index(names=["!Sample_title"], inplace=True)
#     ca_val_meta["!Sample_title"] = ca_val_meta["!Sample_title"].str.split(" ").str[0]
#     ca_val_meta.set_index(keys="!Sample_title", drop=True, inplace=True)

#     all_ca_data = pd.concat([import_ca_rna(), import_ca_val_rna()], axis=0)

#     same_index = pd.concat([all_ca_data, ca_val_meta], axis=1, join="inner") # empty DataFrame if none of the indices are the same between the two dataframes: True

#     same = 0 # stays 0 if no subject_id's overlap with index of rna data: True
#     for i in ca_val_meta.index:
#         for j in range(len(all_ca_data.index)):
#             if ca_val_meta.loc[i, "!Sample_characteristics_ch1.0.subject_id"] == all_ca_data.index[j]:
#                 same += 1

#     repeated_pats = ca_val_meta.loc[ca_val_meta["!Sample_characteristics_ch1.0.subject_id"].duplicated(), "!Sample_characteristics_ch1.3.daysreltofirsttimepoin"]

#     rna_repeated = pd.concat([repeated_pats, all_ca_data], axis=1)

#     return ca_val_meta


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
            ca_meta_ch_t.loc[:, ["subject_id", "time", "status", "disease"]],
            ca_rna,
        ],
        axis=1,
        keys=["meta", "rna"],
        join="inner",
    )

    return ca_rna_timed


def concat_datasets():
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
    mrsa_rna.loc[mrsa_rna["status"].str.contains("Unknown"), "status"] = mrsa_val_meta[
        "status"
    ].astype(str)
    mrsa_rna = mrsa_rna.reset_index(names=["subject_id"]).set_index(
        "subject_id", drop=False
    )
    mrsa_rna.index.name = None
    rna_list.append(mrsa_rna)

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
                ["subject_id", "disease", "status"],
            ],
            cah_rna,
        ],
        axis=1,
        join="inner",
    )
    rna_list.append(ca_rna)

    ca_timed = extract_time_data()
    desired_meta, desired_rna = (
        pd.IndexSlice["meta", ["subject_id", "disease", "status"]],
        pd.IndexSlice["rna", :],
    )
    ca_timed: pd.DataFrame = ca_timed.loc[:, desired_meta].join(
        ca_timed.loc[:, desired_rna]
    )
    ca_timed.columns = ca_timed.columns.droplevel(0)
    rna_list.append(ca_timed)

    healthy_rna = pd.concat(
        [
            ca_meta_ch_nt.loc[
                ca_meta_ch_nt["disease"] == "Healthy",
                ["subject_id", "disease", "status"],
            ],
            cah_rna,
        ],
        axis=1,
        join="inner",
    )
    rna_list.append(healthy_rna)

    # concat everything within the list we've been appending onto.
    rna_df = pd.concat(rna_list, axis=0, join="inner")

    rna_dfmi = rna_df.set_index(["disease", rna_df.index])

    meta_arr = ["meta" for _ in range(2)]
    rna_arr = ["rna" for _ in range(2, len(rna_dfmi.columns))]
    meta_rna = list(np.concatenate((meta_arr, rna_arr), axis=None))
    mi_columns = pd.MultiIndex.from_arrays([meta_rna, rna_dfmi.columns])

    rna_dfmi.columns = mi_columns

    # scale the rna values before returning
    rna_dfmi["rna"] = scale(rna_dfmi["rna"].to_numpy())

    return rna_dfmi


def import_rna_weights():
    """
    Imports the wights generated from the elasticnet logistic regression performed
    on the RNA post-scaling.
    """

    rna_weights = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "rna_weights_scaled.csv"),
        delimiter=",",
        index_col=0,
    )

    new_weights = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "weights_scaled_newCA.csv"),
        delimiter=",",
        index_col=0,
    )

    nested_weights = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "nested_weights.csv"),
        delimiter=",",
        index_col=0,
    )

    return rna_weights, new_weights, nested_weights
