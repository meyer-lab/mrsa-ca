"""
Import mrsa and ca rna data from tfac-mrsa and ca-rna repos respectively.

To-do:

    Evaluate the need for the extra healthy data.
        I am importing extra healthy data from a paper cited by the CA paper.
        Do I really need to do this and might it break something? Cannot confirm
        where and how that data was collected and this data may be particularly
        context sensitive.

    Impute CA presistence/resolving data:
        Take paper information on probability of healthy vs. time and use that
        probability distribution to assign patients as reolving or persisting
        based on their "daysreltofirsttimepoint" data. Then mrsa regression
        model to see performance.

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
    ca_healthy_pats = import_ca_meta()
    ca_rna = ca_rna.loc[ca_healthy_pats.index]

    return ca_rna


def import_ca_val_meta():
    """import validation ca meta data to find longitudinal data"""
    ca_val_meta = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "validation_metadata_ca.txt"),
        delimiter=",",
        index_col=0,
    )

    ca_val_meta = ca_val_meta.loc[
        ca_val_meta["characteristics_ch1.4.phenotype"].str.contains(
            "Candidemia|Healthy"
        )
    ]

    return ca_val_meta


def import_ca_val_rna():
    """import rna data from the validation group"""
    ca_val_rna = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "validation_data_ca.txt.gz"),
        delimiter=",",
        index_col=0,
    )

    ca_val_rna = ca_val_rna.mul(1000000 / ca_val_rna.sum(axis=1), axis=0)

    # trim ca_val_rna to just those patients with CA or are Healthy
    ca_healthy_pats = import_ca_val_meta()
    ca_val_rna = ca_val_rna.loc[ca_healthy_pats.index]

    return ca_val_rna


# further digesting the full CA_series_matrix from the primary source required
def import_ca_series():
    ca_series = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "CA_series_matrix_compat.txt"),
        delimiter="\t",
        index_col=0,
    )

    ca_series.reset_index(inplace=True)
    j = 0
    for i in range(len(ca_series.index)):
        if ca_series["!Sample_title"].duplicated()[i]:
            j += 1
            ca_series.iloc[i, 0] = (
                ca_series.iloc[i, 0]
                + "."
                + str(j)
                + "."
                + ca_series.iloc[i, 1].split(": ")[0]
            )
        elif j != 0:
            ca_series.iloc[i - j - 1, 0] = (
                ca_series.iloc[i - j - 1, 0]
                + ".0."
                + ca_series.iloc[i - j - 1, 1].split(": ")[0]
            )
            j = 0

    ca_series.set_index("!Sample_title", inplace=True)
    ca_series = ca_series.T

    for label in ca_series.columns[8:21]:
        ca_series[label] = ca_series[label].str.split(": ").str[1]

    ca_series.reset_index(names=["!Sample_title"], inplace=True)
    ca_series["!Sample_title"] = ca_series["!Sample_title"].str.split(" ").str[0]
    ca_series.set_index("!Sample_title", drop=True)

    return ca_series


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
    mrsa_rna.insert(0, column="disease", value=np.full(len(mrsa_rna.index), "MRSA"))
    mrsa_rna.insert(0, column="status", value=mrsa_meta["status"].astype(str))
    mrsa_rna.loc[mrsa_rna["status"].str.contains("Unknown"), "status"] = mrsa_val_meta[
        "status"
    ].astype(str)
    rna_list.append(mrsa_rna)

    """import ca data and set up ca_rna df with all required annotations. Includes 'validation' dataset"""
    ca_meta = import_ca_meta()
    ca_val_meta = import_ca_val_meta()
    ca_rna = import_ca_rna()
    ca_val_rna = import_ca_val_rna()

    # combine the discovery and validation data together, then make the ca_rna df with annotations: Healthy->NA, CA->Unknown
    ca_rna = pd.concat([ca_rna, ca_val_rna], axis=0, join="inner")
    ca_meta = pd.concat([ca_meta, ca_val_meta], axis=0, join="inner")
    ca_rna.insert(0, column="disease", value=ca_meta["characteristics_ch1.4.phenotype"])
    ca_rna.insert(0, column="status", value=np.full(len(ca_rna.index), "NA"))
    ca_rna.loc[ca_rna["disease"].str.contains("Candidemia"), "status"] = "Unknown"
    rna_list.append(ca_rna)

    """Might be ditching the extra healthy data as context might be important"""
    # # import any additional healthy data to help with analysis power
    # gse_meta, _ = import_GSE_metadata()
    # gse_healthy = import_GSE_rna()

    # gse_healthy.insert(0, column="disease", np.full(len(gse_healthy.index), "Healthy"))
    # gse_healthy.insert(0, column="status", np.full(len(gse_healthy.index), "NA"))
    # rna_list.append(gse_healthy)

    # concat everything within the list we've been appending onto.
    rna_df = pd.concat(rna_list, axis=0, join="inner")

    return rna_df


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

    return rna_weights


import_rna_weights()

"""Unclear whether this function will be valid after refactoring the concat_datasets function"""


def validation_data():
    # import mrsa data for validation dataset generation
    mrsa_val_meta = import_mrsa_val_meta()
    mrsa_rna = import_mrsa_rna()

    # attach statuses and drop all non-statused patients (leaving only validation set), then specify disease
    mrsa_rna.insert(0, "status", mrsa_val_meta["status"])
    mrsa_rna.dropna(axis=0, inplace=True)
    mrsa_rna.insert(0, "disease", np.full(len(mrsa_rna.index), "mrsa"))
    mrsa_rna["status"] = mrsa_rna["status"].astype(
        int
    )  # astype did not work during insertion step above

    # import ca data for validation dataset generation
    ca_val_meta = import_ca_val_meta()
    # ca_series = import_ca_series()
    ca_val_rna = import_ca_val_rna()

    """Will be used later when I get back to cross-checking ca_series data with the rest of the CA data"""
    # df1 = pd.DataFrame(np.zeros(len(ca_val_rna.index)), index=ca_val_rna.index, columns=["id1"])
    # df2 = pd.DataFrame(np.zeros(len(ca_series["!Sample_characteristics_ch1.0.subject_id"])), index=ca_series["!Sample_characteristics_ch1.0.subject_id"], columns=["id2"])
    # df_comb = pd.concat([df1, df2]).dropna()

    # bring in CA validation data and attach disease ("Candidemia"/"Healthy") and statuses ("unknown"/"N/A")
    ca_val_rna.insert(0, "disease", ca_val_meta["characteristics_ch1.4.phenotype"])
    ca_val_rna.dropna(axis=0, inplace=True)
    ca_val_rna.insert(1, "status", np.full(len(ca_val_rna.index), "unknown"))
    ca_val_rna.loc[ca_val_rna["disease"].str.contains("Healthy"), "status"] = "N/A"

    val_rna_combined = pd.concat([mrsa_rna, ca_val_rna], axis=0, join="inner")

    val_rna_combined.iloc[:, 2:] = scale(val_rna_combined.iloc[:, 2:].to_numpy())

    return val_rna_combined
