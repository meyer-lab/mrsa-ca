"""
Import data from our compiled aligned RNA-seq datasets.

Import MRSA and Candida metadata specifically for persistence predicitons
downstream.

For now, we keep track of the study numbers and the primary disease represented
in all other datasets.
"""

from os.path import abspath, dirname, join

import anndata as ad
import pandas as pd

BASE_DIR = dirname(dirname(abspath(__file__)))


def parse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Parses metadata from a DataFrame, extracting key-value pairs from the
    'characteristics' column. Handles both dict and string formats."""
    characteristics = metadata["characteristics"].copy()

    def parse_row(row):
        if isinstance(row, dict):
            return row
        elif isinstance(row, str):
            items = [item.split(": ", 1) for item in row.split(",") if ": " in item]
            return {k.strip(): v.strip() for k, v in items if len(k) > 0 and len(v) > 0}
        else:
            raise TypeError(f"Unhandled metadata format: {type(row)} for row {row}")

    parsed_data = characteristics.apply(parse_row)
    result_df = pd.DataFrame.from_records(
        parsed_data.to_list(), index=characteristics.index
    )
    return result_df


def load_expression_data() -> ad.AnnData:
    """Load expression data from master file, label with disease, and return as AnnData object."""

    disease_registry = {
        "MRSA": "MRSA",
        "GSE176262": "CANDIDA",
        "GSE201085": "BREAST_CANCER",
        "GSE177044": "UC_PSC",
        "GSE89403": "TB",
        "GSE124400": "T1DM",
        "GSE161731": "COVID",
        "GSE116006": "LUPUS",
        "GSE162914": "HIV_CM",
        "GSE133378": "ENTEROVIRUS",
        "GSE129882": "ZIKA",
        "GSE133758": "HEALTHY",
        "GSE120178": "RA",
        "GSE173897": "HBV",
        "GSE112927": "KIDNEY",
        "GSE198449": "COVID_MARINES",
        "GSE239933": "BREAST_CANCER_TCR",
        "GSE185263": "SEPSIS",
        "GSE277354": "LEUKEMIA",
        "GSE215865": "COVID_SINAI",
        "GSE115823": "ASTHEMA",
        "GSE221615": "AETHERSCLEROSIS",
    }

    file_path = join(BASE_DIR, "mrsa_ca_rna", "data", "master_expression_data.csv.gzip")

    exp_df = pd.read_csv(file_path, delimiter=",", compression="gzip")

    # Map the disease registry to the data and rename columns
    exp_df["study_id"] = exp_df["study_id"].map(disease_registry)
    exp_df = exp_df.rename(
        columns={
            "GSM": "sample_id",
            "study_id": "disease",
        })

    # Split metadata from expression data and make AnnData object
    metadata = exp_df.loc[:, ["sample_id", "disease"]].copy()
    exp = exp_df.drop(columns=["sample_id", "disease"])
    data_ad = ad.AnnData(X=exp.values, obs=metadata, var=exp.columns.to_frame(name="gene_id"))

    return data_ad

def import_ca_metadata() -> pd.DataFrame:
    """Imports the metadata for the CA dataset from a JSON file
    and returns it as a DataFrame for later addition to an anndata."""

    path = join(BASE_DIR, "mrsa_ca_rna", "data", "GSE176262_characteristics.json")

    metadata = pd.read_json(path, orient="index")

    clinical_var = parse_metadata(metadata)

    # Remove all but the Candidemia samples
    clinical_var = clinical_var.loc[clinical_var["phenotype"] == "Candidemia", :]

    # Add a "status" column
    clinical_var["status"] = "Unknown"

    # Keep only the status column
    clinical_var = clinical_var[["status"]].copy()

    return clinical_var

def import_mrsa_metadata() -> pd.DataFrame:
    """Imports the metadata for the MRSA dataset from 3 sources, metadata_mrsa_tfac.txt,
    metadata_mrsa_tfac_validation.txt, and metadata_mrsa_sra.csv. The tfac files are
    combined to get a comprehensive metadata DataFrame. Then the SRA metadata is parsed
    to map patient sample numbers to SRA numbers."""

    # Load TFAC metadata
    tfac_path = join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa_tfac.txt")
    tfac_val_path = join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa_tfac_validation.txt")

    tfac_metadata = pd.read_csv(tfac_path, sep=",", index_col=0, dtype=str)
    tfac_val_metadata = pd.read_csv(tfac_val_path, sep=",", index_col=0, dtype=str)

    # Find the common indices between the two DataFrames
    common_indices = tfac_metadata.index.intersection(tfac_val_metadata.index)

    # Fill in the Unknown values in the TFAC metadata with the validation metadata
    tfac_metadata.loc[common_indices, "status"] = tfac_val_metadata.loc[common_indices, "status"]

    # Now that "status" is all 1s and 0s, we can convert it to an integer type
    tfac_metadata["status"] = tfac_metadata["status"].astype(int)

    # Load SRA metadata
    sra_path = join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa_sra.csv")
    sra_metadata = pd.read_csv(sra_path, sep=",", index_col=0)

    # Reorient the SRA metadata to have the Library Name as the index
    sra_mapping = sra_metadata.reset_index(names=["SRA"]).set_index("isolate")["SRA"].to_dict()

    # Map the SRA numbers to the sample numbers in the TFAC metadata
    tfac_metadata.index = tfac_metadata.index.map(sra_mapping)

    # Drop any rows in the TFAC metadata that do not have a corresponding SRA number
    mrsa_labels = tfac_metadata.loc[tfac_metadata.index.notna(), "status"].copy()

    return mrsa_labels
