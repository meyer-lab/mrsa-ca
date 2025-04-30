"""
Import data from the mrsa_ca_rna project for analysis.
Each dataset is imported along with its metadata.
"""

import json
from os.path import abspath, dirname, join

import archs4py as a4
import pandas as pd

BASE_DIR = dirname(dirname(abspath(__file__)))


def parse_metdata(metadata: pd.DataFrame) -> pd.DataFrame:
    # GEO metadata in SOFT format stores clinical data in the characteristics_ch1 column
    metadata_char_ch1 = metadata.loc[:, "characteristics_ch1"].copy()
    metadata_char_ch1 = metadata_char_ch1.str.split(",", expand=True)

    # We're going to rename the columns based on SOFT format (clinical variable: value)
    column_names = {}

    for col in metadata_char_ch1.columns:
        sample = metadata_char_ch1.iloc[0, col]

        # Collect the clinical variables to rename the columns
        col_name = sample.split(": ")[0]
        column_names[col] = col_name

        # Remove the clinical variable name from the value for each element
        metadata_char_ch1[col] = metadata_char_ch1[col].apply(
            lambda x: x.split(": ")[1].strip()
            if isinstance(x, str) and ": " in x
            else x
        )

        # Rename the columns with the extracted clinical variables
        metadata_h = metadata_char_ch1.rename(columns=column_names)

    return metadata_h


def load_archs4(geo_accession):
    file_path = join(BASE_DIR, "mrsa_ca_rna", "data", "human_gene_v2.6.h5")

    # Extract the count data from the ARCHS4 file, fail if not found
    counts = a4.data.series(file_path, geo_accession)
    if not isinstance(counts, pd.DataFrame):
        raise ValueError(
            f"Could not find GEO accession {geo_accession} in the file {file_path}"
        )

    # Extract the metadata from the ARCHS4 file after success with counts
    metadata = a4.meta.series(file_path, geo_accession)

    # Parse the metadata to extract the clinical variables
    metadata = parse_metdata(metadata)

    return counts, metadata


def import_mrsa():
    # Read in mrsa counts
    counts_mrsa = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_mrsa_archs4.csv.gz"),
        index_col=0,
        delimiter=",",
    )

    # Grab mrsa metadata from SRA database since it is not on GEO
    metadata_mrsa = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa.csv.gz"),
        index_col=0,
        delimiter=",",
    )

    # Pair down metadata to only the columns we need
    metadata_mrsa = metadata_mrsa.loc[:, ["Run", "phenotype", "sex"]]
    metadata_mrsa.set_index("Run", inplace=True)

    return counts_mrsa, metadata_mrsa


def import_ca():
    # Read in ca counts
    counts_ca = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_ca_archs4.tsv.gz"),
        index_col=0,
        delimiter="\t",
    )

    # Archs4 web database reports metadata with jsons of dictionaries
    with open(join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_ca_archs4.json")) as f:
        metadata_ca_json = json.load(f)

    metadata_ca = {}
    for gsm_id, gsm_data in metadata_ca_json.items():
        characteristics = gsm_data.get("characteristics", "")

        metadata_ca[gsm_id] = characteristics

    metadata_ca = pd.DataFrame(
        data=metadata_ca.values(),
        index=metadata_ca.keys(),
        columns=["characteristics_ch1"],
    )
    metadata_ca = parse_metdata(metadata_ca)

    # TODO: Pair down metadata to only the columns we need
    # Not yet clear what we need from the metadata
    # Metadata indicates some samples did not pass qc

    return counts_ca, metadata_ca
