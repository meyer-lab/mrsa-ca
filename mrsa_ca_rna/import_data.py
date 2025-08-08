"""
Import data from various sources, compiled by ARCHS4 and format them
into AnnData objects.

For now, we keep track of the study numbers and the primary diseases represented.
We use the primary disease to label the dataset for the time being using
a disease registry. This is primarily to help a human reader associate imports.

In the future, we might create a more comprehensive disease registry to
better keep track of all diseases represented across all datasets.
"""

import json
from os.path import abspath, dirname, join

import anndata as ad
import h5py as h5
import numpy as np
import pandas as pd

BASE_DIR = dirname(dirname(abspath(__file__)))


def parse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Parses metadata from a DataFrame, extracting key-value pairs from the
    'characteristics_ch1' column.
    """
    metadata_char_ch1 = metadata["characteristics_ch1"].copy()

    def parse_row(row):
        if isinstance(row, str):
            items = [item.split(": ", 1) for item in row.split(",") if ": " in item]
            return {k.strip(): v.strip() for k, v in items if len(k) > 0 and len(v) > 0}
        else:
            return {}

    parsed_data = metadata_char_ch1.apply(parse_row)
    result_df = pd.DataFrame.from_records(
        parsed_data.to_list(), index=metadata_char_ch1.index
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

    # Temprorarily remove studies from a list
    studies_to_remove = [
        "GSE176262",
        "GSE89403",
        "GSE116006",
        "GSE129882",
        "GSE129882",
        "GSE198449",
        "GSE277354",
        "GSE215865",
    ]
    exp_df = exp_df[~exp_df["study_id"].isin(studies_to_remove)]

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

