"""
Import data from the mrsa_ca_rna project for analysis.
Each dataset is imported along with its metadata.
"""

import contextlib
import json
import multiprocessing
from os.path import abspath, dirname, join

import anndata as ad
import archs4py as a4
import pandas as pd

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("spawn")  # loss of speed but avoids fork() issues

BASE_DIR = dirname(dirname(abspath(__file__)))


def parse_metdata(metadata: pd.DataFrame) -> pd.DataFrame:
    # GEO metadata in SOFT format stores clinical data in the characteristics_ch1 column
    metadata_char_ch1 = metadata.loc[:, "characteristics_ch1"].copy()

    # First, collect all unique column names from all rows
    all_keys = set()
    for value in metadata_char_ch1:
        if isinstance(value, str):
            items = value.split(",")
            for item in items:
                if ": " in item:
                    key = item.split(": ")[0].strip()
                    all_keys.add(key)

    # Create a DataFrame with these keys as columns
    result_df = pd.DataFrame(
        index=metadata_char_ch1.index, columns=pd.Index(sorted(all_keys))
    )

    # Fill in the data for each row
    for idx, value in enumerate(metadata_char_ch1):
        if isinstance(value, str):
            items = value.split(",")
            for item in items:
                if ": " in item:
                    try:
                        key, val = item.split(": ", 1)
                        key = key.strip()
                        val = val.strip()
                        result_df.loc[metadata_char_ch1.index[idx], key] = val
                    except ValueError:
                        continue

    """I do not know if this is necessary yet"""
    # # Fill NaN values with "Unknown" to indicate we filled in the data
    # result_df = result_df.fillna("Unkown")

    return result_df


def load_archs4(geo_accession):
    file_path = join(BASE_DIR, "mrsa_ca_rna", "data", "human_gene_v2.6.h5")

    # Extract the count data from the ARCHS4 file, fail if not found
    counts = a4.data.series(file_path, geo_accession)
    if not isinstance(counts, pd.DataFrame):
        raise ValueError(
            f"Could not find GEO accession {geo_accession} in the file {file_path}"
        )
    counts = a4.utils.aggregate_duplicate_genes(counts)
    counts_tmm = a4.utils.normalize(counts=counts, method="tmm", tmm_outlier=0.05)

    # Extract the metadata from the ARCHS4 file after success with counts
    metadata = a4.meta.series(file_path, geo_accession)

    # Parse the metadata to extract the clinical variables
    clinical_variables = parse_metdata(metadata)

    return counts.T, counts_tmm.T, clinical_variables


def import_mrsa():
    # Read in mrsa counts
    counts_mrsa = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_mrsa_archs4.csv.gz"),
        index_col=0,
        delimiter=",",
    )
    counts_mrsa = a4.utils.aggregate_duplicate_genes(counts_mrsa)
    counts_mrsa_tmm = a4.utils.normalize(
        counts=counts_mrsa, method="tmm", tmm_outlier=0.05
    )
    counts_mrsa = counts_mrsa.T
    counts_mrsa_tmm = counts_mrsa_tmm.T

    # Grab mrsa metadata from SRA database since it is not on GEO
    metadata_mrsa = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa.csv"),
        index_col=0,
        delimiter=",",
    )

    # Pair down metadata and ensure "disease" and "status" columns are present
    metadata_mrsa = metadata_mrsa.loc[:, ["phenotype", "sex"]]
    metadata_mrsa.index.name = None
    metadata_mrsa = metadata_mrsa.rename(
        columns={
            "phenotype": "status",
        }
    )
    metadata_mrsa["disease"] = "MRSA"

    # Order the indices of the counts and metadata to match for AnnData
    common_idx = counts_mrsa.index.intersection(metadata_mrsa.index)
    counts_mrsa = counts_mrsa.loc[common_idx]
    counts_mrsa_tmm = counts_mrsa_tmm.loc[common_idx]
    metadata_mrsa = metadata_mrsa.loc[common_idx]

    mrsa_adata = ad.AnnData(
        X=counts_mrsa_tmm,
        obs=metadata_mrsa,
        var=pd.DataFrame(index=counts_mrsa.columns),
    )
    mrsa_adata.layers["raw"] = counts_mrsa

    return mrsa_adata


def import_ca():
    # Read in ca counts
    counts_ca = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_ca_archs4.tsv.gz"),
        index_col=0,
        delimiter="\t",
    )
    counts_ca = a4.utils.aggregate_duplicate_genes(counts_ca)
    counts_ca_tmm = a4.utils.normalize(counts=counts_ca, method="tmm", tmm_outlier=0.05)
    counts_ca = counts_ca.T
    counts_ca_tmm = counts_ca_tmm.T

    # Archs4 web database reports metadata with jsons of dictionaries
    with open(join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_ca_archs4.json")) as f:
        metadata_ca_json = json.load(f)

    metadata_ca = {}
    for gsm_id, gsm_data in metadata_ca_json.items():
        characteristics = gsm_data.get("characteristics", "")

        metadata_ca[gsm_id] = characteristics

    metadata_ca = pd.DataFrame(
        data=metadata_ca.values(),
        index=pd.Index(list(metadata_ca.keys())),
        columns=pd.Index(["characteristics_ch1"]),
    )
    metadata_ca = parse_metdata(metadata_ca)  # includes qc failures

    # Pair down metadata and ensure "disease" and "status" columns are present
    metadata_ca = metadata_ca.loc[
        :,
        pd.Index(
            [
                "subject_id",
                "passed sample qc",
                "daysreltofirsttimepoin",
                "phenotype",
                "gender",
                "age",
            ]
        ),
    ]
    metadata_ca = metadata_ca.rename(
        columns={
            "daysreltofirsttimepoin": "time",
            "phenotype": "disease",
        }
    )
    metadata_ca["status"] = "Unknown"

    ca_adata = ad.AnnData(
        X=counts_ca_tmm,
        obs=metadata_ca,
        var=pd.DataFrame(index=counts_ca.columns),
    )
    ca_adata.layers["raw"] = counts_ca

    return ca_adata


def import_bc():
    counts, counts_tmm, metadata = load_archs4("GSE201085")

    metadata["disease"] = "Breast Cancer"
    metadata = metadata.rename(columns={"response": "status"})

    bc_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    bc_adata.layers["raw"] = counts

    return bc_adata


def import_uc():
    counts, counts_tmm, metadata = load_archs4("GSE177044")

    metadata = metadata.loc[:, ["Sex", "age", "disease"]]
    metadata["disease"] = metadata["disease"].str.replace(
        r"\bControl\b", "Healthy", regex=True
    )
    metadata["disease"] = metadata["disease"].str.replace(
        r"\bUC\b", "Ulcerative Colitis", regex=True
    )
    metadata["disease"] = metadata["disease"].str.replace(
        r"\bPSC\b", "Primary Sclerosing Cholangitis", regex=True
    )
    metadata["disease"] = metadata["disease"].str.replace(
        r"\bPSCUC\b", "PSC/UC", regex=True
    )

    metadata["status"] = "NaN"

    uc_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    uc_adata.layers["raw"] = counts

    return uc_adata


def import_tb():
    counts, counts_tmm, metadata = load_archs4("GSE89403")

    metadata = metadata.loc[
        :, ["subject", "disease state", "treatmentresult", "time", "timetonegativity"]
    ]
    metadata["time"] = metadata["time"].str.replace("DX", "week_0")

    # Remove unknown samples.
    # Paper does not describe what these are but they are present in the counts
    # "NA", "Lung Dx Controls", "MTP Controls"
    valid_samples = metadata["disease state"].str.contains(
        "TB Subjects|Healthy Controls"
    )
    metadata = metadata.loc[valid_samples, :]

    """Optional sample filtering processes. Not sure if these are necessary yet."""
    # # Remove technical replicates
    # metadata = metadata[~metadata.duplicated(keep="first")]

    # # Relabel disease state based on time to negativity, removing unevaluable samples
    # metadata = metadata[~metadata["treatmentresult"].str.contains("unevaluable")]
    # metadata["timetonegativity"] = metadata["timetonegativity"].str.replace(
    #     "NA", "Week999"
    # )
    # metadata["time"] = metadata["time"].str.replace("day_7", "week_1")
    # sample_time = metadata["time"].str.split("_", expand=True)[1].astype(int)
    # negative_time = (
    #     metadata["timetonegativity"].str.split("k", expand=True)[1].astype(int)
    # )
    # metadata.loc[sample_time >= negative_time, "disease state"] = "TB Cured"
    # metadata.loc[sample_time < negative_time, "disease state"] = "Tuberculosis"

    # # Keep only first longitudinal measurement for each subject
    # metadata = metadata.loc[metadata["time"].str.contains("week_0"), :]

    # Line up the metadata with the counts to account for any filtering
    common_idx = counts.index.intersection(metadata.index)
    counts = counts.loc[common_idx]
    counts_tmm = counts_tmm.loc[common_idx]
    metadata = metadata.loc[common_idx]

    metadata = metadata.rename(
        columns={
            "subject": "subject_id",
            "disease state": "disease",
            "treatmentresult": "status",
        }
    )
    metadata["disease"] = metadata["disease"].str.replace("Healthy Controls", "Healthy")

    tb_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    tb_adata.layers["raw"] = counts

    return tb_adata


def import_t1dm():
    counts, counts_tmm, metadata = load_archs4("GSE124400")

    metadata = metadata.loc[
        :, ["subject", "age at enrollment", "visit day", "rate of c-peptide change"]
    ]
    metadata = metadata.rename(
        columns={
            "subject": "subject_id",
            "age at enrollment": "age",
            "visit day": "time",
        }
    )
    metadata["disease"] = "T1DM"
    metadata["status"] = "Unknown"

    # Use rate of c-peptide change to determine responder status (conservative)
    metadata.loc[metadata["rate of c-peptide change"].astype(float) < 0, "status"] = (
        "non-responder"
    )
    metadata.loc[metadata["rate of c-peptide change"].astype(float) >= 0, "status"] = (
        "responder"
    )
    metadata = metadata.drop(columns=["rate of c-peptide change"])

    t1dm_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    t1dm_adata.layers["raw"] = counts

    return t1dm_adata


def import_covid():
    counts, counts_tmm, metadata = load_archs4("GSE161731")

    metadata = metadata.loc[
        :, ["subject_id", "age", "gender", "cohort", "time", "hospitalized"]
    ]
    metadata = metadata.rename(
        columns={
            "cohort": "disease",
            "hospitalized": "status",
        }
    )
    metadata["disease"] = metadata["disease"].str.replace("healthy", "Healthy")

    covid_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    covid_adata.layers["raw"] = counts

    return covid_adata
