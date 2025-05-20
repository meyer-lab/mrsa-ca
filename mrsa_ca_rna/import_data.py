"""
Import data from various sources, compiled by ARCHS4 and format them
into AnnData objects.

For now, we keep track of the study numbers and the primary diseases represented.
We use the primary disease to label the dataset for the time being using
a disease registry. This is primarily to help a human reader associate imports.

In the future, we might create a more comprehensive disease registry to
better keep track of all diseases represented across all datasets.
"""

import contextlib
import json
import multiprocessing
from os.path import abspath, dirname, join

import anndata as ad
import archs4py.data as a4_data
import archs4py.meta as a4_meta
import archs4py.utils as a4_utils
import pandas as pd

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("spawn")  # loss of speed but avoids fork() issues

BASE_DIR = dirname(dirname(abspath(__file__)))


def parse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
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
    file_path = "/opt/extra-storage/jpopoli/human_gene_v2.6.h5"

    # Extract the count data from the ARCHS4 file, fail if not found
    counts = a4_data.series(file_path, geo_accession)
    if not isinstance(counts, pd.DataFrame):
        raise ValueError(
            f"Could not find GEO accession {geo_accession} in the file {file_path}"
        )
    counts = a4_utils.aggregate_duplicate_genes(counts)
    counts_tmm = a4_utils.normalize(counts=counts, method="tmm", tmm_outlier=0.05)

    # Extract the metadata from the ARCHS4 file after success with counts
    metadata = a4_meta.series(file_path, geo_accession)

    # Parse the metadata to extract the clinical variables
    clinical_variables = parse_metadata(metadata)

    return counts.T, counts_tmm.T, clinical_variables


def import_mrsa():
    # Read in mrsa counts
    counts_mrsa = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_mrsa_archs4.csv.gz"),
        index_col=0,
        delimiter=",",
    )
    counts_mrsa = a4_utils.aggregate_duplicate_genes(counts_mrsa)
    counts_mrsa_tmm = a4_utils.normalize(
        counts=counts_mrsa, method="tmm", tmm_outlier=0.05
    )
    counts_mrsa = counts_mrsa.T
    counts_mrsa_tmm = counts_mrsa_tmm.T

    # Grab mrsa metadata from SRA database since it is not on GEO
    metadata_ncbi = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa.csv"),
        index_col=0,
        delimiter=",",
    )

    # Pair down metadata and ensure "disease" and "status" columns are present
    metadata_ncbi = metadata_ncbi.loc[:, ["isolate"]]
    metadata_ncbi.index.name = None

    metadata_ncbi["disease"] = "MRSA"
    metadata_ncbi["dataset_id"] = "SRP414349"
    metadata_ncbi = metadata_ncbi.reset_index(
        drop=False, names=["accession"]
    ).set_index("isolate")

    """Cannot reconcile the NCBI uploaded metadata with the TFAC-MRSA metadata.
    Until we get more informationa on the NCBI metadata, we will use
    the TFAC-MRSA metadata, which agrees with Dr. Joshua Thaden's data."""

    # Combine the two parts of the tfac metadata for complete status
    metadata_tfac = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_mrsa_tfac.txt"),
        index_col=0,
        delimiter=",",
    )
    metadata_aux = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_tfac_validation.txt"),
        index_col=0,
        delimiter=",",
    )
    val_idx = metadata_aux.index
    metadata_tfac.loc[val_idx, "status"] = metadata_aux.loc[val_idx, "status"]

    # Combine the tfac metadata with the NCBI metadata to map the isolate numbers
    # to the SRA accessions
    metadata = pd.concat([metadata_ncbi, metadata_tfac], axis=1, join="inner")
    metadata = metadata.loc[
        :, ["accession", "dataset_id", "disease", "status", "cohort", "age", "gender"]
    ]
    metadata = metadata.reset_index(drop=False, names=["subject_id"]).set_index(
        "accession"
    )

    # Order the indices of the counts and metadata to match for AnnData
    common_idx = counts_mrsa.index.intersection(metadata.index)
    counts_mrsa = counts_mrsa.loc[common_idx]
    counts_mrsa_tmm = counts_mrsa_tmm.loc[common_idx]
    metadata = metadata.loc[common_idx]

    mrsa_adata = ad.AnnData(
        X=counts_mrsa_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts_mrsa.columns),
    )
    mrsa_adata.layers["raw"] = counts_mrsa

    mrsa_adata.obs["status"] = mrsa_adata.obs["status"].astype(int)

    return mrsa_adata


def import_ca():
    # Read in ca counts
    counts_ca = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_ca_archs4.tsv.gz"),
        index_col=0,
        delimiter="\t",
    )
    counts_ca = a4_utils.aggregate_duplicate_genes(counts_ca)
    counts_ca_tmm = a4_utils.normalize(counts=counts_ca, method="tmm", tmm_outlier=0.05)
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
    metadata_ca = parse_metadata(metadata_ca)  # includes qc failures

    # Pair down metadata and ensure "disease" and "status" columns are present
    metadata_ca = metadata_ca.loc[
        :,
        [
            "subject_id",
            "passed sample qc",
            "daysreltofirsttimepoin",
            "phenotype",
            "gender",
            "age",
        ],
    ]
    metadata_ca = metadata_ca.rename(
        columns={
            "daysreltofirsttimepoin": "time",
            "phenotype": "disease",
        }
    )
    metadata_ca["status"] = "Unknown"
    metadata_ca["dataset_id"] = "GSE176262"

    ca_adata = ad.AnnData(
        X=counts_ca_tmm,
        obs=metadata_ca,
        var=pd.DataFrame(index=counts_ca.columns),
    )
    ca_adata.layers["raw"] = counts_ca

    # Remove all non-Candidemia samples
    ca_adata = ca_adata[ca_adata.obs["disease"] == "Candidemia"].copy()

    return ca_adata


def import_bc():
    counts, counts_tmm, metadata = load_archs4("GSE201085")

    metadata["disease"] = "Breast Cancer"
    metadata = metadata.rename(columns={"response": "status"})
    metadata["dataset_id"] = "GSE201085"

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
    metadata["dataset_id"] = "GSE177044"

    uc_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    uc_adata.layers["raw"] = counts

    # Remove all non-UC samples
    uc_adata = uc_adata[uc_adata.obs["disease"] == "Ulcerative Colitis"].copy()

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
    metadata["dataset_id"] = "GSE89403"

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
    metadata["dataset_id"] = "GSE124400"

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
        :, ["subject_id", "age", "gender", "cohort", "time_since_onset", "hospitalized"]
    ]
    metadata = metadata.rename(
        columns={
            "cohort": "disease",
            "hospitalized": "status",
            "time_since_onset": "time",
        }
    )
    metadata["disease"] = metadata["disease"].str.replace("healthy", "Healthy")
    metadata["dataset_id"] = "GSE161731"

    covid_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    covid_adata.layers["raw"] = counts

    # Remove all non-COVID samples
    covid_adata = covid_adata[covid_adata.obs["disease"] == "COVID-19"].copy()

    return covid_adata


def import_lupus():
    counts, counts_tmm, metadata = load_archs4("GSE116006")

    metadata = metadata.loc[
        :,
        [
            "drug dose",
            "drug exposure",
            "ifn status",
            "subject age",
            "subject sex",
            "timepoint",
        ],
    ]
    metadata = metadata.rename(
        columns={
            "drug dose": "dose",
            "drug exposure": "status",
            "subject age": "age",
            "subject sex": "sex",
            "timepoint": "time",
        }
    )
    metadata["disease"] = "Lupus"
    metadata["dataset_id"] = "GSE116006"

    lupus_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    lupus_adata.layers["raw"] = counts

    return lupus_adata


def import_hiv():
    counts, counts_tmm, metadata = load_archs4("GSE162914")

    # Pair down metadata and ensure "disease" and "status" columns are present
    metadata = metadata.loc[
        :,
        [
            "patient id",
            "age",
            "baseline cd4+",
            "baseline hiv_rna",
            "outcome",
            "timing",
            "treatment-outcome code",
        ],
    ]
    metadata = metadata.rename(
        columns={
            "patient id": "subject_id",
            "baseline cd4+": "baseline_cd4+",
            "baseline hiv_rna": "baseline_hiv_rna",
            "outcome": "status",
            "timing": "time",
            "treatment-outcome code": "treatment_outcome_code",
        }
    )
    metadata["disease"] = "HIV_CM"
    metadata["dataset_id"] = "GSE162914"

    hiv_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    hiv_adata.layers["raw"] = counts

    return hiv_adata


def import_em():
    counts, counts_tmm, metadata = load_archs4("GSE133378")

    metadata = metadata.loc[:, ["infected with/healthy control"]]
    metadata["dataset_id"] = "GSE133378"
    metadata["status"] = "Unknown"

    metadata = metadata.rename(columns={"infected with/healthy control": "disease"})
    metadata["disease"] = metadata["disease"].str.replace("Control", "Healthy")

    em_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    em_adata.layers["raw"] = counts

    # Take only the Enterovirus and Healthy samples
    em_adata = em_adata[
        em_adata.obs["disease"].str.contains("Enterovirus|Healthy"), :
    ].copy()

    return em_adata


def import_zika():
    counts, counts_tmm, metadata = load_archs4("GSE129882")

    metadata = metadata.loc[
        :,
        [
            "Sex",
            "age",
            "exposure",
            "patient",
            "time",
        ],
    ]
    metadata = metadata.rename(
        columns={
            "exposure": "status",
            "patient": "subject_id",
        }
    )
    metadata["disease"] = "Zika"
    metadata["dataset_id"] = "GSE129882"

    zika_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    zika_adata.layers["raw"] = counts

    return zika_adata


def import_heme():
    counts, counts_tmm, metadata = load_archs4("GSE133758")

    metadata = metadata.loc[:, ["globin-block applied", "identifier"]]
    metadata = metadata.rename(
        columns={
            "globin-block applied": "status",
            "identifier": "subject_id",
        }
    )
    metadata["disease"] = "Healthy_heme"
    metadata["dataset_id"] = "GSE133758"
    heme_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    heme_adata.layers["raw"] = counts

    return heme_adata


def import_ra():
    counts, counts_tmm, metadata = load_archs4("GSE120178")

    metadata = metadata.rename(
        columns={"disease state": "disease", "timepoint": "time"}
    )
    metadata["disease"] = metadata["disease"].str.replace("rheumatoid arthritis", "RA")
    metadata["disease"] = metadata["disease"].str.replace("healthy", "Healthy")
    metadata["status"] = "Unknown"
    metadata["dataset_id"] = "GSE120178"

    ra_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    ra_adata.layers["raw"] = counts

    # Keep only the RA samples
    ra_adata = ra_adata[ra_adata.obs["disease"] == "RA"].copy()

    return ra_adata


def import_hbv():
    counts, counts_tmm, metadata = load_archs4("GSE173897")

    metadata = metadata.loc[:, ["ethnicity", "gender", "hbv status"]]
    metadata = metadata.rename(columns={"hbv status": "status"})
    metadata["disease"] = "HBV"
    metadata["dataset_id"] = "GSE173897"

    hbv_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    hbv_adata.layers["raw"] = counts

    return hbv_adata


def import_kidney():
    counts, counts_tmm, metadata = load_archs4("GSE112927")

    metadata = metadata.loc[:, ["death censored graft loss", "follow up days"]]
    metadata = metadata.rename(
        columns={
            "death censored graft loss": "status",
            "follow up days": "time",
        }
    )
    metadata["disease"] = "Kidney Transplant"
    metadata["dataset_id"] = "GSE112927"

    kidney_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    kidney_adata.layers["raw"] = counts

    return kidney_adata


def import_covid_marine():
    counts, counts_tmm, metadata = load_archs4("GSE198449")

    metadata = metadata.loc[
        :,
        [
            "Sex",
            "age",
            "race",
            "participant id",
            "pcr test for sars-cov-2",
            "sample collection time point (days since t0)",
            "symptom",
        ],
    ]
    metadata = metadata.rename(
        columns={
            "participant id": "subject_id",
            "sample collection time point (days since t0)": "time",
            "pcr test for sars-cov-2": "disease",
            "symptom": "status",
        }
    )

    metadata["dataset_id"] = "GSE198449"

    covid_m_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    covid_m_adata.layers["raw"] = counts

    # Remove all non-COVID or problem samples
    covid_m_adata = covid_m_adata[~covid_m_adata.obs["disease"].isna()]
    covid_m_adata = covid_m_adata[
        covid_m_adata.obs["disease"].str.contains("Not|Detected")
    ].copy()

    # Any recorded symptoms are considered symptomatic, otherwise asymptomatic
    covid_m_adata.obs.loc[covid_m_adata.obs["status"].notna(), ["status"]] = (
        "Symptomatic"
    )
    covid_m_adata.obs.loc[covid_m_adata.obs["status"].isna(), ["status"]] = (
        "Asymptomatic"
    )

    """Not yet sure if we will distinguish between healthy and diseased within a dataset
    of longitudinal samples. For now, we will just label all samples as diseased."""
    covid_m_adata.obs["disease"] = "COVID_marine"
    # # Disease is either "Not" or "Detected"
    # covid_m_adata.obs[
    # covid_m_adata.obs["disease"].str.contains("Not"), "disease"
    # ] = "COVID_m_neg"
    # covid_m_adata.obs[
    # covid_m_adata.obs["disease"].str.contains("Detected"), "disease"
    # ] = "COVID_m_pos"

    return covid_m_adata


def import_bc_tcr():
    # Read in breast cancer tcr counts
    counts = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_bc_archs4.tsv.gz"),
        index_col=0,
        delimiter="\t",
    )
    counts = a4_utils.aggregate_duplicate_genes(counts)
    counts_tmm = a4_utils.normalize(counts=counts, method="tmm", tmm_outlier=0.05)
    counts = counts.T
    counts_tmm = counts_tmm.T

    # Read in breast cancer metadata
    metadata = pd.read_json(
        join(BASE_DIR, "mrsa_ca_rna", "data", "metadata_bc_archs4.json")
    )

    # Transpose and rename the metadata to fit parser
    metadata = metadata.T
    metadata = metadata.rename(
        columns={
            "characteristics": "characteristics_ch1",
        }
    )
    metadata = parse_metadata(metadata)
    metadata = metadata.rename(
        columns={
            "tumor status": "status",
        }
    )
    metadata["disease"] = "Breast Cancer TCR"
    metadata["dataset_id"] = "GSE239933"

    bc_tcr_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    bc_tcr_adata.layers["raw"] = counts

    return bc_tcr_adata


def import_test():
    counts, counts_tmm, metadata = load_archs4("GSE239933")

    test_adata = ad.AnnData(
        X=counts_tmm,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    test_adata.layers["raw"] = counts
    test_adata.obs["disease"] = "Test"
    test_adata.obs["status"] = "Unknown"
    test_adata.obs["dataset_id"] = "GSExxxxxxx"

    return test_adata


if __name__ == "__main__":
    test_adata = import_test()


def build_disease_registry(save_path=None):
    """
    Build a registry mapping diseases to their source datasets.

    Parameters:
        save_path (str, optional): If provided, saves the registry as JSON

    Returns:
        dict: Dictionary mapping disease names to lists of dataset identifiers
    """
    import_functions = {
        "mrsa": import_mrsa,
        "ca": import_ca,
        "bc": import_bc,
        "tb": import_tb,
        "uc": import_uc,
        "t1dm": import_t1dm,
        "covid": import_covid,
        "lupus": import_lupus,
        "hiv": import_hiv,
        "em": import_em,
        "zika": import_zika,
        "heme": import_heme,
        "ra": import_ra,
        "hbv": import_hbv,
        "kidney": import_kidney,
        "covid_marine": import_covid_marine,
        "bc_tcr": import_bc_tcr,
    }

    registry = {}
    print("Building disease registry...")

    for dataset_id, import_func in import_functions.items():
        try:
            print(f"Processing dataset: {dataset_id}")
            adata = import_func()
            for disease in adata.obs["disease"].unique():
                if disease not in registry:
                    registry[disease] = []
                if dataset_id not in registry[disease]:
                    registry[disease].append(dataset_id)
        except Exception as e:
            print(f"Error processing {dataset_id}: {e}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"Registry saved to {save_path}")

    return registry
