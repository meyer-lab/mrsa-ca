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


def series_local(file, series_id) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pyright does not understand that h5py.File supports dict-like access
    f: h5.File = h5.File(file, "r")

    # find samples that correspond to a series
    series = [x.decode("UTF-8") for x in np.array(f["meta/samples/series_id"])]
    sample_idx = [i for i, x in enumerate(series) if x == series_id]
    assert len(sample_idx) > 0

    # find gene names
    genes = np.array([x.decode("UTF-8") for x in np.array(f["meta/genes/symbol"])])
    gsm_ids = np.array(
        [x.decode("UTF-8") for x in np.array(f["meta/samples/geo_accession"])]
    )[sample_idx]

    # get expression counts
    exp = np.array(f["data/expression"][:, sample_idx], dtype=np.uint32)  # type: ignore

    exp = pd.DataFrame(exp, index=genes, columns=gsm_ids, dtype=np.uint32)

    # Extract metadata from the file
    meta_fields = [
        "geo_accession",
        "series_id",
        "characteristics_ch1",
        "extract_protocol_ch1",
        "source_name_ch1",
        "title",
    ]

    dG = f["meta"]["samples"]  # type: ignore

    meta_series = np.array([x.decode("UTF-8") for x in list(np.array(dG["series_id"]))])  # type: ignore
    idx = [i for i, x in enumerate(meta_series) if x == series_id]

    meta = []
    mfields = []

    for field in meta_fields:
        meta.append([x.decode("UTF-8") for x in list(np.array(dG[field][idx]))])  # type: ignore
        mfields.append(field)

    meta = pd.DataFrame(
        meta,
        index=pd.Index(mfields),
        columns=pd.Index(
            [x.decode("UTF-8") for x in list(np.array(dG["geo_accession"][idx]))]  # type: ignore
        ),
    )

    f.close()

    return exp, meta.T


def load_archs4(geo_accession: str) -> ad.AnnData:
    file_path = "/opt/extra-storage/jpopoli/human_gene_v2.6.h5"

    # Extract the count data from the ARCHS4 file, fail if not found
    counts, metadata = series_local(file_path, geo_accession)
    if not isinstance(counts, pd.DataFrame):
        raise ValueError(
            f"Could not find GEO accession {geo_accession} in the file {file_path}"
        )

    # Aggregate duplicate genes due to Ensembl -> Symbol conversion
    counts = aggregate_duplicate_genes(counts)
    counts = counts.T

    # Parse the metadata to extract the clinical variables
    clinical_variables = parse_metadata(metadata)

    adata = ad.AnnData(
        X=counts,
        obs=clinical_variables,
        var=pd.DataFrame(index=counts.columns),
    )

    return adata


def aggregate_duplicate_genes(exp):
    return exp.groupby(exp.index).sum()


def import_mrsa():
    # Read in mrsa counts
    counts_mrsa = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_mrsa_archs4.csv.gz"),
        index_col=0,
        delimiter=",",
    )
    counts_mrsa = aggregate_duplicate_genes(counts_mrsa)
    counts_mrsa = counts_mrsa.T

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
    metadata = metadata.loc[common_idx]

    mrsa_adata = ad.AnnData(
        X=counts_mrsa,
        obs=metadata,
        var=pd.DataFrame(index=counts_mrsa.columns),
    )
    mrsa_adata.obs["status"] = mrsa_adata.obs["status"].astype(int)

    return mrsa_adata


def import_ca():
    # Read in ca counts
    counts_ca = pd.read_csv(
        join(BASE_DIR, "mrsa_ca_rna", "data", "counts_ca_archs4.tsv.gz"),
        index_col=0,
        delimiter="\t",
    )
    counts_ca = aggregate_duplicate_genes(counts_ca)
    counts_ca = counts_ca.T

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
        X=counts_ca,
        obs=metadata_ca,
        var=pd.DataFrame(index=counts_ca.columns),
    )

    # Remove all non-Candidemia samples
    ca_adata = ca_adata[ca_adata.obs["disease"] == "Candidemia"].copy()

    return ca_adata


def import_bc():
    bc_adata = load_archs4("GSE201085")
    bc_adata.obs["disease"] = "Breast Cancer"
    return bc_adata


def import_uc():
    uc_adata = load_archs4("GSE177044")

    uc_adata.obs = uc_adata.obs.loc[:, ["Sex", "age", "disease"]]
    uc_adata.obs["disease"] = uc_adata.obs["disease"].str.replace(
        r"\bControl\b", "Healthy", regex=True
    )
    uc_adata.obs["disease"] = uc_adata.obs["disease"].str.replace(
        r"\bUC\b", "Ulcerative Colitis", regex=True
    )
    uc_adata.obs["disease"] = uc_adata.obs["disease"].str.replace(
        r"\bPSC\b", "Primary Sclerosing Cholangitis", regex=True
    )
    uc_adata.obs["disease"] = uc_adata.obs["disease"].str.replace(
        r"\bPSCUC\b", "PSC/UC", regex=True
    )

    # Add standardized columns
    uc_adata.obs["status"] = "Unknown"
    uc_adata.obs["dataset_id"] = "GSE177044"

    # Remove all non-UC samples
    uc_adata = uc_adata[uc_adata.obs["disease"] == "Ulcerative Colitis"].copy()

    return uc_adata


def import_tb():
    tb_adata = load_archs4("GSE89403")

    tb_adata.obs = tb_adata.obs.loc[
        :, ["subject", "disease state", "treatmentresult", "time", "timetonegativity"]
    ]
    tb_adata.obs["time"] = tb_adata.obs["time"].str.replace("DX", "week_0")

    # Remove unknown samples.
    # Paper does not describe what these are but they are present in the counts
    # "NA", "Lung Dx Controls", "MTP Controls"
    valid_samples = tb_adata.obs["disease state"].str.contains(
        "TB Subjects|Healthy Controls"
    )
    tb_adata = tb_adata[valid_samples].copy()

    """Optional sample filtering processes. Not sure if these are necessary yet."""
    # # Remove technical replicates
    # tb_adata.obs = tb_adata.obs[~tb_adata.obs.duplicated(keep="first")]

    # # Relabel disease state based on time to negativity, removing unevaluable samples
    # tb_adata.obs = tb_adata.obs[
    #     ~tb_adata.obs["treatmentresult"].str.contains("unevaluable")
    # ]
    # tb_adata.obs["timetonegativity"] = tb_adata.obs["timetonegativity"].str.replace(
    #     "NA", "Week999"
    # )
    # tb_adata.obs["time"] = tb_adata.obs["time"].str.replace("day_7", "week_1")
    # sample_time = tb_adata.obs["time"].str.split("_", expand=True)[1].astype(int)
    # negative_time = (
    #     tb_adata.obs["timetonegativity"].str.split("k", expand=True)[1].astype(int)
    # )
    # tb_adata.obs.loc[sample_time >= negative_time, "disease state"] = "TB Cured"
    # tb_adata.obs.loc[sample_time < negative_time, "disease state"] = "Tuberculosis"

    # # Keep only first longitudinal measurement for each subject
    # tb_adata.obs = tb_adata.obs.loc[tb_adata.obs["time"].str.contains("week_0"), :]

    tb_adata.obs = tb_adata.obs.rename(
        columns={
            "subject": "subject_id",
            "disease state": "disease",
            "treatmentresult": "status",
        }
    )
    tb_adata.obs["disease"] = tb_adata.obs["disease"].str.replace(
        "Healthy Controls", "Healthy"
    )
    tb_adata.obs["dataset_id"] = "GSE89403"

    return tb_adata


def import_t1dm():
    t1dm_adata = load_archs4("GSE124400")

    t1dm_adata.obs = t1dm_adata.obs.loc[
        :, ["subject", "age at enrollment", "visit day", "rate of c-peptide change"]
    ]
    t1dm_adata.obs = t1dm_adata.obs.rename(
        columns={
            "subject": "subject_id",
            "age at enrollment": "age",
            "visit day": "time",
        }
    )
    t1dm_adata.obs["disease"] = "T1DM"
    t1dm_adata.obs["status"] = "Unknown"
    t1dm_adata.obs["dataset_id"] = "GSE124400"

    # Use rate of c-peptide change to determine responder status (conservative)
    t1dm_adata.obs.loc[
        t1dm_adata.obs["rate of c-peptide change"].astype(float) < 0, "status"
    ] = "non-responder"
    t1dm_adata.obs.loc[
        t1dm_adata.obs["rate of c-peptide change"].astype(float) >= 0, "status"
    ] = "responder"
    t1dm_adata.obs = t1dm_adata.obs.drop(columns=["rate of c-peptide change"])

    return t1dm_adata


def import_covid():
    covid_adata = load_archs4("GSE161731")

    covid_adata.obs = covid_adata.obs.loc[
        :, ["subject_id", "age", "gender", "cohort", "time_since_onset", "hospitalized"]
    ]
    covid_adata.obs = covid_adata.obs.rename(
        columns={
            "cohort": "disease",
            "hospitalized": "status",
            "time_since_onset": "time",
        }
    )
    covid_adata.obs["disease"] = covid_adata.obs["disease"].str.replace(
        "healthy", "Healthy"
    )
    covid_adata.obs["dataset_id"] = "GSE161731"

    # Remove all non-COVID samples
    covid_adata = covid_adata[covid_adata.obs["disease"] == "COVID-19"].copy()

    return covid_adata


def import_lupus():
    lupus_adata = load_archs4("GSE116006")

    lupus_adata.obs = lupus_adata.obs.loc[
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
    lupus_adata.obs = lupus_adata.obs.rename(
        columns={
            "drug dose": "dose",
            "drug exposure": "status",
            "subject age": "age",
            "subject sex": "sex",
            "timepoint": "time",
        }
    )
    lupus_adata.obs["disease"] = "Lupus"
    lupus_adata.obs["dataset_id"] = "GSE116006"

    return lupus_adata


def import_hiv():
    hiv_adata = load_archs4("GSE162914")

    # Pair down metadata and ensure "disease" and "status" columns are present
    hiv_adata.obs = hiv_adata.obs.loc[
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
    hiv_adata.obs = hiv_adata.obs.rename(
        columns={
            "patient id": "subject_id",
            "baseline cd4+": "baseline_cd4+",
            "baseline hiv_rna": "baseline_hiv_rna",
            "outcome": "status",
            "timing": "time",
            "treatment-outcome code": "treatment_outcome_code",
        }
    )
    hiv_adata.obs["disease"] = "HIV_CM"
    hiv_adata.obs["dataset_id"] = "GSE162914"

    return hiv_adata


def import_em():
    em_adata = load_archs4("GSE133378")

    em_adata.obs = em_adata.obs.loc[:, ["infected with/healthy control"]]
    em_adata.obs["dataset_id"] = "GSE133378"
    em_adata.obs["status"] = "Unknown"

    em_adata.obs = em_adata.obs.rename(
        columns={"infected with/healthy control": "disease"}
    )
    em_adata.obs["disease"] = em_adata.obs["disease"].str.replace("Control", "Healthy")

    # Take only the Enterovirus and Healthy samples
    em_adata = em_adata[
        em_adata.obs["disease"].str.contains("Enterovirus|Healthy"), :
    ].copy()

    return em_adata


def import_zika():
    zika_adata = load_archs4("GSE129882")

    zika_adata.obs = zika_adata.obs.loc[
        :,
        [
            "Sex",
            "age",
            "exposure",
            "patient",
            "time",
        ],
    ]
    zika_adata.obs = zika_adata.obs.rename(
        columns={
            "exposure": "status",
            "patient": "subject_id",
        }
    )
    zika_adata.obs["disease"] = "Zika"
    zika_adata.obs["dataset_id"] = "GSE129882"

    return zika_adata


def import_heme():
    heme_adata = load_archs4("GSE133758")

    heme_adata.obs = heme_adata.obs.loc[:, ["globin-block applied", "identifier"]]
    heme_adata.obs = heme_adata.obs.rename(
        columns={
            "globin-block applied": "status",
            "identifier": "subject_id",
        }
    )
    heme_adata.obs["disease"] = "Healthy_heme"
    heme_adata.obs["dataset_id"] = "GSE133758"

    return heme_adata


def import_ra():
    ra_adata = load_archs4("GSE120178")

    ra_adata.obs = ra_adata.obs.rename(
        columns={"disease state": "disease", "timepoint": "time"}
    )
    ra_adata.obs["disease"] = ra_adata.obs["disease"].str.replace(
        "rheumatoid arthritis", "RA"
    )
    ra_adata.obs["disease"] = ra_adata.obs["disease"].str.replace("healthy", "Healthy")
    ra_adata.obs["status"] = "Unknown"
    ra_adata.obs["dataset_id"] = "GSE120178"

    # Keep only the RA samples
    ra_adata = ra_adata[ra_adata.obs["disease"] == "RA"].copy()

    return ra_adata


def import_hbv():
    hbv_adata = load_archs4("GSE173897")

    hbv_adata.obs = hbv_adata.obs.loc[:, ["ethnicity", "gender", "hbv status"]]
    hbv_adata.obs = hbv_adata.obs.rename(columns={"hbv status": "status"})
    hbv_adata.obs["disease"] = "HBV"
    hbv_adata.obs["dataset_id"] = "GSE173897"

    return hbv_adata


def import_kidney():
    kidney_adata = load_archs4("GSE112927")

    kidney_adata.obs = kidney_adata.obs.loc[
        :, ["death censored graft loss", "follow up days"]
    ]
    kidney_adata.obs = kidney_adata.obs.rename(
        columns={
            "death censored graft loss": "status",
            "follow up days": "time",
        }
    )
    kidney_adata.obs["disease"] = "Kidney Transplant"
    kidney_adata.obs["dataset_id"] = "GSE112927"

    return kidney_adata


def import_covid_marine():
    covid_m_adata = load_archs4("GSE198449")

    covid_m_adata.obs = covid_m_adata.obs.loc[
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
    covid_m_adata.obs = covid_m_adata.obs.rename(
        columns={
            "participant id": "subject_id",
            "sample collection time point (days since t0)": "time",
            "pcr test for sars-cov-2": "disease",
            "symptom": "status",
        }
    )

    covid_m_adata.obs["dataset_id"] = "GSE198449"

    # Remove all non-COVID or problem samples
    covid_m_adata = covid_m_adata[~covid_m_adata.obs["disease"].isna()]  # type: ignore
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
    counts = aggregate_duplicate_genes(counts)
    counts = counts.T

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
        X=counts,
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )

    return bc_tcr_adata
