import os

import numpy as np
import pytest

from mrsa_ca_rna.import_data import (
    BASE_DIR,
    import_bc,
    import_bc_tcr,
    import_ca,
    import_covid,
    import_covid_marine,
    import_em,
    import_hbv,
    import_heme,
    import_hiv,
    import_kidney,
    import_lupus,
    import_mrsa,
    import_ra,
    import_t1dm,
    import_tb,
    import_uc,
    import_zika,
    import_sepsis,
)

# Path to gene list file
GENE_LIST_PATH = os.path.join(BASE_DIR, "mrsa_ca_rna", "data", "gene_list.txt")


# Load gene list for comparison
@pytest.fixture
def gene_list():
    with open(GENE_LIST_PATH) as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


@pytest.mark.parametrize(
    "import_func",
    [
        import_mrsa,
        import_ca,
        import_bc,
        import_tb,
        import_uc,
        import_t1dm,
        import_covid,
        import_lupus,
        import_hiv,
        import_em,
        import_zika,
        import_heme,
        import_ra,
        import_hbv,
        import_kidney,
        import_covid_marine,
        import_bc_tcr,
        import_sepsis,
    ],
)
def test_import_functions(import_func, gene_list):
    """Test import functions for all datasets."""
    # Import the data
    adata = import_func()

    # Test var.index length
    assert len(adata.var.index) == 62548, (
        f"Expected 62548 genes, got {len(adata.var.index)}"
    )

    # Test var.index for duplicates
    assert len(adata.var.index) == len(set(adata.var.index)), (
        "var.index contains duplicates"
    )

    # Test obs contains status column
    assert "status" in adata.obs.columns, "obs does not contain 'status' column"

    # Test obs contains disease column
    assert "disease" in adata.obs.columns, "obs does not contain 'disease' column"

    # Test obs contains dataset_id column
    assert "dataset_id" in adata.obs.columns, "obs does not contain 'dataset_id' column"

    # Test var.index matches gene_list.txt
    assert set(adata.var.index) == set(gene_list), (
        "var.index does not match gene_list.txt"
    )

    # Test data is not empty
    assert adata.n_obs > 0, "Dataset has no observations"
    assert adata.n_vars > 0, "Dataset has no variables"

    # Test no NaN values in status column
    assert not adata.obs["status"].isna().any(), "NaN values found in status column"


def test_all_datasets_gene_compatibility():
    """Test that all datasets have the same genes in the same order."""
    mrsa_adata = import_mrsa()
    ca_adata = import_ca()
    bc_adata = import_bc()
    tb_adata = import_tb()
    uc_adata = import_uc()
    t1dm_adata = import_t1dm()
    covid_adata = import_covid()
    lupus_adata = import_lupus()
    hiv_adata = import_hiv()
    em_adata = import_em()
    zika_adata = import_zika()
    heme_adata = import_heme()
    ra_adata = import_ra()
    hbv_adata = import_hbv()
    kidney_adata = import_kidney()
    covid_marine_adata = import_covid_marine()
    bc_tcr_adata = import_bc_tcr()
    sepsis_adata = import_sepsis()

    assert np.array_equal(mrsa_adata.var.index, ca_adata.var.index), (
        "MRSA and CA datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, bc_adata.var.index), (
        "MRSA and BC datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, tb_adata.var.index), (
        "MRSA and TB datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, uc_adata.var.index), (
        "MRSA and UC datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, t1dm_adata.var.index), (
        "MRSA and T1DM datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, covid_adata.var.index), (
        "MRSA and COVID datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, lupus_adata.var.index), (
        "MRSA and LUPUS datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, hiv_adata.var.index), (
        "MRSA and HIV datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, em_adata.var.index), (
        "MRSA and EM datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, zika_adata.var.index), (
        "MRSA and ZIKA datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, heme_adata.var.index), (
        "MRSA and HEME datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, ra_adata.var.index), (
        "MRSA and RA datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, hbv_adata.var.index), (
        "MRSA and HBV datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, kidney_adata.var.index), (
        "MRSA and KIDNEY datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, covid_marine_adata.var.index), (
        "MRSA and COVID MARINE datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, bc_tcr_adata.var.index), (
        "MRSA and BC TCR datasets have different genes"
    )
    assert np.array_equal(mrsa_adata.var.index, sepsis_adata.var.index), (
        "MRSA and SEPSIS datasets have different genes"
    )
