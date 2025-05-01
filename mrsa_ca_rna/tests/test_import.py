import os

import numpy as np
import pytest

from mrsa_ca_rna.import_data import (
    BASE_DIR,
    import_bc,
    import_ca,
    import_mrsa,
    import_t1dm,
    import_tb,
    import_uc,
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
    [import_mrsa, import_ca, import_bc, import_tb, import_uc, import_t1dm],
)
def test_import_functions(import_func, gene_list):
    """Test import functions for all datasets."""
    # Import the data
    adata = import_func()

    # Test var.index length
    assert (
        len(adata.var.index) == 62548
    ), f"Expected 62548 genes, got {len(adata.var.index)}"

    # Test var.index for duplicates
    assert len(adata.var.index) == len(
        set(adata.var.index)
    ), "var.index contains duplicates"

    # Test obs contains status column
    assert "status" in adata.obs.columns, "obs does not contain 'status' column"

    # Test obs contains disease column
    assert "disease" in adata.obs.columns, "obs does not contain 'disease' column"

    # Test tmm layer exists
    assert "tmm" in adata.layers, "adata.layers['tmm'] does not exist"

    # Test var.index matches gene_list.txt
    assert set(adata.var.index) == set(
        gene_list
    ), "var.index does not match gene_list.txt"

    # Test tmm layer has same shape as X
    assert adata.layers["tmm"].shape == adata.X.shape, "tmm layer shape doesn't match X"

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

    assert np.array_equal(
        mrsa_adata.var.index, ca_adata.var.index
    ), "MRSA and CA datasets have different genes"
    assert np.array_equal(
        mrsa_adata.var.index, bc_adata.var.index
    ), "MRSA and BC datasets have different genes"
    assert np.array_equal(
        mrsa_adata.var.index, tb_adata.var.index
    ), "MRSA and TB datasets have different genes"
    assert np.array_equal(
        mrsa_adata.var.index, uc_adata.var.index
    ), "MRSA and UC datasets have different genes"
    assert np.array_equal(
        mrsa_adata.var.index, t1dm_adata.var.index
    ), "MRSA and T1DM datasets have different genes"
