import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mrsa_ca_rna.import_data import load_expression_data
from mrsa_ca_rna.utils import (
    calculate_cpm,
    gene_filter,
    normalize_counts,
    prepare_data,
)


@pytest.fixture
def sample_adata():
    """Fixture providing a sample AnnData object from prepare_data."""
    return prepare_data()


@pytest.fixture
def raw_adata():
    """Fixture providing raw expression data."""
    return load_expression_data()


@pytest.fixture
def sample_counts():
    """Fixture providing sample count data for testing."""
    return np.array([[100, 200, 300], [50, 150, 800], [0, 0, 1000]])


@pytest.fixture
def sample_gene_data():
    """Fixture providing sample gene data for filtering tests."""
    return pd.DataFrame(
        {
            "gene1": [1000, 2000, 3000, 0],  # High expression, should pass
            "gene2": [1, 2, 3, 1],  # Low expression, should fail
            "gene3": [0, 0, 5000, 0],  # Sparse high expression
            "gene4": [100, 200, 300, 400],  # Medium expression
        }
    )


# Tests for prepare_data function
def test_prepare_data_basic_functionality(sample_adata):
    """Test basic functionality of prepare_data."""
    adata = sample_adata

    # Test AnnData structure and content
    assert isinstance(adata, ad.AnnData), "prepare_data should return AnnData object"
    assert hasattr(adata, "X"), "AnnData object missing X matrix"
    assert hasattr(adata, "obs"), "AnnData object missing obs DataFrame"
    assert hasattr(adata, "var"), "AnnData object missing var DataFrame"
    assert hasattr(adata, "layers"), "AnnData object missing layers"

    # Test data is not empty
    assert adata.n_obs > 0, "Dataset has no observations"
    assert adata.n_vars > 0, "Dataset has no variables"

    # Test metadata columns exist
    assert "sample_id" in adata.obs.columns, "obs missing 'sample_id' column"
    assert "disease" in adata.obs.columns, "obs missing 'disease' column"
    assert "gene_id" in adata.var.columns, "var missing 'gene_id' column"


def test_prepare_data_raw_layer_and_normalization(sample_adata):
    """Test that raw counts are preserved and main matrix is z-scored."""
    adata = sample_adata

    # Test raw layer exists and properties
    assert "raw" in adata.layers, "Raw layer not found in AnnData object"
    assert adata.layers["raw"].shape == adata.X.shape, (
        "Raw layer shape doesn't match main matrix shape"
    )
    assert np.all(adata.layers["raw"] >= 0), "Raw layer contains negative values"

    # Test main matrix has been z-scored (should contain negative values)
    assert np.any(adata.X < 0), (
        "Main matrix doesn't appear to be z-scored (no negative values)"
    )

    # Test z-score properties: mean should be close to 0
    gene_means = np.mean(adata.X, axis=0)
    gene_stds = np.std(adata.X, axis=0, ddof=0)  # Use ddof=0 like StandardScaler

    assert np.allclose(gene_means, 0, atol=1e-10), (
        f"Gene means not close to 0: range "
        f"[{np.min(gene_means):.2e}, {np.max(gene_means):.2e}]"
    )

    # For genes with non-zero variance, std should be close to 1
    nonzero_var_mask = gene_stds > 1e-10
    if np.any(nonzero_var_mask):
        nonzero_stds = gene_stds[nonzero_var_mask]
        assert np.allclose(nonzero_stds, 1, atol=1e-5), (
            "Non-zero variance genes should have std~1"
        )


@pytest.mark.parametrize(
    "filter_threshold,min_pct",
    [
        (1.0, 0.25),  # Default filtering
        (5.0, 0.5),  # Strict filtering
        (-1, 0.0),  # No filtering
    ],
)
def test_prepare_data_filtering(filter_threshold, min_pct, raw_adata):
    """Test prepare_data with different filtering parameters."""
    adata = prepare_data(filter_threshold=filter_threshold, min_pct=min_pct)

    # Basic checks that should pass for all parameter combinations
    assert adata.n_obs > 0, "Should have observations"
    assert adata.n_vars > 0, "Should have variables"
    assert "raw" in adata.layers, "Should have raw layer"

    # Test filtering behavior
    if filter_threshold == -1:
        # No filtering - should have same number of genes as raw data
        assert adata.n_vars == raw_adata.n_vars, (
            "Unfiltered data has different number of genes than raw data"
        )
    else:
        # Filtering enabled - should have fewer or equal genes
        assert adata.n_vars <= raw_adata.n_vars, (
            "Filtered data has more genes than raw data"
        )


def test_prepare_data_data_integrity(sample_adata):
    """Test data types, shapes, and value integrity."""
    adata = sample_adata

    # Test data types
    assert isinstance(adata.X, np.ndarray), "Main matrix is not numpy array"
    assert isinstance(adata.layers["raw"], np.ndarray), "Raw layer is not numpy array"
    assert adata.X.dtype == np.float64, (
        f"Main matrix dtype is {adata.X.dtype}, expected float64"
    )
    assert adata.layers["raw"].dtype in [np.float64, np.int64, np.float32], (
        f"Raw layer dtype is {adata.layers['raw'].dtype}"
    )

    # Test no NaN or infinite values
    assert not np.any(np.isnan(adata.X)), "Main matrix contains NaN values"
    assert not np.any(np.isinf(adata.X)), "Main matrix contains infinite values"
    assert not np.any(np.isnan(adata.layers["raw"])), "Raw layer contains NaN values"
    assert not np.any(np.isinf(adata.layers["raw"])), (
        "Raw layer contains infinite values"
    )


def test_prepare_data_disease_labels(sample_adata):
    """Test that disease labels are preserved correctly."""
    adata = sample_adata

    # Test disease labels exist and are strings
    assert "disease" in adata.obs.columns, "Disease column missing"
    assert adata.obs["disease"].dtype == object, "Disease labels are not strings"

    # Test for expected disease categories (from disease_registry)
    expected_diseases = {
        "MRSA",
        "CANDIDA",
        "BREAST_CANCER",
        "UC_PSC",
        "TB",
        "T1DM",
        "COVID",
        "LUPUS",
        "HIV_CM",
        "ENTEROVIRUS",
        "ZIKA",
        "HEALTHY",
        "RA",
        "HBV",
        "KIDNEY",
        "COVID_MARINES",
        "BREAST_CANCER_TCR",
        "SEPSIS",
        "LEUKEMIA",
        "COVID_SINAI",
        "ASTHEMA",
        "AETHERSCLEROSIS",
    }

    observed_diseases = set(adata.obs["disease"].unique())
    assert observed_diseases.issubset(expected_diseases), (
        f"Unexpected diseases found: {observed_diseases - expected_diseases}"
    )


def test_prepare_data_reproducibility():
    """Test that prepare_data produces consistent results."""
    adata1 = prepare_data(filter_threshold=1.0, min_pct=0.25)
    adata2 = prepare_data(filter_threshold=1.0, min_pct=0.25)

    # Test reproducibility
    assert adata1.shape == adata2.shape, "Inconsistent shapes between runs"
    assert np.allclose(np.asarray(adata1.X), np.asarray(adata2.X)), (
        "Inconsistent main matrix between runs"
    )
    assert np.allclose(
        np.asarray(adata1.layers["raw"]), np.asarray(adata2.layers["raw"])
    ), "Inconsistent raw layer between runs"
    assert adata1.obs.equals(adata2.obs), "Inconsistent observations between runs"
    assert adata1.var.equals(adata2.var), "Inconsistent variables between runs"


# Tests for load_expression_data function
def test_load_expression_data_basic(raw_adata):
    """Test basic functionality of load_expression_data."""
    adata = raw_adata

    # Test basic structure
    assert isinstance(adata, ad.AnnData), (
        "load_expression_data should return AnnData object"
    )
    assert hasattr(adata, "X"), "AnnData object missing X matrix"
    assert hasattr(adata, "obs"), "AnnData object missing obs DataFrame"
    assert hasattr(adata, "var"), "AnnData object missing var DataFrame"

    # Test data integrity
    assert adata.n_obs > 0, "Dataset has no observations"
    assert adata.n_vars > 0, "Dataset has no variables"
    assert not np.any(np.isnan(np.asarray(adata.X))), "Raw data contains NaN values"

    # Test expected columns
    assert "sample_id" in adata.obs.columns, "obs missing 'sample_id' column"
    assert "disease" in adata.obs.columns, "obs missing 'disease' column"
    assert "gene_id" in adata.var.columns, "var missing 'gene_id' column"


# Tests for utility functions
def test_calculate_cpm(sample_counts):
    """Test CPM calculation function."""
    cpm = calculate_cpm(sample_counts)

    # Test CPM properties
    assert cpm.shape == sample_counts.shape, "CPM shape doesn't match input"
    assert cpm.dtype == np.float64, "CPM should be float64"

    # Test CPM calculation manually for first sample
    total_first = np.sum(sample_counts[0, :])
    expected_first_cpm = (sample_counts[0, :] / total_first) * 1e6
    assert np.allclose(cpm[0, :], expected_first_cpm), (
        "CPM calculation incorrect for first sample"
    )


def test_gene_filter(sample_gene_data):
    """Test gene filtering function."""
    filtered_data = gene_filter(sample_gene_data, threshold=1.0, min_pct=0.5)

    # Test filtering results
    assert isinstance(filtered_data, pd.DataFrame), "Output should be DataFrame"
    assert filtered_data.shape[0] == sample_gene_data.shape[0], (
        "Number of samples changed"
    )
    assert filtered_data.shape[1] <= sample_gene_data.shape[1], (
        "Filtering should not increase genes"
    )


def test_gene_filter_specific_behavior():
    """Test gene filtering with specific test data."""
    test_data = pd.DataFrame(
        {
            "high_gene": [10000, 20000, 30000],  # Should pass filter (high CPM)
            "low_gene": [1, 2, 3],  # Should not pass filter (low CPM)
        }
    )

    filtered = gene_filter(test_data, threshold=100.0, min_pct=0.5)

    # Should keep high_gene, remove low_gene
    assert "high_gene" in filtered.columns, "High expression gene should be kept"
    assert "low_gene" not in filtered.columns, "Low expression gene should be removed"


def test_normalize_counts(sample_counts):
    """Test count normalization function."""
    # Convert to float for normalization
    test_counts = sample_counts.astype(np.float64)
    normalized = normalize_counts(test_counts)

    # Test normalization properties
    assert normalized.shape == test_counts.shape, "Shape changed during normalization"
    assert normalized.dtype == np.float64, "Output should be float64"
    assert not np.any(np.isnan(normalized)), "Should not contain NaN values"
    assert not np.any(np.isinf(normalized)), "Should not contain infinite values"

    # Test z-score properties (use ddof=0 like StandardScaler)
    gene_means = np.mean(normalized, axis=0)
    gene_stds = np.std(normalized, axis=0, ddof=0)

    assert np.allclose(gene_means, 0, atol=1e-10), "Normalized data not centered"

    # Check which features had variance in the original log-transformed data
    cpm = calculate_cpm(test_counts)
    log_cpm = np.log2(cpm + 1)
    original_stds = np.std(log_cpm, axis=0, ddof=0)

    # Features with original variance should have std=1, zero variance should have std=0
    for i, orig_std in enumerate(original_stds):
        if orig_std > 1e-10:  # Non-zero variance
            assert np.isclose(gene_stds[i], 1.0, atol=1e-10), (
                f"Gene {i} with variance should have std=1, got {gene_stds[i]}"
            )
        else:  # Zero variance
            assert np.isclose(gene_stds[i], 0.0, atol=1e-10), (
                f"Gene {i} with zero variance should have std=0, got {gene_stds[i]}"
            )


def test_normalize_counts_zero_variance():
    """Test normalize_counts with zero variance features."""
    # Create data with a zero variance feature (all same values)
    counts = np.array(
        [
            [100, 5, 300],  # Gene 1: variable, Gene 2: constant, Gene 3: variable
            [200, 5, 600],
            [150, 5, 450],
            [300, 5, 900],
        ],
        dtype=np.float64,
    )

    normalized = normalize_counts(counts)

    # Should handle zero variance gracefully
    assert normalized.shape == counts.shape, "Shape should be preserved"
    assert not np.any(np.isnan(normalized)), "Should not contain NaN values"
    assert not np.any(np.isinf(normalized)), "Should not contain infinite values"
