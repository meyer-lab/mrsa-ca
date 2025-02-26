"""
This file will test the pf2 factorization methods in the factorization.py file
concat_datasets, prepare_data, and perform_parafac2

concat_datasets will concatenate the datasets and scale them
  check: type, shape, and that the data is scaled properly

prepare_data will prepare the data for the factorization
  check: type, shape, that the data has new idx and gene means
  and that the gene means are zero

perform_parafac2 will perform the factorization
  check: type, shape, that everything returned as expected
  and that the factors are finite, not nan, and not all zero
"""

import anndata as ad
import numpy as np
import numpy.testing as npt

from mrsa_ca_rna.factorization import perform_parafac2, prepare_data
from mrsa_ca_rna.utils import concat_datasets


def test_concat_datasets():
    concat_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    # Check type and shape
    assert isinstance(concat_data, ad.AnnData)
    assert concat_data.shape == (543, 16315)

    # Check that the data is demeaned and scaled
    X = np.array(concat_data.X)
    npt.assert_allclose(X.mean(axis=0), 0.0, atol=1e-5)
    npt.assert_allclose(X.std(axis=0), 1.0, rtol=1e-5)


def test_prepare_data():
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    prepared_data = prepare_data(disease_data, expansion_dim="disease")

    # Check that the data has new idx and gene means
    assert isinstance(prepared_data, ad.AnnData)
    assert "condition_unique_idxs" in prepared_data.obs
    assert "means" in prepared_data.var
    assert prepared_data.obs["condition_unique_idxs"].dtype == "category"
    assert prepared_data.var["means"].shape == (16315,)

    # Gene means should be zero if the data is scaled properly
    npt.assert_allclose(prepared_data.var["means"].mean(), 0.0, atol=1e-5)


def test_perform_parafac2():
    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    disease_data = concat_datasets(disease_list, scale=True, tpm=True)
    rank = 50
    l1 = 0.1
    factors, projections, r2x = perform_parafac2(
        disease_data, condition_name="disease", rank=rank, l1=l1
    )

    # Did everything return as expected?
    assert len(factors) == 3
    assert len(projections) == len(disease_list)
    assert r2x > 0.0
    assert r2x <= 1.0

    # Check the shapes of the factors
    for factor in factors:
        assert isinstance(factor, np.ndarray)
        assert factor.shape[1] == rank

    # Check contents of factors
    for factor in factors:
        assert np.isfinite(factor).all()
        assert not np.isnan(factor).any()
        assert (factor != 0).any()
