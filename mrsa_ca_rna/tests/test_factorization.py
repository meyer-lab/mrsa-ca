"""This file will test the pf2 factorization methods in the factorization.py file"""

import anndata as ad
import numpy as np
import numpy.testing as npt
import xarray as xr

from mrsa_ca_rna.factorization import normalize_factors, perform_parafac2, prepare_data
from mrsa_ca_rna.utils import concat_datasets


def test_concat_datasets():
    concat_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    assert isinstance(concat_data, ad.AnnData)
    assert concat_data.shape == (543, 16315)

    X = np.array(concat_data.X)
    npt.assert_allclose(X.mean(axis=0), 0.0, atol=1e-5)
    npt.assert_allclose(X.std(axis=0), 1.0, rtol=1e-5)


def test_prepare_data():
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    disease_xr = prepare_data(disease_data, expansion_dim="disease")
    assert isinstance(disease_xr, xr.Dataset)
    assert disease_xr.sizes == {
        "sample_MRSA": 88,
        "gene": 16315,
        "sample_Candidemia": 104,
        "sample_BreastCancer": 53,
        "sample_COVID-19": 55,
        "sample_Healthy": 243,
    }


def test_perform_parafac2():
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    disease_xr = prepare_data(disease_data, expansion_dim="disease")
    tensor_decomp, _, _ = perform_parafac2(disease_xr, rank=10)
    _, factors, _ = tensor_decomp

    assert isinstance(factors, list)
    assert len(factors) == 3
    assert factors[0].shape == (5, 10)
    # assert factors[1].shape == (10, 10)
    assert factors[2].shape == (16315, 10)


def test_normalize_factors():
    disease_data = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    disease_xr = prepare_data(disease_data, expansion_dim="disease")
    tensor_decomp, _, _ = perform_parafac2(disease_xr, rank=10)
    _, factors, _ = tensor_decomp
    normalized_factors, _ = normalize_factors([factors[0], factors[2]])

    for factor in normalized_factors:
        npt.assert_allclose(np.linalg.norm(factor, axis=0), 1.0, rtol=1e-5)
