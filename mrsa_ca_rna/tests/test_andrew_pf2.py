"""This file will test the new pf2 implementation from Andrew's repo"""

import anndata as ad
import numpy as np
import numpy.testing as npt

from mrsa_ca_rna.factorization import new_parafac2
from mrsa_ca_rna.utils import concat_datasets


def test_new_parafac2():

    X = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )
    condition_name = "disease"

    # make a separate prepare_dataset function to test for making sgIndex?
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex
    X.obs["condition_unique_idxs"] = X.obs["condition_unique_idxs"].astype("category")

    # Pre-calculate gene means
    means = np.mean(X.X, axis=0)  # type: ignore
    X.var["means"] = means

    l1 = 1e-4
    factors, _, _ = new_parafac2(X, condition_name="disease", rank=10, l1=l1)

    if l1:
        assert np.any(factors[0] >= 0)

    for factor in factors:
        assert isinstance(factor, np.ndarray)
        assert factor.shape[1] == 10
        assert not np.all(factor == 0)
        assert not np.any(np.isnan(factor))
        assert not np.any(np.isinf(factor))