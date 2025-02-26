"""
This file explores the stability of the parafac2 factor matrices with changing ranks
and L1 strengths.

TODO: Add a loop to test different ranks and L1 strengths.
    Add genFig() to plot FMS vs. rank and L1 strength.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    """Collect and organize data for plotting. This function will load the data and
    perform resampling, then pf2 factorization with different ranks and L1 strengths.
    
    TODO: Make a loop to test different ranks and L1 strengths.
    """

    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    disease_data = concat_datasets(disease_list, scale=False, tpm=True)

    # store the original data and resampled data
    X = disease_data.copy()
    X.X = StandardScaler().fit_transform(X.X)

    X_resampled = resample(disease_data).copy()
    X_resampled.X = StandardScaler().fit_transform(X_resampled.X)

    # perform the parafac2 factorization on the original data
    rank = 50

    weights_true, factors_true, _, R2X_true = perform_parafac2(
        X, condition_name="disease", rank=rank, l1=0
    )

    # perform the parafac2 factorization on the resampled data
    weights_resampled, factors_resampled, _, R2X_resampled = perform_parafac2(
        X_resampled, condition_name="disease", rank=rank, l1=0
    )

    # convert the factors to cp_tensors
    factors_true = (weights_true, factors_true)
    factors_resampled = (weights_resampled, factors_resampled)

    # calculate the factor match score
    factor_match = factor_match_score(
        factors_true, factors_resampled, consider_weights=False, skip_mode=1
        )

    print(f"R2X True: {R2X_true}")
    print(f"R2X Resampled: {R2X_resampled}")
    print(f"Factor Match Score: {factor_match}")

    return X, X_resampled, factors_true, factors_resampled
