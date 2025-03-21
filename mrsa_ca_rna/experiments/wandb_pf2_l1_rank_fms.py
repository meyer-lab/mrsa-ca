"""This file will perform a sweep of the parafac2 tensor factorization to find the best
rank, L1 strength, and data size for the model."""

from datetime import datetime

import numpy as np
import wandb as wb
from sklearn.preprocessing import StandardScaler
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.utils import check_sparsity, concat_datasets, resample_adata


def objective(config):
    rank = config.rank
    l1 = config.l1
    thresh = config.thresh

    disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
    disease_data = concat_datasets(
        disease_list, filter_threshold=thresh, scale=False, tpm=True
    )
    data_size = disease_data.shape[1] / 16315
    wb.log({"data_size": data_size})

    # scale original data
    X = disease_data.copy()
    X.X = StandardScaler().fit_transform(X.X)

    # resample the data and scale the data
    X_resampled = disease_data.copy()
    X_resampled = resample_adata(X_resampled)
    X_resampled.X = StandardScaler().fit_transform(X_resampled.X)

    def callback(it, error, factors, _):
        R2X = 1 - error
        sparsity = check_sparsity(factors[2])
        wb.log({"iteration": it, "R2X": R2X, "sparsity": sparsity})

    # factorize the original and resampled data with the same random state
    random_state = np.random.randint(0, 1000)

    _, factors_true, _, _ = perform_parafac2(
        X,
        condition_name="disease",
        rank=rank,
        l1=l1,
        rnd_seed=random_state,
        callback=callback,
    )

    _, factors_resampled, _, _ = perform_parafac2(
        X_resampled,
        condition_name="disease",
        rank=rank,
        l1=l1,
        rnd_seed=random_state,
    )

    """calculate the factor match score for different ranks and L1 strengths."""
    fms = factor_match_score(
        (None, factors_true),
        (None, factors_resampled),
        consider_weights=False,
        skip_mode=1,
    )

    return fms


def sweep():
    wb.init()
    fms = objective(wb.config)
    wb.log({"fms": fms})


def perform_experiment():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ranks = [x for x in range(10, 21)]
    l1_values = [0, 8e-5, 1e-4, 1.2e-4, 1.4e-4]
    thresh_values = [0, 4.0, 8.1]

    sweep_config = {
        "name": "sweep_pf2_" + current_time,
        "method": "grid",
        "metric": {"name": "fms", "goal": "maximize"},
        "parameters": {
            "rank": {"values": ranks},
            "l1": {"values": l1_values},
            "thresh": {"values": thresh_values},
        },
    }

    sweep_id = wb.sweep(sweep=sweep_config, project="pf2_r2x_fms")
    wb.agent(sweep_id, function=sweep)

    return sweep_id
