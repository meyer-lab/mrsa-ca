"""This file will perform a sweep of the parafac2 tensor factorization to find the best
rank, L1 strength, and data size for the model."""

from datetime import datetime

import wandb as wb
from sklearn.preprocessing import StandardScaler
from tlviz.factor_tools import factor_match_score

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.utils import prepare_data, resample_adata


def objective(config):
    rank = config.rank
    thresh = config.thresh

    disease_data = prepare_data(
        filter_threshold=thresh,
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

    X, _ = perform_parafac2(
        X,
        slice_col="disease",
        rank=rank,
    )
    cp_true = (
        X.uns["Pf2_weights"],
        [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]],
    )

    X, _ = perform_parafac2(
        X_resampled,
        slice_col="disease",
        rank=rank,
    )
    cp_resampled = (
        X.uns["Pf2_weights"],
        [X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"]],
    )

    """calculate the factor match score for different ranks and L1 strengths."""
    fms = factor_match_score(
        cp_true,
        cp_resampled,
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
    thresh_values = [0, 4.0, 8.1]

    sweep_config = {
        "name": "sweep_pf2_" + current_time,
        "method": "grid",
        "metric": {"name": "fms", "goal": "maximize"},
        "parameters": {
            "rank": {"values": ranks},
            "thresh": {"values": thresh_values},
        },
    }

    sweep_id = wb.sweep(sweep=sweep_config, project="pf2_r2x_fms")
    wb.agent(sweep_id, function=sweep)

    return sweep_id
