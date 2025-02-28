import anndata as ad
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tlviz.factor_tools import factor_match_score

import wandb as wb
from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.utils import check_sparsity, concat_datasets


def perform_sweep():

    def objective(config):
        
        rank = config.rank
        l1 = config.l1
        thresh = config.thresh

        disease_list = ["mrsa", "ca", "bc", "covid", "healthy"]
        disease_data = concat_datasets(
            disease_list, filter_threshold=thresh, scale=False, tpm=True
        )
        data_size = disease_data.shape[1]
        wb.log({"data_size": data_size})

        # scale original data
        X = disease_data.copy()
        X.X = StandardScaler().fit_transform(X.X)

        # resample the data
        df = disease_data.to_df()
        df.insert(0, "disease", X.obs["disease"].values)
        df_resampled: pd.DataFrame = resample(df, replace=True)  # type: ignore

        # make a unique index
        df_resampled = df_resampled.reset_index(drop=True)

        # convert back to AnnData and scale
        df_resampled.index = df_resampled.index.astype(str)
        X_resampled = ad.AnnData(df_resampled.loc[:, df_resampled.columns != "disease"])
        X_resampled.obs["disease"] = df_resampled["disease"].to_numpy()
        X_resampled.X = StandardScaler().fit_transform(X_resampled.X)

        def callback(it, error, factors, _):
            sparsity = check_sparsity(factors[2])
            wb.log({"iteration": it, "error": error, "sparsity": sparsity})

        _, factors_true, _, _ = perform_parafac2(
            X, condition_name="disease", rank=rank, l1=l1, callback=callback
        )
        
        _, factors_resampled, _, _ = perform_parafac2(
            X_resampled, condition_name="disease", rank=rank, l1=l1
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

    sweep_config = {
        "name": "size_l1_rank_sweep",
        "method": "grid",
        "metric": {"name": "fms", "goal": "maximize"},
        "parameters": {
            "rank": {"values": [10, 20, 30, 40, 50]},
            "l1": {"values": [0, 1e-6, 1e-5, 2e-5, 3e-5]},
            "thresh": {"values": [0, 0.3, 0.9, 2.7, 8.1]},
        },
    }

    sweep_id = wb.sweep(sweep=sweep_config, project="size_l1_rank_sweep")
    wb.agent(sweep_id, function=sweep)

    return sweep_id
