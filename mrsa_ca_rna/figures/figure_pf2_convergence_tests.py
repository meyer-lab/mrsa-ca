"""
This file will test Pf2 convergence across different settings.
These tests will be used to determine the optimal settings for the Pf2 convergence
in the project.

The tests will be performed on the following datasets:
Test #1: Regularization
Diseases: all datasets
L1 Strength: [250, 500, 750]
Genes: all genes

Test #2: Genes
Diseases: all datasets
L1 Strength: 750
Genes: threshold [0.1, 5, 10], method [mean]

Test #3: Datasets
Diseases: all but one dataset [mrsa, healthy, covid]
L1 Strength: 750
Genes: all genes
"""

from mrsa_ca_rna.utils import concat_datasets
from mrsa_ca_rna.factorization import prepare_data, perform_parafac2

import wandb as wb


def best_set():
    # Test #1: Regularization
    # Diseases: all datasets
    # L1 Strength: [250, 500, 750]
    # Genes: all genes

    best_rel_loss = 53

    # Load datasets
    for exclude in ["none", "mrsa", "covid", "healthy"]:
        if exclude == "none":
            ad_list = ["mrsa", "ca", "bc", "covid", "healthy"]
        else:
            ad_list = ["mrsa", "ca", "bc", "covid", "healthy"].remove(exclude)

        for threshold in [0, 1, 3]:
            adata = concat_datasets(
                ad_list=ad_list,
                filter_threshold=threshold,
                filter_method="mean",
                shrink=True,
                scale=True,
                tpm=True,
            )
            data_xr = prepare_data(adata, expansion_dim="disease")

            # Test L1 Strengths
            for l1_strength in [250, 500, 750]:
                wb.init(
                    project="pf2_large_scale_params",
                    config={
                        "l1_strength": l1_strength,
                        "threshold": threshold,
                        "exclude": exclude,
                    },
                )

                def callback(rec_error, rel_loss, C_sparsity):
                    wb.log(
                        {
                            "rec_error": rec_error,
                            "rel_loss": rel_loss,
                            "C_sparsity": C_sparsity,
                        }
                    )

                try:
                    decomposition, _, diag = perform_parafac2(
                        data_xr, rank=5, l1=l1_strength, callback=callback
                    )

                finally:
                    wb.finish()

                factors = decomposition[1]
                rec_errors = diag.rec_errors[-1]
                abs_loss = diag.regularized_loss[-1]
                rel_loss = (
                    diag.regularized_loss[-2] - diag.regularized_loss[-1]
                ) / diag.regularized_loss[-2]
                feasible = diag.satisfied_feasibility_condition

                if rel_loss < best_rel_loss:
                    best_rel_loss = rel_loss
                    best_l1_strength = l1_strength
                    best_threshold = threshold
                    best_exclude = exclude

    return best_rel_loss, best_l1_strength, best_threshold, best_exclude


def l1_testing():
    ad_list = ["mrsa", "ca", "bc", "covid", "healthy"]

    adata = concat_datasets(ad_list=ad_list, filter_threshold=0, filter_method="mean")
    data_xr = prepare_data(adata, expansion_dim="disease")

    max_l1 = 10000
    min_l1 = 1000
    trials = 20
    step = int((max_l1 - min_l1) / trials)

    for l1_strength in range(min_l1, max_l1, step):
        wb.init(project="pf2_l1_expansion_fixed", config={"l1_strength": l1_strength})

        def callback(rec_error, rel_loss, C_sparsity):
            wb.log(
                {"rec_error": rec_error, "rel_loss": rel_loss, "C_sparsity": C_sparsity}
            )

        try:
            decomposition, _, diag = perform_parafac2(
                data_xr, rank=10, l1=l1_strength, normalize=False, callback=callback
            )

        finally:
            wb.finish()

        factors = decomposition[1]
        rec_errors = diag.rec_errors[-1]
        abs_loss = diag.regularized_loss[-1]
        rel_loss = (
            diag.regularized_loss[-2] - diag.regularized_loss[-1]
        ) / diag.regularized_loss[-2]
        feasible = diag.satisfied_feasibility_condition

    return rel_loss


l1_testing()
