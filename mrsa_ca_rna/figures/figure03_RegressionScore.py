"""
This figure will show regression performance of the
MRSA outcome data in the context of the full PCA'd data
containing MRSA, CA, and Healthy patients.

Running logistic regression here is broken. The current dataset
being accessed needs to be recognized and handled case-by-case.
For the MRSA+CA case, the data needs to be truncated to just MRSA,
for the CA case, I need to figure out how to transform CA patient
data to MRSA patient data (30x60) -> (88x60)
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import scale

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.regression import perform_PC_LR
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets


def figure_03_setup(components: int = 60):
    """Create a dataFrame of regression performance over component #"""

    whole_data = concat_datasets(scale=False, tpm=True)

    # send adata to df for compatibility with previously written code
    mrsa_df = whole_data[whole_data.obs["disease"] == "MRSA"].to_df()
    combined_df = whole_data.to_df()
    ca_df = whole_data[whole_data.obs["disease"] == "Candidemia"].to_df()
    y_data = whole_data.obs.loc[whole_data.obs["disease"] == "MRSA", "status"]

    datasets = {"MRSA": mrsa_df, "MRSA+CA+Healthy": combined_df, "CA": ca_df}
    performance_dict = {}

    for dataset in datasets:
        # print(f"Performing PCA on {dataset} dataset.")
        scores_df, loadings_df, pca = perform_PCA(datasets[dataset])

        if dataset == "MRSA+CA+Healthy":
            scores_df = scores_df.loc[whole_data.obs["disease"] == "MRSA", :]

        if dataset == "CA":
            ## transform the MRSA data using CA's loadings
            # multed = np.matmul(whole_data.loc["MRSA", "rna"].to_numpy(), loadings_df.T.to_numpy())
            # scores_df = pd.DataFrame(multed, index=whole_data.loc["MRSA"].index, columns=scores_df.columns)

            # use sklearn PCA object's transform method to project CA data onto it
            scaled_MRSA = scale(mrsa_df.to_numpy())
            transformed_MRSA = pca.transform(scaled_MRSA)
            scores_df = pd.DataFrame(
                transformed_MRSA, index=mrsa_df.index, columns=scores_df.columns
            )

        # keep track of the nested CV performance (balanced accuracy) of the model. Reset for each dataset
        performance = []
        for i in range(components):
            # print(f"Regressing {dataset} dataset on MRSA outcomes w/ {i+1} components")

            # slice our X_data to our current components and set y_data to be MRSA status from whole data
            X_data = scores_df.iloc[:, : i + 1]

            # perform the logistic regression and store the nested CV performance
            nested_performance, _ = perform_PC_LR(
                X_data,
                y_data,
            )
            performance.append(nested_performance)

        performance_df = pd.DataFrame(
            {
                "Components": np.arange(1, components + 1),
                "Nested Performance": performance,
            }
        )

        performance_dict[dataset] = performance_df

    return performance_dict


def genFig():
    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data_dict = figure_03_setup()

    for i, data in enumerate(data_dict):
        a = sns.lineplot(
            data=data_dict[data], x="Components", y="Nested Performance", ax=ax[i]
        )

        a.set_xlabel("# of components")
        a.set_ylabel("Balanced Accuracy")
        a.set_title(
            f"Nested-CV Score of Logistic Regression\nPCA ({data}), 'saga' solver, 'elasticnet' penalty"
        )

    return f
