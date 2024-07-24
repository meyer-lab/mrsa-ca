"""
Graphing a heatmap of candidemia patient RNA signals prior to scaling, after scaling,
and after weighting via regression coef output.
"""

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import seaborn as sns

from mrsa_ca_rna.import_data import import_rna_weights, concat_datasets
from mrsa_ca_rna.figures.base import setupBase


def figure04_setup():
    """Organize data for plotting"""

    scaler = StandardScaler().set_output(transform="pandas")

    scaled_weights, new_weights, nested_weights = import_rna_weights()
    rna_data = concat_datasets()
    rna_scaled_data = rna_data.copy()

    rna_scaled_data.loc[:, ~rna_scaled_data.columns.str.contains("status|disease")] = (
        scaler.fit_transform(
            rna_scaled_data.loc[
                :, ~rna_scaled_data.columns.str.contains("status|disease")
            ]
        )
    )

    ca_data = rna_data.loc[
        rna_data["disease"].str.contains("Candidemia"),
        ~rna_data.columns.str.contains("status|disease"),
    ]
    ca_scaled_data = rna_scaled_data.loc[
        rna_scaled_data["disease"].str.contains("Candidemia"),
        ~rna_scaled_data.columns.str.contains("status|disease"),
    ]

    ca_weighted_data = ca_scaled_data.copy()
    for index in ca_weighted_data.index:
        ca_weighted_data.loc[index, :] = (
            ca_weighted_data.loc[index, :].values * scaled_weights.values
        )

    return ca_data, ca_scaled_data, ca_weighted_data


def genFig():
    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    df_dict: pd.DataFrame = {"raw": 0, "scaled": 0, "weighted": 0}

    df_dict["raw"], df_dict["scaled"], df_dict["weighted"] = figure04_setup()
    totals_dict = {"raw": 0, "scaled": 0, "weighted": 0}

    for key in df_dict:
        totals = []
        for col in df_dict[key].columns:
            totals.append(df_dict[key][col].sum())
        totals = np.reshape(totals, (1, -1))
        totals_dict[key] = pd.DataFrame(
            totals, index=[key + " totals"], columns=df_dict[key].columns
        )

    largest_dict = {"raw": 0, "scaled": 0, "weighted": 0}
    smallest_dict = {"raw": 0, "scaled": 0, "weighted": 0}

    for key in largest_dict:
        largest_dict[key] = totals_dict[key].T.nlargest(
            3, totals_dict[key].index, keep="all"
        )
    for key in smallest_dict:
        smallest_dict[key] = totals_dict[key].T.nsmallest(
            3, totals_dict[key].index, keep="all"
        )

    exclusion_dict = {
        "raw": np.concatenate((largest_dict["raw"].index, smallest_dict["raw"].index)),
        "scaled": np.concatenate(
            (largest_dict["scaled"].index, smallest_dict["scaled"].index)
        ),
        "weighted": np.concatenate(
            ([largest_dict["weighted"].index, smallest_dict["weighted"].index])
        ),
    }

    for i, key in enumerate(df_dict):
        a = sns.heatmap(
            df_dict[key][exclusion_dict[key]], cmap="viridis", center=0, ax=ax[i]
        )
        a.set_title(key + " ca rna data (3 most positively/negatively correlated)")

    return f
