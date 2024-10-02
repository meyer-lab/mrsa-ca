"""
Plotting components of MRSA data against patient metadata
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.regression import perform_PLSR


def figure08_setup():
    # should I scale the data prior to splitting it and performing PLSR?
    whole_data = concat_datasets(scale=False, tpm=True)

    mrsa_df = whole_data[whole_data.obs["disease"] == "MRSA"].to_df()
    ca_df = whole_data[whole_data.obs["disease"] == "Candidemia"].to_df()

    # independently scale, using StandardScaler, the two datasets to avoid data leakage
    scaler = StandardScaler()
    mrsa_df.loc[:, :] = scaler.fit_transform(mrsa_df.values)
    ca_df.loc[:, :] = scaler.fit_transform(ca_df.values)

    X_data = mrsa_df.T
    y_data = ca_df.T

    _, loadings, _ = perform_PLSR(X_data, y_data, 10)

    """
    Set up two dfs to pass to genFig(), one for the loadings of the components
    and one for the MRSA metadata
    """

    mrsa_loadings: pd.DataFrame = loadings["X"]
    mrsa_meta = whole_data.obs.loc[
        whole_data.obs["disease"] == "MRSA", ["gender", "age", "status"]
    ]

    # # start by plotting mrsa (x) loadings with status metadata to find components associated with outcome
    # mrsa_loadings :pd.DataFrame = pd.concat([mrsa_whole["meta"]["status"], loadings["X"]], axis=1)

    # # order mrsa_laodings dataframe by status to better visualize the components
    # mrsa_loadings = mrsa_loadings.sort_values(by="status")

    return mrsa_loadings, mrsa_meta


def genFig():
    fig_size = (4, 4)
    layout = {"ncols": 1, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    data, meta = figure08_setup()

    meta.loc[:, :] = meta.to_numpy(dtype=float)

    """For age coloring, we need to first sort the metadata by age so that our colormap
    is continuous in increasing age. Then we colormap all of our metadata before sorting again
    for plotting"""

    # sort metadata by age for continuous colormap
    meta = meta.sort_values(by="age")

    # Get the unique values of the age metadata
    unique_age_values = meta["age"].unique()

    # Create a subset of the "viridis" colormap that spans the unique age values
    original_cmap = plt.get_cmap("viridis")
    subset = np.linspace(0, 1, unique_age_values.size)
    new_cmap = original_cmap(subset)

    # grab two paired subsets of "Paired" colormap to color the status and gender metadata
    paired_cmap = plt.get_cmap("Paired")

    # create 3 color maps for age, gender, and status metadata
    status_colors = dict(zip(meta["status"].unique(), paired_cmap([0, 1]), strict=False))
    gender_colors = dict(zip(meta["gender"].unique(), paired_cmap([2, 3]), strict=False))
    age_colors = dict(zip(unique_age_values, new_cmap, strict=False))

    """Now that the colors are mapped, we can sort the data and metadata before plotting"""
    # sort the meta data by status before plotting
    meta = meta.sort_values(by=["status", "age"])

    # sort the data by status before plotting
    data = data.loc[meta.index]

    row_colors = pd.concat(
        [
            meta["gender"].map(gender_colors),
            meta["status"].map(status_colors),
            meta["age"].map(age_colors),
        ],
        axis=1,
    )
    f = sns.clustermap(data, row_cluster=False, col_cluster=True, row_colors=row_colors)
    f.ax_row_dendrogram.set_visible(False)
    f.ax_col_dendrogram.set_visible(True)

    legend_elements = []
    for label, color in status_colors.items():
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor="black", label=f"Status: {label}")
        )
    for label, color in gender_colors.items():
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor="black", label=f"Gender: {label}")
        )
    for label, color in age_colors.items():
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor="black", label=f"Age: {label}")
        )

    f.ax_heatmap.legend(
        handles=legend_elements,
        title="Metadata",
        bbox_to_anchor=[-0.35, 0],
        loc="lower left",
    )

    # # make a legend for all 3 metadata
    # handles1 = [Patch(facecolor=lut[name]) for name in lut]
    # plt.legend(handles1, lut.keys(), title="Status", loc="upper right")
    # handles2 = [Patch(facecolor=lut2[name]) for name in lut2]
    # plt.legend(handles2, lut2.keys(), title="gender", loc="upper right")
    # handles3 = [Patch(facecolor=lut3[name]) for name in lut3]
    # plt.legend(handles3, lut3.keys(), title="age", loc="upper right")

    return f
