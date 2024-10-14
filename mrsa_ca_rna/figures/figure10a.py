"""This file will plot the factor matrices of the CA time data"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.figures.figure10 import figure10_setup
from mrsa_ca_rna.import_data import ca_data_split


def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    fig_size = (12, 4)
    layout = {"ncols": 3, "nrows": 1}
    ax, f, _ = setupBase(fig_size, layout)

    time_data, _, _ = ca_data_split()

    time_factors = figure10_setup()

    # push time_factors[0] to a pandas to color and sort by metadata
    time_df = pd.DataFrame(
        time_factors[0],
        index=time_data.obs.loc[:, "subject_id"].unique(),
        columns=pd.Index([f"Eigenstate {i+1}" for i in range(time_factors[0].shape[1])])
    )

    # clustermap setup for time metadata to indicate # time points
    # count the number of time points for each subject_id
    time_counts = time_data.obs["subject_id"].value_counts(ascending=True)
    time_df = pd.concat([time_counts, time_df], axis=1)

    # make subject_id column from index to map to color
    time_df = time_df.reset_index(names=["subject_id"]).set_index(
        "subject_id", drop=False
    )

    # create a color map for the number of time points
    # get original colormap
    original_cmap = plt.get_cmap("cividis")

    # create a subset of the "viridis" colormap that spans the unique time values
    subset = np.linspace(0, 1, time_counts.unique().size)
    new_cmap = original_cmap(subset)

    # create a dictionary of time_counts to color
    time_colors = dict(zip(time_counts.unique(), new_cmap, strict=False))

    # replace the count column with the color map
    row_colors = time_df.loc[:, "count"].map(time_colors)

    data = time_df.loc[:, time_df.columns.str.contains("Eigenstate")]

    f = sns.clustermap(
        data,
        row_cluster=False,
        col_cluster=False,
        cmap="viridis",
        row_colors=row_colors,
    )

    return f
