"""
Graph PC's against each other in pairs (PC1 vs PC2, PC3 vs PC4, etc.)
and analyze the results. We are hoping to see interesting patterns
across patients i.e. the scores matrix.

To-do:

    Choose either to:
        1. Swap to sns.pairplot, comparing whole chunks of PCA plots together.
        2. Apply a GMM model and select the best performing PCA plots 
"""

import numpy as np

from mrsa_ca_rna.pca import perform_PCA
from mrsa_ca_rna.figures.base import setupBase
import seaborn as sns

# from sklearn.mixture import GaussianMixture
# from scipy.spatial.distance import cdist



def genFig():
    fig_size = (12, 9)
    layout = {
        "ncols": 4,
        "nrows": 3,
    }
    ax, f, _ = setupBase(fig_size, layout)

    scores, _, _ = perform_PCA()

    # modify what components you want to compare to one another:
    component_pairs = np.array(
        [
            [1, 2],
            [1, 3],
            [2, 3],
            [2, 4],
            [3, 4],
            [3, 5],
            [4, 5],
            [4, 6],
            [5, 6],
            [5, 7],
            [6, 7],
            [7, 8],
        ],
        dtype=int,
    )

    assert (
        component_pairs.shape[0] == layout["ncols"] * layout["nrows"]
    ), "component pairs to be graphed do not match figure layout size"

    """GMM on hold until LogisticalRegressionCV is implemented"""
    # largest = np.zeros(10)
    # for i in range(2,len(scores.columns[2:])+1):
    #     j = i
    #     while j < len(scores.columns[2:]):
    #         data = scores.iloc[:,[i,j]]

    #         gmm = GaussianMixture(n_components=4, random_state=0).fit(data)
    #         distances = cdist(gmm.means_, gmm.means_)
    #         total_dist = distances.sum() / 2

    #         for k in range(len(largest)):
    #             if total_dist >= largest[k]:
    #                 largest[k] = total_dist
    #                 print(f"new largest found at {scores.columns[i]} vs {scores.columns[j]}. New largest: {largest}")
    #                 break
    #         j += 1

    a = sns.pairplot(scores.loc[:,"disease":"PC10"], hue="disease", palette="viridis")
    return a

    # for i, (j, k) in enumerate(component_pairs):
    #     a = sns.scatterplot(
    #         data=scores.loc[:, (scores.columns[j+1], scores.columns[k+1])],
    #         x=scores.columns[j+1],
    #         y=scores.columns[k+1],
    #         hue=scores.loc[:, "disease"],
    #         ax=ax[i],
    #     )

    #     a.set_xlabel(scores.columns[j+1])
    #     a.set_ylabel(scores.columns[k+1])
    #     a.set_title(f"Var Comp {scores.columns[j+1]} vs {scores.columns[k+1]}")

    # return f


"""Debug function call section"""
fig = genFig()
fig.savefig("./mrsa_ca_rna/output/fig01_Pairplot.png")
