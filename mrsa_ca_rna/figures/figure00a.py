# main module imports
import numpy as np
import pandas as pd
import seaborn as sns

# secondary module imports
# local module imports
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.import_data import concat_datasets
from mrsa_ca_rna.pca import perform_pca


def setup_figure00a():
    """
    Set up the data for the PCA explained variance figure.
    Goal is to explore the variance within each data set.
    """

    # data import, concatenation, scaling, and preparation
    concat_diseases = concat_datasets(
        ["mrsa", "ca", "bc", "covid", "healthy"], scale=True, tpm=True
    )

    data_labels = concat_diseases.obs["disease"].unique()
    dataset_list = [concat_diseases[concat_diseases.obs["disease"] == label] for label in data_labels]

    for i, dataset in enumerate(dataset_list):
        _, _, pca = perform_pca(dataset.to_df(), scale=False)

        components = np.arange(1, pca.n_components_ + 1, dtype=int)
        total_explained = np.cumsum(pca.explained_variance_ratio_)

        data = pd.DataFrame(components, columns=pd.Index(["components"]))
        data["total_explained"] = total_explained

        dataset_list[i] = data

    variance_data = dict(zip(data_labels, dataset_list, strict=False))

    return variance_data

def genFig():
    """
    Generate the PCA explained variance figure.
    """

    fig_size = (9, 3)
    layout = {"ncols": 3, "nrows": 2}
    ax, f, _ = setupBase(fig_size, layout)

    datasets = setup_figure00a()

    for i, keys in enumerate(datasets):
        a = sns.lineplot(
            data=datasets[keys], x="components", y="total_explained", ax=ax[i]
        )
        a.set_xlabel("# of Components")
        a.set_ylabel("Fraction of explained variance")
        a.set_title(f"PCA performance of {keys} dataset")

    return f
genFig()