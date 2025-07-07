"""This file will plot the projections of the PF2 model
to observe patient distributions in the latent space.
We suspect there might be a few outliers in the latent space."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import setupBase
from mrsa_ca_rna.utils import concat_datasets


def figure_setup():
    X = concat_datasets()
    
    # # Define outlier samples to remove
    # outlier_samples = [
    #     "SRR22854005", "SRR22854037", "SRR22854038", "SRR22854058", 
    #     "GSM5361028", "GSM3534389", "GSM3926766", "GSM3926810", 
    #     "GSM3926774", "GSM3926857", "GSM7677818"
    # ]
    
    # # Remove specific outlier samples instead of entire dataset
    # X = X[~X.obs.index.isin(outlier_samples)].copy()
    
    # # Standardize the data
    # from sklearn.preprocessing import StandardScaler
    # X.X = StandardScaler().fit_transform(X.X)

    rank = 1

    X, _ = perform_parafac2(X, slice_col="disease", rank=rank)

    # Make a weighted projection DataFrame for easier plotting and disease labeling
    p_df = pd.DataFrame(
        X.obsm["Pf2_projections"],
        index=X.obs.index,
        columns=pd.Index([x for x in range(1, X.obsm["Pf2_projections"].shape[1] + 1)]),
    )
    p_df["disease"] = X.obs["disease"].values

    return p_df


def genFig():
    """Generate the figure for the projections of the PF2 model"""

    # Get the weighted projections
    projections = figure_setup()

    # Setup the projections figure
    layout = {"ncols": 4, "nrows": 4}
    fig_size = (16, 16)
    ax, f, _ = setupBase(fig_size, layout)

    # Find the absolute maximum value across all projections
    max_abs_value = projections.drop(columns=["disease"]).abs().max().max()

    # Normalize the projections by the maximum absolute value
    projections.iloc[:, :-1] /= max_abs_value

    # For each disease, plot the projections
    for i, disease in enumerate(projections["disease"].unique()):
        # Subset the DataFrame for the current disease
        projection: pd.DataFrame = projections.loc[
            projections["disease"] == disease
        ].drop(columns=["disease"])

        # We normalized the projections so we could directly compare them
        # across diseases, so we can use the same vmax and vmin for all heatmaps
        a = sns.heatmap(
            projection,
            ax=ax[i],
            cmap="coolwarm",
            vmax=1,
            vmin=-1,
            center=0,
            cbar=True,
            xticklabels=projection.columns.to_list(),
            yticklabels=False,
        )
        a.set_title(f"Weighted Projections for {disease}")
        a.set_xlabel("Eigenstate")

    # Setup a new figure for the overall distribution of samples
    layout = {"ncols": 1, "nrows": 1}
    fig_size = (4, 4)
    ax, g, _ = setupBase(fig_size, layout)

    # Calculate the percentiles based on absolute values from zero
    column_name = projections.columns[0]
    abs_values = projections[column_name].abs()
    p50 = abs_values.quantile(0.5)
    p75 = abs_values.quantile(0.75)
    p95 = abs_values.quantile(0.95)

    # Add sample counts to disease labels
    disease_counts = projections["disease"].value_counts().to_dict()
    projections["disease_with_count"] = projections["disease"].apply(
        lambda x: f"{x} (n={disease_counts[x]})"
    )

    # Plot a stripplot of the samples across all diseases
    b = sns.stripplot(
        data=projections,
        x="disease_with_count",
        y=column_name,
        ax=ax[0],
        jitter=True,
        alpha=0.5,
        color="black",
    )

    # Add horizontal lines for the percentiles (symmetric around 0)
    b.axhline(y=0, color='black', linestyle='-', alpha=0.7, label='Zero')
    b.axhline(y=p50, color='red', linestyle='--', alpha=0.7, label='50th percentile')
    b.axhline(y=-p50, color='red', linestyle='--', alpha=0.7)
    b.axhline(y=p75, color='orange', linestyle='--', alpha=0.7, label='75th percentile')
    b.axhline(y=-p75, color='orange', linestyle='--', alpha=0.7)
    b.axhline(y=p95, color='green', linestyle='--', alpha=0.7, label='95th percentile')
    b.axhline(y=-p95, color='green', linestyle='--', alpha=0.7)

    # Add a legend
    b.legend(loc="best", frameon=True, framealpha=0.7)

    # Rotate x-tick labels
    plt.setp(b.get_xticklabels(), rotation=45, ha="right")

    b.set_title("Distribution of Samples Across Diseases")
    b.set_xlabel("Disease")
    b.set_ylabel("Projection Value")
    
    # Identify outliers within each disease
    column_name = projections.columns[0]  # The first projection column
    outliers = identify_disease_specific_outliers(projections, column_name)
    
    return f, g


def identify_disease_specific_outliers(projections, column_name, z_threshold=2.5, iqr_factor=1.5):
    """
    Identify outliers within each disease separately.
    
    Parameters:
    - projections: DataFrame with projection values and disease labels
    - column_name: Column containing projection values to analyze
    - z_threshold: Z-score threshold (default: 2.5 standard deviations)
    - iqr_factor: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
    - Dictionary mapping diseases to their outliers
    """
    print("\n--- OUTLIER ANALYSIS BY DISEASE ---")
    outliers_by_disease = {}
    
    for disease in projections["disease"].unique():
        # Get samples for this disease only
        disease_df = projections[projections["disease"] == disease]
        disease_values = disease_df[column_name]
        
        # Calculate disease-specific statistics
        mean = disease_values.mean()
        std = disease_values.std()
        q1 = disease_values.quantile(0.25)
        q3 = disease_values.quantile(0.75)
        iqr = q3 - q1
        
        # Find outliers using Z-score method
        z_outliers = disease_df[abs(disease_values - mean) > z_threshold * std]
        
        # Find outliers using IQR method
        iqr_outliers = disease_df[(disease_values < q1 - iqr_factor * iqr) | 
                                 (disease_values > q3 + iqr_factor * iqr)]
        
        # Combine outliers from both methods
        all_outliers = pd.concat([z_outliers, iqr_outliers]).drop_duplicates()
        
        if not all_outliers.empty:
            outliers_by_disease[disease] = all_outliers
            
            print(f"\n{disease} (n={len(disease_df)}):")
            print(f"  Disease mean: {mean:.4f}, std: {std:.4f}")
            print(f"  Disease Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
            print(f"  Found {len(all_outliers)} outliers:")
            
            for idx, row in all_outliers.iterrows():
                z_score = (row[column_name] - mean) / std if std > 0 else float('inf')
                print(f"    Sample {idx}: value = {row[column_name]:.4f}, z-score = {z_score:.2f}")
    
    return outliers_by_disease
