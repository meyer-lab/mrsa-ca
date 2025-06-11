import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mrsa_ca_rna.figures.figure_plot_pf2_genes import (
    plot_datashader_rasterized,
    plot_matplot_rasterized,
)


def generate_test_data(n_genes=50000, n_ranks=5, seed=42):
    """Generate random gene expression data similar in structure to real data."""

    # Setup random generator
    rng = np.random.default_rng(seed)
    # Create data with positive and negative values, similar to gene loadings
    data = rng.normal(0, 0.01, (n_genes, n_ranks))

    # Add structured patterns to mimic real gene expression data
    for rank in range(n_ranks):
        # Subset genes in each rank and amplify their values
        subset_size = n_genes // 10
        start_idx = (rank * subset_size) % n_genes
        data[start_idx : start_idx + subset_size, rank] *= 3.0

    # Create DataFrame to pass to vizualization functions
    df = pd.DataFrame(
        data,
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=[str(i + 1) for i in range(n_ranks)],
    )
    return df


def test_datashader_visualization():
    """Test the datashader visualization function with synthetic data."""
    # Generate test data
    gene_df = generate_test_data(n_genes=50000, n_ranks=5)

    # Print some datashader statistics
    min_val, max_val = gene_df.values.min(), gene_df.values.max()
    p5, p95 = np.percentile(gene_df.values, [5, 95])
    print(f"Data range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"5-95 percentile: [{p5:.6f}, {p95:.6f}]")

    # Create figure with both plotting methods for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Test standard matplotlib visualization (for comparison)
    plot_matplot_rasterized(
        gene_df, ax=ax1, title="Matplotlib Rasterized", cmap="coolwarm"
    )

    # Test the datashader visualization
    plot_datashader_rasterized(
        gene_df, ax=ax2, title="Datashader Rasterized", cmap="coolwarm"
    )

    fig.suptitle("Visualization Comparison", fontsize=16)
    plt.tight_layout()

    plt.savefig("mrsa_ca_rna/tests/test_datashader.png", dpi=150, bbox_inches="tight")

    plt.close(fig)
