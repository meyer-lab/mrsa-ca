"""This file plots the pf2 factor matrices for the disease datasets"""

import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import seaborn as sns

from mrsa_ca_rna.factorization import perform_parafac2
from mrsa_ca_rna.figures.base import calculate_layout, setupBase
from mrsa_ca_rna.utils import (
    check_sparsity,
    concat_datasets,
)


def figure_setup():
    """Set up the data for the tensor factorization and return the results"""

    rank = 5

    disease_data = concat_datasets()

    _, factors, _, r2x = perform_parafac2(
        disease_data,
        condition_name="disease",
        rank=rank,
    )

    return factors, r2x, disease_data

def plot_with_datashader(data_df: pd.DataFrame, ax, x_label="Rank", y_label="Genes", 
                         title=None, cmap="coolwarm", center=0):
    """
    Plot a large dataframe using datashader for efficient rendering.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame to plot (rows=genes, columns=ranks)
    ax : matplotlib.axes.Axes
        The matplotlib axes to plot on
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap name
    center : float, optional
        Center value for diverging colormaps
        
    Returns:
    --------
    img : matplotlib.image.AxesImage
        The image object for further customization
    """
    
    # Convert DataFrame to long format for datashader
    df_long = data_df.reset_index().melt(
        id_vars='index', 
        var_name='rank', 
        value_name='value'
    )
    
    # Create canvas with appropriate dimensions
    cvs = ds.Canvas(
        plot_width=len(data_df.columns)*20,
        plot_height=min(1000, len(data_df))
    )
    
    # Aggregate points by rank and gene index
    agg = cvs.points(
        df_long, 
        x='rank', 
        y='index', 
        agg=ds.mean('value')
    )
    
    # Render the aggregate as an image
    if center is not None:
        img = tf.shade(agg, cmap=cmap, how='eq_hist', span=[-abs(agg.data.max()), abs(agg.data.max())])
    else:
        img = tf.shade(agg, cmap=cmap, how='eq_hist')
    
    # Convert to RGB array for matplotlib
    img_data = tf.Image(img).rgb.values
    
    # Plot the image on matplotlib axes
    img_plot = ax.imshow(
        img_data,
        aspect='auto',
        extent=(-0.5, len(data_df.columns)-0.5, len(data_df)-0.5, -0.5)
    )
    
    # Set axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    
    # Set x-ticks to rank names
    ax.set_xticks(range(len(data_df.columns)))
    ax.set_xticklabels(data_df.columns)
    
    # Hide y-tick labels (too many genes)
    ax.set_yticks([])
    
    return img_plot

def genFig():
    """Start by generating heatmaps of the factor matrices for the diseases and time"""

    layout, fig_size = calculate_layout(num_plots=3, scale_factor=4)
    ax, f, _ = setupBase(fig_size, layout)

    disease_factors, r2x, disease_data = figure_setup()

    disease_ranks = range(1, disease_factors[0].shape[1] + 1)
    disease_ranks_labels = [str(x) for x in disease_ranks]
    # x axis label: rank
    x_ax_label = "Rank"
    # y axis labels: disease, eigen, genes
    d_ax_labels = ["Disease", "Eigen-states", "Genes"]

    genes_df = pd.DataFrame(
        disease_factors[2],
        index=disease_data.var.index,
        columns=pd.Index(disease_ranks_labels),
    )

    genes_df.to_csv("output/pf2_genes.csv")

    # Check sparsity of the gene factor matrix
    sparsity = check_sparsity(genes_df.to_numpy())
    gene_count = len(genes_df.index)

    # tick labels: disease, rank, genes
    disease_labels = [
        disease_data.obs["disease"].unique(),
        disease_ranks_labels,
        False,
    ]

    # Set the A matrix colors
    A_cmap = sns.color_palette("light:#df20df", as_cmap=True)

    # set the B and C matrix colors
    BC_cmap = sns.diverging_palette(145, 300, as_cmap=True)

    # plot the disease factor matrix using non-negative cmap
    a = sns.heatmap(
        disease_factors[0],
        ax=ax[0],
        cmap=A_cmap,
        vmin=0,
        xticklabels=disease_ranks_labels,
        yticklabels=disease_labels[0],
    )
    a.set_title(f"Disease Factor Matrix\nR2X: {r2x:.2f}")
    a.set_xlabel(x_ax_label)
    a.set_ylabel(d_ax_labels[0])

    # plot the eigenstate factor matrix using diverging cmap
    b = sns.heatmap(
        disease_factors[1],
        ax=ax[1],
        cmap=BC_cmap,
        xticklabels=disease_ranks_labels,
        yticklabels=disease_labels[1],
    )
    b.set_title("Eigenstate Factor Matrix")
    b.set_xlabel(x_ax_label)
    b.set_ylabel(d_ax_labels[1])

    # plot the gene factor matrix using diverging cmap
    c = plot_with_datashader(
        genes_df,
        ax=ax[2],
        x_label=x_ax_label,
        y_label=d_ax_labels[2],
        title=f"Gene Factor Matrix\n {gene_count} genes, {sparsity:.2%} sparsity",
        cmap=BC_cmap,
        center=0,
    )

    return f
