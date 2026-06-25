from ..data import Atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
from typing import Literal


def highly_variable_genes(
        atlas: Atlas,
        flavor: Literal["seurat", "cv", "var"] = "seurat",

        # General parameters: supported by both underlying functions
        hvg_key: str = "highly_variable_genes",
        sample_other: int | None = 20000,

        # Parameters for the cv / var versions: only passed to highly_variable_genes_plot()
        mean_key: str = "hvg_mean",
        var_key: str = "hvg_var",
        std_key: str = "hvg_std",
        score_key: str = "hvg_score",
        figsize: tuple[float, float] | None = None,
        point_size_hvg: float = 8,
        point_size_other: float = 6,
        alpha_hvg: float = 0.9,
        alpha_other: float = 0.6,

        save_path: PathLike[str] | str | None = None,
):
    """Plot diagnostic figures for highly variable gene selection results.

    This function is the public entry point for HVG plotting. It calls the corresponding
    underlying plotting logic according to ``flavor``:
    ``"seurat"`` reads ``means``, ``dispersions``, and ``dispersions_norm``;
    ``"cv"`` or ``"var"`` reads ``hvg_mean``, ``hvg_var``, ``hvg_std``, and ``hvg_score``.
    All styles highlight genes already marked as highly variable according to ``hvg_key``.

    This plot is similar to Scanpy ``sc.pl.highly_variable_genes`` and is mainly used
    to check whether the highly variable gene selection results are reasonable, rather
    than recalculating highly variable genes.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        ``var`` table must already contain the HVG statistic columns required by
        the corresponding ``flavor``.

    flavor
        Plotting style. ``"seurat"`` uses Seurat-style dispersion results;
        ``"cv"`` and ``"var"`` use mean, variance, standard deviation, and highly
        variable gene score results.

    hvg_key
        Column name in ``var`` that marks highly variable genes.

    sample_other
        Number of non-highly-variable genes to sample for display.
        If ``None``, all non-highly-variable genes are plotted.

    mean_key
        Column name in ``var`` storing the mean.

    var_key
        Column name in ``var`` storing the variance.

    std_key
        Column name in ``var`` storing the standard deviation.

    score_key
        Column name in ``var`` storing the highly variable gene score.

    figsize
        Matplotlib figure size for the CV/variance-style plot; the Seurat-style plot
        uses the fixed size in the underlying function.

    point_size_hvg
        Scatter point size for highly variable genes.

    point_size_other
        Scatter point size for non-highly-variable genes.

    alpha_hvg
        Scatter point transparency for highly variable genes.

    alpha_other
        Scatter point transparency for non-highly-variable genes.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Returns
    -------
    None

    Examples
    --------
    Plot the default highly variable gene results::

        sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
        sap.pl.highly_variable_genes(atlas)

    Plot CV-style results::

        sap.pl.highly_variable_genes(
            atlas,
            flavor="cv",
            hvg_key="highly_variable_genes",
        )

    Save a Seurat-style plot::

        sap.pl.highly_variable_genes(
            atlas,
            flavor="seurat",
            hvg_key="highly_variable_genes",
            sample_other=50000,
            save_path=r"F:\\figures\\hvg.png",
        )"""

    flavor = str(flavor).lower().strip()

    if flavor == "seurat":
        return _highly_variable_genes_plot_seurat(
            atlas=atlas,
            hvg_key=hvg_key,
            sample_other=sample_other,
            save_path=save_path,
        )

    elif flavor in ["cv", "var"]:
        return _highly_variable_genes_plot(
            atlas=atlas,
            hvg_key=hvg_key,
            mean_key=mean_key,
            var_key=var_key,
            std_key=std_key,
            score_key=score_key,
            sample_other=sample_other,
            figsize=figsize,
            point_size_hvg=point_size_hvg,
            point_size_other=point_size_other,
            alpha_hvg=alpha_hvg,
            alpha_other=alpha_other,
            save_path=save_path,
        )

    else:
        raise ValueError(
            f"Unsupported flavor: {flavor}. "
            "Available values are: 'seurat', 'cv', 'var'"
        )


# HVG: cv / var version
def _highly_variable_genes_plot(
        atlas: Atlas,
        hvg_key: str = "highly_variable_genes",
        mean_key: str = "hvg_mean",
        var_key: str = "hvg_var",
        std_key: str = "hvg_std",
        score_key: str = "hvg_score",
        sample_other: int | None = 20000,
        figsize: tuple[float, float] | None = None,
        point_size_hvg: float = 8,
        point_size_other: float = 6,
        alpha_hvg: float = 0.9,
        alpha_other: float = 0.6,
        save_path: PathLike[str] | str | None = None
):

    """Plot a cv/var-style diagnostic figure for highly variable gene selection.

    This internal plotting function reads each gene's mean, variance, standard deviation,
    highly variable gene score, and HVG marker from the ``var`` table, and draws two
    diagnostic scatter plots: the left panel shows the relationship between the
    normalized highly variable gene score and average expression, while the right panel
    shows the relationship between raw variance and average expression. Highly variable
    genes are highlighted separately.

    This plot is used to check whether CV- or variance-style HVG selection is reasonable,
    for example whether highly variable genes are concentrated in the expected expression
    range, whether the non-highly-variable gene background is too dense, and whether
    ``score_key`` can effectively distinguish highly variable genes from other genes.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        ``var`` table must contain HVG statistic columns.

    hvg_key
        Boolean column name in ``var`` indicating highly variable genes.

    mean_key
        Column name in ``var`` storing gene means.

    var_key
        Column name in ``var`` storing gene variances.

    std_key
        Column name in ``var`` storing gene standard deviations.

    score_key
        Column name in ``var`` storing scores.

    sample_other
        Number of non-highly-variable genes to sample for plotting.
        If ``None``, all non-highly-variable genes are plotted.

    figsize
        Matplotlib figure size. If ``None``, the Matplotlib default size is used.

    point_size_hvg
        Point size for highly variable genes.

    point_size_other
        Point size for non-highly-variable genes.

    alpha_hvg
        Point transparency for highly variable genes.

    alpha_other
        Point transparency for non-highly-variable genes.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Notes
    -----
    Before plotting, you need to run a highly variable gene calculation workflow that
    writes ``hvg_mean``, ``hvg_var``, ``hvg_std``, ``hvg_score``, and ``hvg_key``.
    This function only reads the results and plots them; it does not recalculate HVGs.

    Examples
    --------
    Plot CV/variance-style HVG results::

        sap.pl.highly_variable_genes(atlas, flavor="cv")
    """

    start = datetime.now()
    conn = atlas.connection

    # Check whether columns exist in var
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    needed = [hvg_key, mean_key, var_key, std_key, score_key, "atlas_gene_name"]
    missing = [c for c in needed if c not in var_cols]
    if missing:
        raise ValueError(
            f"These columns do not exist in var: {missing}\n"
            f"Please run the modified sap.pp.highly_variable_genes(atlas) first"
        )

    # Read gene-level results directly from var
    df = conn.execute(f"""
        SELECT
            atlas_gene_id,
            atlas_gene_name,
            COALESCE({hvg_key}, FALSE) AS is_hvg,
            COALESCE({mean_key}, 0.0)  AS mean_expr,
            COALESCE({var_key}, 0.0)   AS var_expr,
            COALESCE({std_key}, 0.0)   AS std_expr,
            COALESCE({score_key}, 0.0) AS score_expr
        FROM var
        ORDER BY atlas_gene_id
    """).fetchdf()

    df["var_norm_display"] = df["score_expr"]

    # Optional sampling of non-HVG genes
    if sample_other is not None:
        df_hvg = df[df["is_hvg"]].copy()
        df_other = df[~df["is_hvg"]].copy()

        if len(df_other) > sample_other:
            df_other = df_other.sample(sample_other, random_state=0)

        plot_df = pd.concat([df_hvg, df_other], axis=0, ignore_index=True)
    else:
        plot_df = df.copy()

    plot_hvg = plot_df[plot_df["is_hvg"]].copy()
    plot_other = plot_df[~plot_df["is_hvg"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    # ---------- Left panel ----------
    ax = axes[0]

    if len(plot_other) > 0:
        ax.scatter(
            plot_other["mean_expr"].to_numpy(),
            plot_other["var_norm_display"].to_numpy(),
            s=point_size_other,
            c="#8c8c8c",
            alpha=alpha_other,
            linewidths=0,
            label="other genes"
        )

    if len(plot_hvg) > 0:
        ax.scatter(
            plot_hvg["mean_expr"].to_numpy(),
            plot_hvg["var_norm_display"].to_numpy(),
            s=point_size_hvg,
            c="black",
            alpha=alpha_hvg,
            linewidths=0,
            label="highly variable genes"
        )

    ax.set_xlabel("mean expressions of genes", fontsize=16)
    ax.set_ylabel("variances of genes (normalized)", fontsize=16)

    ax.legend(
        frameon=True,
        fontsize=11,
        markerscale=1.0,
        loc="upper left",
        borderpad=0.4,
        handlelength=1.2,
        handletextpad=0.4
    )

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    # ---------- Right panel ----------
    ax = axes[1]

    if len(plot_other) > 0:
        ax.scatter(
            plot_other["mean_expr"].to_numpy(),
            plot_other["var_expr"].to_numpy(),
            s=point_size_other,
            c="#8c8c8c",
            alpha=alpha_other,
            linewidths=0
        )

    if len(plot_hvg) > 0:
        ax.scatter(
            plot_hvg["mean_expr"].to_numpy(),
            plot_hvg["var_expr"].to_numpy(),
            s=point_size_hvg,
            c="black",
            alpha=alpha_hvg,
            linewidths=0
        )

    ax.set_xlabel("mean expressions of genes", fontsize=16)
    ax.set_ylabel("variances of genes (not normalized)", fontsize=16)

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    plt.tight_layout(pad=1.0)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# HVG: seurat version
def _highly_variable_genes_plot_seurat(
        atlas: Atlas,
        hvg_key: str = "highly_variable_genes",
        sample_other: int | None = 20000,
        save_path: PathLike[str] | str | None = None,
):

    """Plot a Seurat-style diagnostic figure for highly variable gene selection.

    This internal plotting function reads Seurat-style HVG results from the ``var``
    table, including ``means``, ``dispersions``, ``dispersions_norm``, and ``hvg_key``.
    The function draws two scatter plots: the left panel shows the relationship between
    normalized dispersion and average expression, and the right panel shows the
    relationship between raw dispersion and average expression, with highly variable
    genes marked by ``hvg_key`` highlighted.

    This plot is suitable for checking whether the Seurat-style binned normalized
    dispersion is reasonable and whether the finally selected highly variable genes
    are distributed within the expected expression range.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        ``var`` table must contain Seurat-style HVG statistic columns.

    hvg_key
        Boolean column name in ``var`` indicating highly variable genes.

    sample_other
        Number of non-highly-variable genes to sample for plotting.
        If ``None``, all non-highly-variable genes are plotted.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not written to a file.

    Returns
    -------
    None
        The function directly plots the figure and saves it when ``save_path`` is not
        ``None``.

    Notes
    -----
    Before plotting, you need to run the Seurat-style highly variable gene calculation
    workflow and ensure that ``means``, ``dispersions``, and ``dispersions_norm``
    already exist in the ``var`` table.

    Examples
    --------
    Plot the default Seurat-style HVG results::

        sap.pl.highly_variable_genes(atlas, flavor="seurat")
    """

    start = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    # Safe quoting for DuckDB fields
    def _q(name: str) -> str:
        """Add double-quote quoting for DuckDB SQL identifiers.

        This internal helper is used to safely concatenate column names in the ``var``
        table, avoiding SQL parsing issues when column names contain special characters
        or conflict with SQL keywords. The function only handles identifier quoting,
        not SQL value escaping.

        Parameters
        ----------
        name
            Column name to use as a SQL identifier.

        Returns
        -------
        str
            SQL identifier with double quotes added and internal double quotes escaped.
        """
        return '"' + name.replace('"', '""') + '"'

    # Check whether columns exist in var
    var_cols = [
        r[0]
        for r in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'var'
        """).fetchall()
    ]

    needed = [
        "atlas_gene_id",
        "atlas_gene_name",
        hvg_key,
        "means",
        "dispersions",
        "dispersions_norm",
    ]

    missing = [c for c in needed if c not in var_cols]

    if missing:
        raise ValueError(
            f"These columns do not exist in var: {missing}\n"
            f"Please run highly_variable_genes_seurat(atlas) first"
        )

    # Read the HVG results already saved in var
    df = conn.execute(f"""
        SELECT
            atlas_gene_id,
            atlas_gene_name,
            COALESCE({_q(hvg_key)}, FALSE) AS is_hvg,
            means,
            dispersions,
            dispersions_norm
        FROM var
        ORDER BY atlas_gene_id
    """).fetchdf()

    # Clean nan / inf
    for col in ["means", "dispersions", "dispersions_norm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df["is_hvg"] = df["is_hvg"].fillna(False).astype(bool)

    df = df[df["means"].notna()].copy()

    if len(df) == 0:
        raise ValueError(
            "var.means is entirely empty, unable to plot. Please run highly_variable_genes_seurat(atlas) first."
        )

    # Optional sampling of non-HVG genes
    if sample_other is not None:
        df_hvg = df[df["is_hvg"]].copy()
        df_other = df[~df["is_hvg"]].copy()

        if len(df_other) > sample_other:
            df_other = df_other.sample(
                n=int(sample_other),
                random_state=0,
            )

        plot_df = pd.concat(
            [df_hvg, df_other],
            axis=0,
            ignore_index=True,
        )
    else:
        plot_df = df.copy()

    plot_hvg = plot_df[plot_df["is_hvg"]].copy()
    plot_other = plot_df[~plot_df["is_hvg"]].copy()

    # Plot
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        facecolor="white",
    )

    # Left panel: normalized dispersions
    ax = axes[0]

    other_norm = plot_other[plot_other["dispersions_norm"].notna()]
    hvg_norm = plot_hvg[plot_hvg["dispersions_norm"].notna()]

    if len(other_norm) > 0:
        ax.scatter(
            other_norm["means"].to_numpy(),
            other_norm["dispersions_norm"].to_numpy(),
            s=6,
            c="#9a9a9a",
            alpha=0.55,
            linewidths=0,
            label="other genes",
        )

    if len(hvg_norm) > 0:
        ax.scatter(
            hvg_norm["means"].to_numpy(),
            hvg_norm["dispersions_norm"].to_numpy(),
            s=8,
            c="black",
            alpha=0.9,
            linewidths=0,
            label="highly variable genes",
        )

    ax.set_xlabel("means of genes", fontsize=14)
    ax.set_ylabel("dispersions of genes (normalized)", fontsize=14)

    ax.legend(
        frameon=True,
        fontsize=10,
        markerscale=1.2,
        loc="upper left",
    )

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    # Right panel: raw dispersions
    ax = axes[1]

    other_disp = plot_other[plot_other["dispersions"].notna()]
    hvg_disp = plot_hvg[plot_hvg["dispersions"].notna()]

    if len(other_disp) > 0:
        ax.scatter(
            other_disp["means"].to_numpy(),
            other_disp["dispersions"].to_numpy(),
            s=6,
            c="#9a9a9a",
            alpha=0.55,
            linewidths=0,
        )

    if len(hvg_disp) > 0:
        ax.scatter(
            hvg_disp["means"].to_numpy(),
            hvg_disp["dispersions"].to_numpy(),
            s=8,
            c="black",
            alpha=0.9,
            linewidths=0,
        )

    ax.set_xlabel("means of genes", fontsize=14)
    ax.set_ylabel("dispersions of genes (not normalized)", fontsize=14)

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout(pad=1.0)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
