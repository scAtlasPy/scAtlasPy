from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Any
from ..data import Atlas
import logging
logger = logging.getLogger('Atlas')


def _import_classic_umap() -> Any:
    """Import the classic UMAP estimator without importing TensorFlow eagerly."""

    try:
        from umap.umap_ import UMAP
    except ImportError as exc:
        raise ImportError(
            "sap.tl.umap requires umap-learn. Install it with "
            "`pip install umap-learn` and call this function again."
        ) from exc

    return UMAP


def _import_parametric_umap() -> Any:
    """Import ParametricUMAP only when the user calls the parametric workflow."""

    try:
        from umap.parametric_umap import ParametricUMAP
    except ImportError as exc:
        raise ImportError(
            "sap.tl.parametric_umap requires the optional ParametricUMAP "
            "dependencies, including TensorFlow. Install them in your analysis "
            "environment with `pip install 'umap-learn[parametric_umap]'` or "
            "install a compatible TensorFlow package, then call this function again."
        ) from exc

    return ParametricUMAP


def _get_pca_columns(
    atlas: Atlas,
    n_pcs: int | None = None,
    input_table: str = "obsm_X_pca",
) -> list[str]:
    """Return PCA coordinate columns from an Atlas PCA table."""

    conn = atlas.connection

    tables = conn.execute(f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = '{input_table}'
    """).fetchdf()

    if len(tables) == 0:
        raise ValueError(
            f"{input_table} does not exist in the database.\n"
            "Please run sap.tl.pca(atlas) first"
        )

    pca_cols = [
        r[1]
        for r in conn.execute(f"PRAGMA table_info({input_table})").fetchall()
    ]

    pc_cols = [c for c in pca_cols if c.startswith("pc")]

    if len(pc_cols) == 0:
        raise ValueError(f"No pc columns exist in {input_table}")

    pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("pc", "")))

    if n_pcs is not None:
        n_pcs = int(n_pcs)

        if n_pcs <= 0:
            raise ValueError("n_pcs must be greater than 0")

        if n_pcs > len(pc_cols):
            raise ValueError(
                f"n_pcs={n_pcs} exceeds the number of available PCs in {input_table}: {len(pc_cols)}"
            )

        pc_cols = pc_cols[:n_pcs]

    return pc_cols


def knn_overlap(X_high: np.ndarray, X_low: np.ndarray, k: int = 15) -> float:

    """Calculate the nearest-neighbor overlap between high-dimensional and low-dimensional spaces.

    This function calculates the k-nearest-neighbor set of each sample in the
    high-dimensional coordinates and in the low-dimensional embedding,
    respectively, and then returns the average overlap ratio between the two
    neighbor sets. It is used to evaluate how well the UMAP embedding preserves
    the local neighborhood structure of PCA.

    For each sample, the function takes the first ``k`` nearest neighbors other
    than itself from ``X_high`` and ``X_low``, calculates the intersection ratio
    of the two neighbor sets, and finally averages it over all samples. The closer
    the score is to ``1``, the better the low-dimensional embedding preserves the
    local neighborhood in the high-dimensional space.

    Parameters
    ----------
    X_high
        Coordinate matrix in the high-dimensional space, such as PCA coordinates.

    X_low
        Coordinate matrix in the low-dimensional space, such as UMAP coordinates.

    k
        Number of nearest neighbors used to calculate the overlap ratio.

    Returns
    -------
    float
        Average kNN overlap over all samples, usually ranging from ``0`` to ``1``.

    Notes
    -----
    When the number of samples is small, the function automatically limits ``k``
    to ``n_samples - 1`` to avoid including the sample itself or nonexistent
    neighbors in the calculation.

    Examples
    --------
    Compare the local-neighborhood consistency between PCA and UMAP coordinates::

        score = knn_overlap(X_pca, X_umap, k=15)
        print(score)

    Use it together with trustworthiness in the UMAP evaluation workflow::

        tw = trustworthiness(X_pca, X_umap, n_neighbors=15)
        overlap = knn_overlap(X_pca, X_umap, k=15)"""
    n = X_high.shape[0]

    # Prevent too few samples
    k = min(k, n - 1)

    nn_high = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_low = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")

    nn_high.fit(X_high)
    nn_low.fit(X_low)

    high_idx = nn_high.kneighbors(X_high, return_distance=False)[:, 1:]
    low_idx = nn_low.kneighbors(X_low, return_distance=False)[:, 1:]

    overlap = []

    for i in range(n):
        s1 = set(high_idx[i])
        s2 = set(low_idx[i])
        overlap.append(len(s1 & s2) / k)

    return float(np.mean(overlap))


def umap(
    atlas: Atlas,
    fit_sample_n: int | None = None,
    transform_batch_size: int = 50_000,
    n_components: int = 2,
    n_pcs: int | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    spread: float = 1.0,
    metric: str = "euclidean",
    random_state: int = 42,
    n_jobs: int = 1,
    add_table: str = "obsm_X_umap",
    save_params_table: str = "uns_umap_params",
    eval_sample_n: int = 5000,
    save_eval_table: str = "uns_umap_eval"
) -> Any:
    """Calculate UMAP based on PCA embeddings.

    This function reads PCA coordinates from the ``obsm_X_pca`` table in the
    Atlas database, first fits a UMAP model on a sampled subset, then
    transforms all cells into the low-dimensional space in batches, and writes
    the UMAP coordinates to the database. It is similar to Scanpy's
    ``sc.tl.umap``, but for large-scale data it uses a "sampled fitting + full
    batched transformation" strategy to reduce the memory pressure caused by
    loading all PCA coordinates at once.

    After running, the function writes three types of results:

    - ``add_table``: UMAP coordinates for each cell, defaulting to ``obsm_X_umap``;
    - ``save_params_table``: key parameters used in this UMAP run;
    - ``save_eval_table``: trustworthiness and kNN overlap obtained from sampled evaluation.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the ``obsm_X_pca`` table must already exist in the database.

    fit_sample_n
        Number of sampled cells used to fit the UMAP model. The default value is
        ``None``, meaning that all cells in ``obsm_X_pca`` are used for fitting.

        For very large datasets, this can be set to a smaller sample size, such
        as ``500_000``, to speed up fitting and reduce memory usage.

    transform_batch_size
        Number of cells in each batch when performing ``transform`` on the full
        PCA coordinates after the model has been fitted. A larger value is
        usually faster, but increases memory usage for each batch.

    n_components
        Number of dimensions in the low-dimensional UMAP embedding. The default
        value is ``2``.

        The current result table always writes two columns, ``umap1`` and
        ``umap2``, so regular usage should keep ``n_components=2``.

    n_pcs
        Number of PCA dimensions used for UMAP. When set to ``None``, all PC
        columns in ``obsm_X_pca`` are used; when an integer is passed, only the
        first ``n_pcs`` PCs are used.

    n_neighbors
        Number of nearest neighbors used by UMAP when constructing the local
        neighborhood graph. Smaller values emphasize local structure more, while
        larger values emphasize global continuous structure more.

    min_dist
        Minimum distance allowed between points in the low-dimensional UMAP
        space. Smaller values make clusters more compact, while larger values
        make the embedding more dispersed.

    spread
        Overall spread scale of the low-dimensional UMAP space. Usually,
        ``spread >= min_dist`` is required.

    metric
        Distance metric. PCA embeddings usually use ``"euclidean"``.

    random_state
        Random seed. Setting it to a fixed integer makes the result easier to reproduce.

    n_jobs
        Number of threads used for computation.

    add_table
        Name of the result table written to the database.

    save_params_table
        Name of the database table used to save the parameters of this run.

    eval_sample_n
        Number of sampled cells used to evaluate embedding quality. Evaluation is
        performed only on a subset of the fitting sample to avoid excessive memory
        usage from large-scale distance calculations.

    save_eval_table
        Name of the database table used to save evaluation results.

    Returns
    -------
    umap.UMAP
        Fitted UMAP object, which can continue to be used for ``transform`` or
        parameter inspection.

    Notes
    -----
    This function does not calculate PCA automatically. If ``obsm_X_pca`` does
    not exist in the database, run ``sap.tl.pca(atlas)`` first.

    The fitting sample is selected by sorting on ``hash(atlas_cell_id + seed)``
    and taking the first ``fit_sample_n`` cells, making it easier to obtain a
    relatively stable sample when ``random_state`` is fixed.

    Examples
    --------
    Calculate two-dimensional UMAP using default parameters::

        sap.tl.pca(atlas)
        sap.tl.umap(atlas)

    Fit UMAP with 500,000 cells and transform the full dataset in batches::

        sap.tl.umap(
            atlas,
            fit_sample_n=500_000,
            transform_batch_size=100_000,
            n_neighbors=45,
            min_dist=0.2,
            random_state=42,
        )

    """

    start = datetime.now()
    conn = atlas.connection

    # Check obsm_X_pca
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_pca'
    """).fetchdf()

    if len(tables) == 0:
        raise ValueError(
            "obsm_X_pca does not exist in the database.\n"
            "Please run sap.tl.pca(atlas) first"
        )

    pca_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info(obsm_X_pca)").fetchall()
    ]

    pc_cols = [c for c in pca_cols if c.startswith("pc")]

    if len(pc_cols) == 0:
        raise ValueError("No pc columns exist in obsm_X_pca")

    # Ensure pc0, pc1, pc2 ... are sorted in numeric order
    pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("pc", "")))

    if n_pcs is not None:
        n_pcs = int(n_pcs)

        if n_pcs <= 0:
            raise ValueError("n_pcs must be greater than 0")

        if n_pcs > len(pc_cols):
            raise ValueError(
                f"n_pcs={n_pcs} exceeds the number of available PCs in obsm_X_pca: {len(pc_cols)}"
            )

        pc_cols = pc_cols[:n_pcs]

    pc_cols_sql = ", ".join(pc_cols)

    logger.info(f"[UMAP] using n_pcs = {len(pc_cols)}")
    logger.info(f"[UMAP] n_neighbors = {n_neighbors}")
    logger.info(f"[UMAP] min_dist = {min_dist}")
    logger.info(f"[UMAP] spread = {spread}")

    if float(spread) < float(min_dist):
        raise ValueError(
            f"spread must be greater than or equal to min_dist; current spread={spread}, min_dist={min_dist}"
        )

    # Fit UMAP: sample first in SQL
    if fit_sample_n is None:
        fit_query = f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY atlas_cell_id
        """
    else:
        seed = 0 if random_state is None else int(random_state)

        fit_query = f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY hash(atlas_cell_id + {seed})
            LIMIT {int(fit_sample_n)}
        """

    fit_df = conn.execute(fit_query).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("The sample used to fit UMAP is empty")

    X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

    UMAP = _import_classic_umap()

    # Fit the UMAP model
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs
    )

    reducer.fit(X_fit)

    # Evaluate UMAP embedding quality after training
    X_fit_umap = reducer.transform(X_fit).astype(np.float32)

    # Evaluate only on a subsample to avoid eval_n x eval_n distance matrices from exploding memory
    eval_n = min(eval_sample_n, X_fit.shape[0])

    rng = np.random.default_rng(random_state)

    eval_idx = rng.choice(
        X_fit.shape[0],
        size=eval_n,
        replace=False
    )

    X_eval = X_fit[eval_idx]
    X_eval_umap = X_fit_umap[eval_idx]

    eval_k = min(n_neighbors, eval_n - 1)

    trustworthiness_score = trustworthiness(
        X_eval,
        X_eval_umap,
        n_neighbors=eval_k
    )

    knn_overlap_score = knn_overlap(
        X_eval,
        X_eval_umap,
        k=eval_k
    )

    logger.info(f"[UMAP] trustworthiness = {trustworthiness_score:.4f}")
    logger.info(f"[UMAP] knn_overlap     = {knn_overlap_score:.4f}")

    # Simple automatic evaluation
    if trustworthiness_score < 0.80:
        logger.info(" UMAP local structure preservation is weak; it is recommended to increase fit_sample_n / adjust n_neighbors / check PCA")
    elif trustworthiness_score < 0.90:
        logger.info(" UMAP local structure preservation is normal")
    else:
        logger.info(" UMAP local structure preservation is very good")

    if knn_overlap_score < 0.20:
        logger.info(" KNN overlap is low; nearest neighbors in the low-dimensional space differ greatly from those in PCA space")
    elif knn_overlap_score < 0.40:
        logger.info(" KNN overlap is normal, which is common in single-cell UMAP")
    else:
        logger.info(" KNN overlap is high, and local neighborhoods are preserved very well")

    # Create output table
    conn.execute(f"DROP TABLE IF EXISTS {add_table}")
    conn.execute(f"""
        CREATE TABLE {add_table} (
            atlas_cell_id BIGINT,
            umap1 FLOAT,
            umap2 FLOAT
        )
    """)

    conn.execute(f"DROP TABLE IF EXISTS {save_params_table}")
    conn.execute(f"""
        CREATE TABLE {save_params_table} (
            param_name VARCHAR,
            param_value VARCHAR
        )
    """)

    params_df = pd.DataFrame({
        "param_name": [
            "n_components",
            "n_pcs",
            "n_neighbors",
            "min_dist",
            "spread",
            "metric",
            "random_state",
            "fit_sample_n",
            "transform_batch_size",
            "input_table",
            "output_table",
            "eval_sample_n"
        ],
        "param_value": [
            str(n_components),
            str(len(pc_cols)),
            str(n_neighbors),
            str(min_dist),
            str(spread),
            str(metric),
            str(random_state),
            str(fit_sample_n),
            str(transform_batch_size),
            "obsm_X_pca",
            add_table,
            str(eval_sample_n)
        ]
    })

    conn.append(save_params_table, params_df)

    # Save evaluation results
    conn.execute(f"DROP TABLE IF EXISTS {save_eval_table}")
    conn.execute(f"""
        CREATE TABLE {save_eval_table} (
            metric_name VARCHAR,
            metric_value DOUBLE
        )
    """)

    eval_df = pd.DataFrame({
        "metric_name": [
            "trustworthiness",
            "knn_overlap",
            "eval_sample_n",
            "eval_n_neighbors"
        ],
        "metric_value": [
            float(trustworthiness_score),
            float(knn_overlap_score),
            float(eval_n),
            float(eval_k)
        ]
    })

    conn.append(save_eval_table, eval_df)

    # Full-data batched transform
    total_n = conn.execute("SELECT COUNT(*) FROM obsm_X_pca").fetchone()[0]

    offset = 0

    while offset < total_n:

        batch_df = conn.execute(f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY atlas_cell_id
            LIMIT {int(transform_batch_size)}
            OFFSET {int(offset)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        X_batch = batch_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)
        X_umap = reducer.transform(X_batch).astype(np.float32)

        out_df = pd.DataFrame({
            "atlas_cell_id": batch_df["atlas_cell_id"].to_numpy(dtype=np.int64),
            "umap1": X_umap[:, 0],
            "umap2": X_umap[:, 1]
        })

        conn.append(add_table, out_df)

        offset += len(batch_df)

        logger.info(f"[UMAP] transformed {offset}/{total_n}")

    logger.info(f"UMAP Done, elapsed time: {(datetime.now() - start).total_seconds():.2f} seconds")

    return reducer


def parametric_umap(
    atlas: Atlas,
    fit_sample_n: int = 200_000,
    transform_batch_size: int = 100_000,
    parametric_batch_size: int = 1024,
    training_epochs: int = 1,
    n_components: int = 2,
    n_pcs: int | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    spread: float = 1.0,
    metric: str = "euclidean",
    random_state: int = 42,
    n_jobs: int = 1,
    add_table: str = "obsm_X_umap",
    save_params_table: str = "uns_umap_params",
    eval_sample_n: int = 5000,
    save_eval_table: str = "uns_umap_eval",
    keras_fit_kwargs: dict[str, Any] | None = None,
    verbose: int = 1,
) -> Any:
    """Calculate UMAP coordinates with ParametricUMAP based on PCA embeddings.

    This workflow trains a neural mapping from PCA coordinates to UMAP coordinates
    on a sampled subset of cells, then predicts coordinates for all cells in
    batches. It is intended for atlas-scale datasets where the classic
    ``umap.UMAP.transform`` step is too slow.

    Parameters
    ----------
    atlas
        Atlas object with an existing ``obsm_X_pca`` table.

    fit_sample_n
        Number of cells sampled from ``obsm_X_pca`` for ParametricUMAP training.
        The default is ``200_000``.

    transform_batch_size
        Number of cells read from ``obsm_X_pca`` for each full-dataset prediction
        batch.

    parametric_batch_size
        Keras training batch size used inside ParametricUMAP.

    training_epochs
        ParametricUMAP training epoch multiplier. In umap-learn, one unit here
        corresponds to ``loss_report_frequency`` Keras epochs, which is currently
        10 by default.

    n_components
        Number of UMAP dimensions. scAtlasPy currently writes ``umap1`` and
        ``umap2``, so this function requires ``n_components=2``.

    n_pcs
        Number of PCA dimensions used as input. If ``None``, all PC columns from
        ``obsm_X_pca`` are used.

    n_neighbors, min_dist, spread, metric, random_state, n_jobs
        Standard UMAP parameters passed to ParametricUMAP.

    add_table
        Output table for UMAP coordinates. The default is ``obsm_X_umap`` so that
        existing plotting functions can be used unchanged.

    save_params_table
        Table used to store the parameters of this run.

    eval_sample_n
        Number of training-sample cells used for trustworthiness and kNN-overlap
        evaluation.

    save_eval_table
        Table used to store evaluation metrics.

    keras_fit_kwargs
        Optional keyword arguments passed to the internal Keras ``fit`` call.
        Do not pass ``epochs`` here; use ``training_epochs`` instead.

    verbose
        Verbosity passed to ParametricUMAP and Keras prediction.

    Returns
    -------
    umap.parametric_umap.ParametricUMAP
        Fitted ParametricUMAP object.

    Notes
    -----
    TensorFlow is an optional dependency. It is imported only when this function
    is called. If TensorFlow is missing, scAtlasPy raises a clear ImportError
    instead of making TensorFlow a required package dependency.
    """

    start = datetime.now()
    conn = atlas.connection

    if n_components != 2:
        raise ValueError("parametric_umap currently requires n_components=2")

    if fit_sample_n <= 0:
        raise ValueError("fit_sample_n must be greater than 0")

    if transform_batch_size <= 0:
        raise ValueError("transform_batch_size must be greater than 0")

    if parametric_batch_size <= 0:
        raise ValueError("parametric_batch_size must be greater than 0")

    if training_epochs <= 0:
        raise ValueError("training_epochs must be greater than 0")

    if float(spread) < float(min_dist):
        raise ValueError(
            f"spread must be greater than or equal to min_dist; current spread={spread}, min_dist={min_dist}"
        )

    if keras_fit_kwargs is None:
        keras_fit_kwargs = {}
    else:
        keras_fit_kwargs = dict(keras_fit_kwargs)

    if "epochs" in keras_fit_kwargs:
        raise ValueError(
            "Do not pass keras_fit_kwargs['epochs']; use training_epochs instead."
        )

    keras_fit_kwargs.setdefault("verbose", verbose)

    ParametricUMAP = _import_parametric_umap()

    pc_cols = _get_pca_columns(atlas, n_pcs=n_pcs, input_table="obsm_X_pca")
    pc_cols_sql = ", ".join(pc_cols)

    total_n = conn.execute("SELECT COUNT(*) FROM obsm_X_pca").fetchone()[0]
    sample_n = min(int(fit_sample_n), int(total_n))
    seed = 0 if random_state is None else int(random_state)

    logger.info("[ParametricUMAP] input_table = obsm_X_pca")
    logger.info(f"[ParametricUMAP] output_table = {add_table}")
    logger.info(f"[ParametricUMAP] total cells = {total_n:,}")
    logger.info(f"[ParametricUMAP] fit_sample_n = {sample_n:,}")
    logger.info(f"[ParametricUMAP] n_pcs = {len(pc_cols)}")
    logger.info(f"[ParametricUMAP] transform_batch_size = {transform_batch_size:,}")
    logger.info(f"[ParametricUMAP] parametric_batch_size = {parametric_batch_size:,}")
    logger.info(f"[ParametricUMAP] training_epochs = {training_epochs}")

    fit_df = conn.execute(f"""
        SELECT atlas_cell_id, {pc_cols_sql}
        FROM obsm_X_pca
        ORDER BY hash(atlas_cell_id + {seed})
        LIMIT {sample_n}
    """).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("The sample used to fit ParametricUMAP is empty")

    X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

    reducer = ParametricUMAP(
        batch_size=int(parametric_batch_size),
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        keras_fit_kwargs=keras_fit_kwargs,
    )
    reducer.n_training_epochs = int(training_epochs)

    logger.info("[ParametricUMAP] fitting model")
    reducer.fit(X_fit)

    logger.info("[ParametricUMAP] evaluating training sample")
    X_fit_umap = reducer.transform(X_fit).astype(np.float32)

    eval_n = min(int(eval_sample_n), X_fit.shape[0])

    if eval_n >= 3:
        rng = np.random.default_rng(random_state)
        eval_idx = rng.choice(X_fit.shape[0], size=eval_n, replace=False)
        X_eval = X_fit[eval_idx]
        X_eval_umap = X_fit_umap[eval_idx]
        eval_k = min(n_neighbors, eval_n - 1)

        trustworthiness_score = trustworthiness(
            X_eval,
            X_eval_umap,
            n_neighbors=eval_k,
        )
        knn_overlap_score = knn_overlap(
            X_eval,
            X_eval_umap,
            k=eval_k,
        )
    else:
        eval_k = 0
        trustworthiness_score = np.nan
        knn_overlap_score = np.nan

    logger.info(f"[ParametricUMAP] trustworthiness = {trustworthiness_score:.4f}")
    logger.info(f"[ParametricUMAP] knn_overlap     = {knn_overlap_score:.4f}")

    conn.execute(f"DROP TABLE IF EXISTS {add_table}")
    conn.execute(f"""
        CREATE TABLE {add_table} (
            atlas_cell_id BIGINT,
            umap1 FLOAT,
            umap2 FLOAT
        )
    """)

    conn.execute(f"DROP TABLE IF EXISTS {save_params_table}")
    conn.execute(f"""
        CREATE TABLE {save_params_table} (
            param_name VARCHAR,
            param_value VARCHAR
        )
    """)

    params_df = pd.DataFrame({
        "param_name": [
            "method",
            "n_components",
            "n_pcs",
            "n_neighbors",
            "min_dist",
            "spread",
            "metric",
            "random_state",
            "fit_sample_n",
            "transform_batch_size",
            "parametric_batch_size",
            "training_epochs",
            "input_table",
            "output_table",
            "eval_sample_n",
        ],
        "param_value": [
            "parametric_umap",
            str(n_components),
            str(len(pc_cols)),
            str(n_neighbors),
            str(min_dist),
            str(spread),
            str(metric),
            str(random_state),
            str(sample_n),
            str(transform_batch_size),
            str(parametric_batch_size),
            str(training_epochs),
            "obsm_X_pca",
            add_table,
            str(eval_sample_n),
        ],
    })

    conn.append(save_params_table, params_df)

    conn.execute(f"DROP TABLE IF EXISTS {save_eval_table}")
    conn.execute(f"""
        CREATE TABLE {save_eval_table} (
            metric_name VARCHAR,
            metric_value DOUBLE
        )
    """)

    eval_df = pd.DataFrame({
        "metric_name": [
            "trustworthiness",
            "knn_overlap",
            "eval_sample_n",
            "eval_n_neighbors",
        ],
        "metric_value": [
            float(trustworthiness_score),
            float(knn_overlap_score),
            float(eval_n),
            float(eval_k),
        ],
    })

    conn.append(save_eval_table, eval_df)

    logger.info("[ParametricUMAP] predicting full dataset")
    written_n = 0
    last_cell_id = -1

    while written_n < total_n:
        batch_df = conn.execute(f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            WHERE atlas_cell_id > {int(last_cell_id)}
            ORDER BY atlas_cell_id
            LIMIT {int(transform_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        X_batch = batch_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)
        X_umap = reducer.transform(X_batch).astype(np.float32)

        out_df = pd.DataFrame({
            "atlas_cell_id": batch_df["atlas_cell_id"].to_numpy(dtype=np.int64),
            "umap1": X_umap[:, 0],
            "umap2": X_umap[:, 1],
        })

        conn.append(add_table, out_df)

        written_n += len(batch_df)
        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])

        logger.info(f"[ParametricUMAP] predicted {written_n:,}/{total_n:,}")

    logger.info(
        f"ParametricUMAP Done, elapsed time: {(datetime.now() - start).total_seconds():.2f} seconds"
    )

    return reducer
