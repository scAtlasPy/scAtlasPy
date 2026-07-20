from __future__ import annotations

from datetime import datetime
from typing import Any

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import kneighbors_graph

from ..data import Atlas, duckdb_memory_limit
from ._umap import (
    _get_pca_columns,
    _import_torch,
    _resolve_torch_device,
    _standardize_apply,
    _standardize_fit,
)

logger = logging.getLogger("Atlas")


def _q(name: str) -> str:
    """Quote a DuckDB identifier."""

    return '"' + name.replace('"', '""') + '"'


def _import_louvain_backend() -> tuple[Any, Any]:
    """Import igraph and louvain only when the graph teacher is fitted."""

    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError(
            "sap.tl.graph_clustering(mode='distilled_louvain') requires "
            "`igraph` to fit the Louvain teacher."
        ) from exc

    try:
        import louvain
    except ImportError as exc:
        if getattr(exc, "name", None) == "pkg_resources":
            raise ImportError(
                "The installed `louvain` package imports the legacy "
                "`pkg_resources` module, which is not available in recent "
                "setuptools releases. Please install `setuptools<81`, for "
                "example: `python -m pip install 'setuptools<81'`."
            ) from exc
        raise ImportError(
            "sap.tl.graph_clustering(mode='distilled_louvain') requires "
            "`louvain` to fit the Louvain teacher."
        ) from exc

    return ig, louvain


class _TorchClassifier:
    """Small PyTorch MLP classifier used to distill graph-clustering labels."""

    def __init__(
        self,
        torch: Any,
        n_input: int,
        n_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        nn = torch.nn
        layers: list[Any] = []
        current_dim = int(n_input)

        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)

        layers.append(nn.Linear(current_dim, int(n_classes)))
        self.model = nn.Sequential(*layers)


def _fit_louvain_teacher(
    X_fit: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float,
    random_state: int,
    metric: str,
) -> np.ndarray:
    """Run the Louvain teacher in PCA space and return integer labels."""

    ig, louvain = _import_louvain_backend()

    graph_csr = kneighbors_graph(
        X_fit,
        n_neighbors=int(n_neighbors),
        mode="connectivity",
        metric=metric,
        include_self=False,
        n_jobs=1,
    )
    graph_csr = graph_csr.maximum(graph_csr.T).tocoo()
    keep = graph_csr.row < graph_csr.col
    edge_rows = graph_csr.row[keep]
    edge_cols = graph_csr.col[keep]
    edge_weights = graph_csr.data[keep]

    edges = list(zip(edge_rows.tolist(), edge_cols.tolist()))
    graph = ig.Graph(
        n=X_fit.shape[0],
        edges=edges,
        directed=False,
    )
    graph.es["weight"] = edge_weights.astype(float).tolist()

    partition = louvain.find_partition(
        graph,
        louvain.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(random_state),
    )
    return np.asarray(partition.membership, dtype=np.int64)


def _fit_torch_classifier(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    torch: Any,
    device: Any,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    validation_fraction: float,
    random_state: int,
    verbose: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, float]]:
    """Train a PyTorch classifier and report held-out teacher-label metrics."""

    seed = int(random_state)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_scaled, x_mean, x_scale = _standardize_fit(X_fit)
    n = X_scaled.shape[0]
    n_classes = int(np.max(y_fit)) + 1

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    validation_n = int(round(float(validation_fraction) * n))
    validation_n = min(max(validation_n, 0), n - 1)

    if validation_n > 0:
        val_idx = order[:validation_n]
        train_idx = order[validation_n:]
    else:
        val_idx = np.array([], dtype=np.int64)
        train_idx = order

    x_train = torch.from_numpy(X_scaled[train_idx])
    y_train = torch.from_numpy(y_fit[train_idx].astype(np.int64, copy=False))
    dataset = torch.utils.data.TensorDataset(x_train, y_train)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )

    classifier = _TorchClassifier(
        torch=torch,
        n_input=X_scaled.shape[1],
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = classifier.model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(int(epochs)):
        total_loss = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]

        if verbose:
            mean_loss = total_loss / max(seen, 1)
            logger.debug(f"[DistilledLouvain] epoch {epoch + 1}/{epochs}, loss={mean_loss:.4f}")

    metrics: dict[str, float] = {
        "validation_n": float(validation_n),
        "n_classes": float(n_classes),
    }

    if validation_n > 0:
        y_pred = _predict_torch_classifier(
            X_fit[val_idx],
            model=model,
            torch=torch,
            device=device,
            x_mean=x_mean,
            x_scale=x_scale,
            batch_size=batch_size,
        )
        y_true = y_fit[val_idx]
        metrics["validation_accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["validation_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["validation_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    else:
        metrics["validation_accuracy"] = float("nan")
        metrics["validation_balanced_accuracy"] = float("nan")
        metrics["validation_macro_f1"] = float("nan")

    return model, x_mean, x_scale, metrics


def _predict_torch_classifier(
    X: np.ndarray,
    *,
    model: Any,
    torch: Any,
    device: Any,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict integer cluster labels for a dense coordinate matrix."""

    X_scaled = _standardize_apply(X, x_mean, x_scale)
    labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, X_scaled.shape[0], int(batch_size)):
            xb = torch.from_numpy(X_scaled[start:start + int(batch_size)]).to(device)
            pred = torch.argmax(model(xb), dim=1).detach().cpu().numpy().astype(np.int32)
            labels.append(pred)

    return np.concatenate(labels) if labels else np.empty(0, dtype=np.int32)


@duckdb_memory_limit("5G")
def graph_clustering(
    atlas: Atlas,
    mode: str = "distilled_louvain",
    input_table: str = "obsm_X_pca",
    n_pcs: int | None = None,
    fit_sample_n: int = 200_000,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "euclidean",
    random_state: int = 42,
    transform_batch_size: int = 100_000,
    torch_batch_size: int = 1024,
    torch_epochs: int = 50,
    device: str = "auto",
    torch_hidden_dim: int = 512,
    torch_num_layers: int = 3,
    torch_learning_rate: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_dropout: float = 0.0,
    validation_fraction: float = 0.1,
    add_obs_col: str = "scatlas_cluster",
    use_cluster_table: str = "obs_cluster_distilled_louvain",
    save_params_table: str = "uns_graph_clustering_params",
    save_eval_table: str = "uns_graph_clustering_eval",
    verbose: int = 1,
) -> dict[str, Any]:
    """Cluster an Atlas with a graph teacher and streaming classifier.

    ``mode="distilled_louvain"`` fits a Louvain graph-clustering teacher in PCA
    space, trains a PyTorch classifier to reproduce the teacher labels, and
    predicts labels for the full Atlas in batches. This provides a graph-like
    clustering signal without building a full atlas-scale graph.
    """

    if mode != "distilled_louvain":
        raise ValueError("graph_clustering currently only supports mode='distilled_louvain'")
    if fit_sample_n <= 0:
        raise ValueError("fit_sample_n must be greater than 0")
    if transform_batch_size <= 0:
        raise ValueError("transform_batch_size must be greater than 0")
    if torch_batch_size <= 0:
        raise ValueError("torch_batch_size must be greater than 0")
    if torch_epochs <= 0:
        raise ValueError("torch_epochs must be greater than 0")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    start = datetime.now()
    conn = atlas.connection
    seed = int(random_state)

    torch = _import_torch(
        caller="sap.tl.graph_clustering",
        purpose="the distilled Louvain student model",
        fallback=(
            "If you cannot install PyTorch in this environment, run "
            "`sap.tl.kmeans(atlas, use_rep='X_umap')` after UMAP, or "
            "`sap.tl.kmeans(atlas, use_rep='X_pca')` after PCA."
        ),
    )
    torch_device = _resolve_torch_device(device, torch)

    pc_cols = _get_pca_columns(atlas, n_pcs=n_pcs, input_table=input_table)
    pc_cols_sql = ", ".join(_q(c) for c in pc_cols)

    total_n = conn.execute(f"SELECT COUNT(*) FROM {_q(input_table)}").fetchone()[0]
    sample_n = min(int(fit_sample_n), int(total_n))

    logger.info(
        f"[DistilledLouvain] cells={total_n:,}, fit_sample_n={sample_n:,}, "
        f"n_pcs={len(pc_cols)}, resolution={resolution}, device={torch_device}"
    )
    logger.debug("[DistilledLouvain] mode = distilled_louvain")
    logger.debug(f"[DistilledLouvain] input_table = {input_table}")
    logger.debug(f"[DistilledLouvain] total cells = {total_n:,}")
    logger.debug(f"[DistilledLouvain] fit_sample_n = {sample_n:,}")
    logger.debug(f"[DistilledLouvain] n_pcs = {len(pc_cols)}")
    logger.debug(f"[DistilledLouvain] n_neighbors = {n_neighbors}")
    logger.debug(f"[DistilledLouvain] resolution = {resolution}")
    logger.debug(f"[DistilledLouvain] resolved PyTorch device = {torch_device}")

    fit_df = conn.execute(f"""
        SELECT atlas_cell_id, {pc_cols_sql}
        FROM {_q(input_table)}
        ORDER BY hash(atlas_cell_id + {seed})
        LIMIT {sample_n}
    """).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("The teacher fitting data for distilled Louvain is empty")

    X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

    logger.info("[DistilledLouvain] fitting Louvain teacher")
    y_teacher = _fit_louvain_teacher(
        X_fit,
        n_neighbors=n_neighbors,
        resolution=resolution,
        random_state=seed,
        metric=metric,
    )
    n_classes = int(np.max(y_teacher)) + 1
    logger.info(f"[DistilledLouvain] teacher clusters = {n_classes}")

    logger.info("[DistilledLouvain] fitting PyTorch classifier")
    model, x_mean, x_scale, metrics = _fit_torch_classifier(
        X_fit,
        y_teacher,
        torch=torch,
        device=torch_device,
        batch_size=torch_batch_size,
        epochs=torch_epochs,
        learning_rate=torch_learning_rate,
        weight_decay=torch_weight_decay,
        hidden_dim=torch_hidden_dim,
        num_layers=torch_num_layers,
        dropout=torch_dropout,
        validation_fraction=validation_fraction,
        random_state=seed,
        verbose=verbose,
    )

    logger.info(
        "[DistilledLouvain] validation: "
        f"accuracy={metrics['validation_accuracy']:.4f}, "
        f"balanced_accuracy={metrics['validation_balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['validation_macro_f1']:.4f}"
    )

    conn.execute(f"DROP TABLE IF EXISTS {_q(use_cluster_table)}")
    conn.execute(f"""
        CREATE TABLE {_q(use_cluster_table)} (
            atlas_cell_id BIGINT,
            cluster_id INTEGER
        )
    """)

    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    if add_obs_col not in obs_cols:
        conn.execute(f"ALTER TABLE obs ADD COLUMN {_q(add_obs_col)} INTEGER")
    conn.execute(f"UPDATE obs SET {_q(add_obs_col)} = NULL")

    conn.execute(f"DROP TABLE IF EXISTS {_q(save_params_table)}")
    conn.execute(f"""
        CREATE TABLE {_q(save_params_table)} (
            param_name VARCHAR,
            param_value VARCHAR
        )
    """)
    params_df = pd.DataFrame({
        "param_name": [
            "mode",
            "teacher",
            "student",
            "input_table",
            "n_pcs",
            "fit_sample_n",
            "n_neighbors",
            "resolution",
            "metric",
            "random_state",
            "transform_batch_size",
            "torch_batch_size",
            "torch_epochs",
            "torch_hidden_dim",
            "torch_num_layers",
            "torch_learning_rate",
            "torch_weight_decay",
            "torch_dropout",
            "device",
            "add_obs_col",
        ],
        "param_value": [
            mode,
            "louvain_teacher",
            "pytorch_mlp_classifier",
            input_table,
            str(len(pc_cols)),
            str(sample_n),
            str(n_neighbors),
            str(resolution),
            metric,
            str(seed),
            str(transform_batch_size),
            str(torch_batch_size),
            str(torch_epochs),
            str(torch_hidden_dim),
            str(torch_num_layers),
            str(torch_learning_rate),
            str(torch_weight_decay),
            str(torch_dropout),
            str(torch_device),
            add_obs_col,
        ],
    })
    conn.append(save_params_table, params_df)

    conn.execute(f"DROP TABLE IF EXISTS {_q(save_eval_table)}")
    conn.execute(f"""
        CREATE TABLE {_q(save_eval_table)} (
            metric_name VARCHAR,
            metric_value DOUBLE
        )
    """)
    eval_df = pd.DataFrame({
        "metric_name": list(metrics.keys()),
        "metric_value": list(metrics.values()),
    })
    conn.append(save_eval_table, eval_df)

    logger.info("[DistilledLouvain] predicting full dataset")
    written_n = 0
    last_cell_id = -1

    while written_n < total_n:
        batch_df = conn.execute(f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM {_q(input_table)}
            WHERE atlas_cell_id > {int(last_cell_id)}
            ORDER BY atlas_cell_id
            LIMIT {int(transform_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        X_batch = batch_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)
        labels = _predict_torch_classifier(
            X_batch,
            model=model,
            torch=torch,
            device=torch_device,
            x_mean=x_mean,
            x_scale=x_scale,
            batch_size=torch_batch_size,
        )

        out_df = pd.DataFrame({
            "atlas_cell_id": batch_df["atlas_cell_id"].to_numpy(dtype=np.int64),
            "cluster_id": labels,
        })
        conn.append(use_cluster_table, out_df)

        conn.register("_graph_cluster_batch_tmp", out_df)
        conn.execute(f"""
            UPDATE obs
            SET {_q(add_obs_col)} = t.cluster_id
            FROM _graph_cluster_batch_tmp t
            WHERE obs.atlas_cell_id = t.atlas_cell_id
        """)
        conn.unregister("_graph_cluster_batch_tmp")

        written_n += len(batch_df)
        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        logger.debug(f"[DistilledLouvain] predicted {written_n:,}/{total_n:,}")

    elapsed_sec = (datetime.now() - start).total_seconds()
    logger.info(f"[DistilledLouvain] done, elapsed time = {elapsed_sec:.2f} seconds")

    return {
        "mode": mode,
        "n_cells": int(total_n),
        "fit_sample_n": int(sample_n),
        "n_clusters": int(n_classes),
        "elapsed_sec": float(elapsed_sec),
        **metrics,
    }
