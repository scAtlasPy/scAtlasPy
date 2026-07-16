from __future__ import annotations

from datetime import datetime
import importlib.abc
import sys
from typing import Any

import logging
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from ..data import Atlas

logger = logging.getLogger("Atlas")


class _BlockUMAPParametricImport(importlib.abc.MetaPathFinder):
    """Block umap-learn's optional ParametricUMAP import.

    ``umap.__init__`` imports the classic ``UMAP`` class first and then tries to
    import ``umap.parametric_umap``. If TensorFlow is installed, that optional
    import loads TensorFlow into the current process even when scAtlasPy only
    needs classic UMAP as the teacher. TensorFlow and PyTorch can conflict at
    the native-runtime level, so the PyTorch UMAP backend deliberately forces
    this optional import to fail while still using umap-learn's classic UMAP
    implementation.
    """

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        if fullname == "umap.parametric_umap":
            raise ImportError("scAtlasPy blocks umap.parametric_umap for the PyTorch UMAP backend")
        return None


def _import_classic_umap() -> Any:
    """Import umap-learn's classic UMAP without loading ParametricUMAP."""

    if "umap.parametric_umap" in sys.modules:
        raise RuntimeError(
            "umap.parametric_umap has already been imported in this process. "
            "Restart Python before calling sap.tl.umap with the PyTorch backend."
        )

    blocker = _BlockUMAPParametricImport()
    sys.meta_path.insert(0, blocker)

    try:
        from umap import UMAP
    finally:
        sys.meta_path.remove(blocker)

    return UMAP


def _import_torch() -> Any:
    """Import PyTorch only when the user calls the parametric UMAP workflow."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "sap.tl.umap requires PyTorch for the teacher-student UMAP backend. "
            "Install it in your analysis environment with "
            "`pip install 'scatlaspy[parametric]'` or install the PyTorch build "
            "that matches your GPU/accelerator."
        ) from exc

    return torch


def _resolve_torch_device(device: str, torch: Any) -> Any:
    """Resolve ``"auto"`` or a PyTorch device string to ``torch.device``."""

    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    """Calculate average nearest-neighbor overlap between two coordinate spaces."""

    n = X_high.shape[0]
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


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize a matrix and return the transformed data plus parameters."""

    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return ((X - mean) / scale).astype(np.float32), mean, scale


def _standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply stored standardization parameters."""

    return ((X - mean) / scale).astype(np.float32)


class _TorchUMAPRegressor:
    """Small PyTorch MLP that learns a PCA-to-UMAP mapping."""

    def __init__(
        self,
        torch: Any,
        n_input: int,
        n_output: int,
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

        layers.append(nn.Linear(current_dim, int(n_output)))
        self.model = nn.Sequential(*layers)


def _fit_torch_student(
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
    random_state: int | None,
    verbose: int,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train a PyTorch student model to reproduce teacher UMAP coordinates."""

    seed = 0 if random_state is None else int(random_state)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_scaled, x_mean, x_scale = _standardize_fit(X_fit)
    y_scaled, y_mean, y_scale = _standardize_fit(y_fit)

    x_tensor = torch.from_numpy(X_scaled)
    y_tensor = torch.from_numpy(y_scaled)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        pin_memory=(device.type == "cuda"),
    )

    student = _TorchUMAPRegressor(
        torch=torch,
        n_input=X_fit.shape[1],
        n_output=y_fit.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).model.to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = torch.nn.MSELoss()

    student.train()
    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = student(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            batch_n = xb.shape[0]
            epoch_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n

        if verbose:
            logger.info(
                f"[TorchUMAP] epoch {epoch + 1}/{int(epochs)}, "
                f"loss={epoch_loss / max(seen, 1):.6f}"
            )

    return student, x_mean, x_scale, y_mean, y_scale


def _predict_torch_student(
    X: np.ndarray,
    *,
    model: Any,
    torch: Any,
    device: Any,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict UMAP coordinates in batches with a trained PyTorch student."""

    model.eval()
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = min(start + int(batch_size), X.shape[0])
            xb = _standardize_apply(X[start:end], x_mean, x_scale)
            xb_tensor = torch.from_numpy(xb).to(device)
            pred = model(xb_tensor).detach().cpu().numpy().astype(np.float32)
            outputs.append(pred)

    y_scaled = np.vstack(outputs)
    return (y_scaled * y_scale + y_mean).astype(np.float32)


def umap(
    atlas: Atlas,
    fit_sample_n: int = 200_000,
    transform_batch_size: int = 100_000,
    torch_batch_size: int = 1024,
    torch_epochs: int = 20,
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
    verbose: int = 1,
    device: str = "auto",
    torch_hidden_dim: int = 512,
    torch_num_layers: int = 3,
    torch_learning_rate: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_dropout: float = 0.0,
) -> dict[str, Any]:
    """Calculate UMAP coordinates with a PyTorch teacher-student workflow.

    A standard UMAP model is fitted on sampled PCA coordinates and serves as the
    teacher. A small PyTorch MLP then learns the mapping from PCA coordinates to
    teacher UMAP coordinates. The student is used to transform the full Atlas in
    batches, avoiding graph optimization on every cell.
    """

    start = datetime.now()
    conn = atlas.connection

    if n_components != 2:
        raise ValueError("sap.tl.umap currently requires n_components=2")
    if fit_sample_n <= 0:
        raise ValueError("fit_sample_n must be greater than 0")
    if transform_batch_size <= 0:
        raise ValueError("transform_batch_size must be greater than 0")
    if torch_batch_size <= 0:
        raise ValueError("torch_batch_size must be greater than 0")
    if torch_hidden_dim <= 0:
        raise ValueError("torch_hidden_dim must be greater than 0")
    if torch_num_layers <= 0:
        raise ValueError("torch_num_layers must be greater than 0")
    if torch_learning_rate <= 0:
        raise ValueError("torch_learning_rate must be greater than 0")
    if float(spread) < float(min_dist):
        raise ValueError(
            f"spread must be greater than or equal to min_dist; current spread={spread}, min_dist={min_dist}"
        )

    if torch_epochs <= 0:
        raise ValueError("torch_epochs must be greater than 0")

    torch = _import_torch()
    torch_device = _resolve_torch_device(device, torch)
    logger.info(f"[TorchUMAP] resolved PyTorch device = {torch_device}")

    pc_cols = _get_pca_columns(atlas, n_pcs=n_pcs, input_table="obsm_X_pca")
    pc_cols_sql = ", ".join(pc_cols)

    total_n = conn.execute("SELECT COUNT(*) FROM obsm_X_pca").fetchone()[0]
    sample_n = min(int(fit_sample_n), int(total_n))
    seed = 0 if random_state is None else int(random_state)

    logger.info("[TorchUMAP] input_table = obsm_X_pca")
    logger.info(f"[TorchUMAP] output_table = {add_table}")
    logger.info(f"[TorchUMAP] total cells = {total_n:,}")
    logger.info(f"[TorchUMAP] fit_sample_n = {sample_n:,}")
    logger.info(f"[TorchUMAP] n_pcs = {len(pc_cols)}")
    logger.info(f"[TorchUMAP] transform_batch_size = {transform_batch_size:,}")
    logger.info(f"[TorchUMAP] torch_batch_size = {torch_batch_size:,}")
    logger.info(f"[TorchUMAP] torch_epochs = {torch_epochs}")
    logger.info(f"[TorchUMAP] requested device = {device}")

    fit_df = conn.execute(f"""
        SELECT atlas_cell_id, {pc_cols_sql}
        FROM obsm_X_pca
        ORDER BY hash(atlas_cell_id + {seed})
        LIMIT {sample_n}
    """).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("The sample used to fit UMAP is empty")

    X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

    logger.info("[TorchUMAP] fitting teacher UMAP")
    UMAP = _import_classic_umap()
    teacher = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=bool(verbose),
    )
    X_teacher = teacher.fit_transform(X_fit).astype(np.float32)

    logger.info("[TorchUMAP] fitting PyTorch student")
    student, x_mean, x_scale, y_mean, y_scale = _fit_torch_student(
        X_fit,
        X_teacher,
        torch=torch,
        device=torch_device,
        batch_size=torch_batch_size,
        epochs=torch_epochs,
        learning_rate=torch_learning_rate,
        weight_decay=torch_weight_decay,
        hidden_dim=torch_hidden_dim,
        num_layers=torch_num_layers,
        dropout=torch_dropout,
        random_state=random_state,
        verbose=verbose,
    )

    logger.info("[TorchUMAP] evaluating training sample")
    X_fit_umap = _predict_torch_student(
        X_fit,
        model=student,
        torch=torch,
        device=torch_device,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        batch_size=transform_batch_size,
    )

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

    teacher_mse = float(np.mean((X_fit_umap - X_teacher) ** 2))
    logger.info(f"[TorchUMAP] trustworthiness = {trustworthiness_score:.4f}")
    logger.info(f"[TorchUMAP] knn_overlap     = {knn_overlap_score:.4f}")
    logger.info(f"[TorchUMAP] teacher_mse     = {teacher_mse:.6f}")

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
            "teacher",
            "student",
            "n_components",
            "n_pcs",
            "n_neighbors",
            "min_dist",
            "spread",
            "metric",
            "random_state",
            "fit_sample_n",
            "transform_batch_size",
            "torch_batch_size",
            "torch_epochs",
            "torch_hidden_dim",
            "torch_num_layers",
            "torch_learning_rate",
            "torch_weight_decay",
            "torch_dropout",
            "device",
            "input_table",
            "output_table",
            "eval_sample_n",
        ],
        "param_value": [
            "torch_teacher_student_umap",
            "umap-learn",
            "pytorch_mlp",
            str(n_components),
            str(len(pc_cols)),
            str(n_neighbors),
            str(min_dist),
            str(spread),
            str(metric),
            str(random_state),
            str(sample_n),
            str(transform_batch_size),
            str(torch_batch_size),
            str(torch_epochs),
            str(torch_hidden_dim),
            str(torch_num_layers),
            str(torch_learning_rate),
            str(torch_weight_decay),
            str(torch_dropout),
            str(torch_device),
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
            "teacher_mse",
            "eval_sample_n",
            "eval_n_neighbors",
        ],
        "metric_value": [
            float(trustworthiness_score),
            float(knn_overlap_score),
            teacher_mse,
            float(eval_n),
            float(eval_k),
        ],
    })
    conn.append(save_eval_table, eval_df)

    logger.info("[TorchUMAP] predicting full dataset")
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
        X_umap = _predict_torch_student(
            X_batch,
            model=student,
            torch=torch,
            device=torch_device,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            batch_size=transform_batch_size,
        )

        out_df = pd.DataFrame({
            "atlas_cell_id": batch_df["atlas_cell_id"].to_numpy(dtype=np.int64),
            "umap1": X_umap[:, 0],
            "umap2": X_umap[:, 1],
        })
        conn.append(add_table, out_df)

        written_n += len(batch_df)
        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])

        logger.info(f"[TorchUMAP] predicted {written_n:,}/{total_n:,}")

    logger.info(
        f"TorchUMAP Done, elapsed time: {(datetime.now() - start).total_seconds():.2f} seconds"
    )

    return {
        "teacher": teacher,
        "student": student,
        "device": str(torch_device),
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }
