from ..data import Atlas
import numpy as np
from ..io import progress
import pandas as pd
import time
import logging
logger = logging.getLogger('Atlas')


class StreamingRandomizedPCA:
    """Streaming randomized PCA backend for Atlas minibatch data.

    This class implements a randomized eigensolver for the covariance matrix
    without materializing the full cell-by-gene matrix or full projected matrix.
    It reads dense minibatches from ``atlas.get_minibatch_dense`` and assumes
    that upstream preprocessing and the read index have already produced the
    desired centered representation.

    During fitting it keeps only feature-space matrices:

    - ``H = X.T @ X @ Omega`` with shape ``(n_features, projection_dim)``;
    - ``T = Q.T @ X.T @ X @ Q`` with shape ``(projection_dim, projection_dim)``.

    Optional subspace iterations replace ``H`` by repeated applications of
    ``X.T @ X``. The final PCA components are obtained from the eigendecomposition
    of the compact covariance matrix ``T``. Transform is then performed by
    another streaming pass using ``X_batch @ components_.T``.
    """

    def __init__(
        self,
        n_components: int = 30,
        oversample: int = 200,
        batch_size: int = 2048,
        random_state: int | None = 0,
        n_iter: int = 2,
    ):
        """Initialize the streaming randomized PCA backend.

        Parameters
        ----------
        n_components
            Number of principal components to retain.

        oversample
            Additional random projection dimensions. The internal projection
            dimension is ``n_components + oversample``.

        batch_size
            Number of cells per dense minibatch.

        random_state
            Seed used to generate the Gaussian random projection matrix.

        n_iter
            Number of covariance subspace iterations. ``0`` uses one pass to
            form ``X.T @ X @ Omega``. Larger values improve accuracy for slowly
            decaying spectra at the cost of one extra full scan per iteration.
        """

        if n_components <= 0:
            raise ValueError("n_components must be greater than 0")
        if oversample < 0:
            raise ValueError("oversample must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative")

        self.n_components = int(n_components)
        self.oversample = int(oversample)
        self.projection_dim = self.n_components + self.oversample
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.n_iter = int(n_iter)

        self.omega_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.n_samples_seen_: int = 0
        self.total_sum_squares_: float = 0.0
        self.fit_scan_time_: float = 0.0
        self.projection_time_: float = 0.0
        self.decomposition_time_: float = 0.0
        self.transform_scan_time_: float = 0.0
        self.number_of_fit_passes_: int = 0

    def _create_pca_table(self, atlas: Atlas, n_components: int = 30, table_name: str = "obsm_X_pca") -> None:
        """Create the PCA embedding table."""

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)
        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])
        atlas.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                atlas_cell_id INTEGER,
                {cols}
            );
        """)

    def _create_pcs_table(self, atlas: Atlas, n_components: int = 30, table_name: str = "varm_PCs") -> None:
        """Create the PCA loading table."""

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)
        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])
        atlas.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                atlas_gene_id USMALLINT,
                {cols}
            );
        """)

    def _create_pca_stats_table(self, atlas: Atlas, table_name: str = "uns_pca_stats") -> None:
        """Create the PCA explained-variance table."""

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)
        atlas.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                pc_index USMALLINT,
                variance REAL,
                variance_ratio REAL
            );
        """)

    def _writer_obsm_x_pca(
        self,
        atlas: Atlas,
        X_batch: np.ndarray,
        atlas_cell_ids: np.ndarray,
        table_name: str = "obsm_X_pca",
    ) -> None:
        """Write one PCA embedding batch."""

        n = X_batch.shape[0]
        atlas_cell_ids = np.asarray(atlas_cell_ids, dtype=np.int32)
        if atlas_cell_ids.shape[0] != n:
            raise ValueError(
                "atlas_cell_ids length must match the number of PCA rows: "
                f"{atlas_cell_ids.shape[0]} != {n}"
            )

        df = pd.DataFrame(
            X_batch.astype(np.float32, copy=False),
            columns=[f"pc{i}" for i in range(X_batch.shape[1])],
        )
        df.insert(0, "atlas_cell_id", atlas_cell_ids)
        atlas.connection.append(table_name, df)

    def _writer_varm_pcs(self, atlas: Atlas, table_name: str = "varm_PCs") -> None:
        """Write PCA components to ``varm_PCs``."""

        if self.components_ is None:
            raise RuntimeError("components_ is None. Please run fit first.")

        pcs = self.components_.T.astype(np.float32)
        df = pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(pcs.shape[1])])
        df.insert(0, "atlas_gene_id", np.arange(pcs.shape[0], dtype=np.int32))
        atlas.connection.append(table_name, df)

    def _writer_uns_pca_stats(self, atlas: Atlas, table_name: str = "uns_pca_stats") -> None:
        """Write explained variance statistics."""

        if self.explained_variance_ is None or self.explained_variance_ratio_ is None:
            raise RuntimeError("Explained variance is missing. Please run fit first.")

        df = pd.DataFrame({
            "pc_index": np.arange(len(self.explained_variance_), dtype=np.int32),
            "variance": self.explained_variance_.astype(np.float32),
            "variance_ratio": self.explained_variance_ratio_.astype(np.float32),
        })
        atlas.connection.append(table_name, df)

    def fit(self, atlas: Atlas) -> "StreamingRandomizedPCA":
        """Fit randomized PCA by streaming covariance products."""

        rng = np.random.default_rng(self.random_state)
        H: np.ndarray | None = None
        total_sum_squares = 0.0
        n_samples = 0
        n_features: int | None = None

        fit_start = time.time()
        projection_time = 0.0

        input_matrix: np.ndarray | None = None

        for pass_id in range(self.n_iter + 1):
            H = None
            pass_samples = 0

            for batch_id, X_batch in enumerate(
                progress(
                    atlas.get_minibatch_dense(
                        batch_size=self.batch_size,
                        pass_mode="single-pass",
                    ),
                    desc=f"Randomized PCA covariance pass {pass_id + 1}",
                )
            ):
                X_batch = np.asarray(X_batch, dtype=np.float32)

                if n_features is None:
                    n_features = X_batch.shape[1]
                    if self.n_components > n_features:
                        raise ValueError(
                            "n_components cannot exceed the number of features: "
                            f"{self.n_components} > {n_features}"
                        )
                    effective_projection_dim = min(self.projection_dim, n_features)
                    if effective_projection_dim < self.projection_dim:
                        logger.info(
                            "[RandomizedPCA] projection dimension capped by feature count: "
                            f"{self.projection_dim} -> {effective_projection_dim}"
                        )
                    self.omega_ = rng.standard_normal(
                        size=(n_features, effective_projection_dim),
                        dtype=np.float64,
                    )
                    input_matrix = self.omega_

                if input_matrix is None or n_features is None:
                    raise RuntimeError("[RandomizedPCA] Projection matrix was not initialized")

                if H is None:
                    H = np.zeros((n_features, input_matrix.shape[1]), dtype=np.float64)

                X64 = X_batch.astype(np.float64, copy=False)

                t0 = time.time()
                projected = X64 @ input_matrix
                projection_time += time.time() - t0

                H += X64.T @ projected
                pass_samples += X_batch.shape[0]

                if pass_id == 0:
                    total_sum_squares += float(np.sum(X64 ** 2))
                    n_samples += X_batch.shape[0]

                if (batch_id + 1) % 20 == 0:
                    logger.info(
                        f"[RandomizedPCA] covariance pass={pass_id + 1}/{self.n_iter + 1}, "
                        f"batches={batch_id + 1}, samples={pass_samples:,}"
                    )

            if H is None:
                raise RuntimeError("[RandomizedPCA] No minibatch was obtained, so PCA cannot be trained")

            input_matrix, _ = np.linalg.qr(H, mode="reduced")
            self.number_of_fit_passes_ += 1

        if n_samples == 0 or H is None or input_matrix is None:
            raise RuntimeError("[RandomizedPCA] No minibatch was obtained, so PCA cannot be trained")

        self.fit_scan_time_ = time.time() - fit_start
        self.projection_time_ = projection_time
        self.n_samples_seen_ = n_samples
        self.total_sum_squares_ = total_sum_squares

        decomp_start = time.time()

        Q = input_matrix
        T = np.zeros((Q.shape[1], Q.shape[1]), dtype=np.float64)

        for batch_id, X_batch in enumerate(
            progress(
                atlas.get_minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="single-pass",
                ),
                desc="Randomized PCA compact covariance",
            )
        ):
            X64 = np.asarray(X_batch, dtype=np.float64)
            t0 = time.time()
            Z_batch = X64 @ Q
            projection_time += time.time() - t0
            T += Z_batch.T @ Z_batch

            if (batch_id + 1) % 20 == 0:
                logger.info(
                    f"[RandomizedPCA] compact covariance batches={batch_id + 1}"
                )

        self.number_of_fit_passes_ += 1
        self.projection_time_ = projection_time

        eigvals, eigvecs = np.linalg.eigh(T)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eps = np.finfo(np.float64).eps
        keep = eigvals > eps * max(float(eigvals[0]), 1.0)
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]

        if eigvals.size < self.n_components:
            raise RuntimeError(
                "The randomized projection rank is smaller than n_components. "
                "Try increasing oversample or checking the input data."
            )

        components = (Q @ eigvecs[:, :self.n_components]).T
        self.components_ = components.astype(np.float32)
        self.singular_values_ = np.sqrt(eigvals[:self.n_components]).astype(np.float32)

        denom = max(n_samples - 1, 1)
        explained_variance = eigvals[:self.n_components] / denom
        total_variance = total_sum_squares / denom
        self.explained_variance_ = explained_variance.astype(np.float32)
        self.explained_variance_ratio_ = (explained_variance / total_variance).astype(np.float32)
        self.decomposition_time_ = time.time() - decomp_start

        logger.info(
            "[RandomizedPCA] cumulative explained variance ratio: "
            f"{self.explained_variance_ratio_.sum():.4f}"
        )
        logger.info(f"[RandomizedPCA] fit scan time = {self.fit_scan_time_:.2f} seconds")
        logger.info(f"[RandomizedPCA] projection time = {self.projection_time_:.2f} seconds")
        logger.info(f"[RandomizedPCA] decomposition time = {self.decomposition_time_:.2f} seconds")
        logger.info(f"[RandomizedPCA] fit passes = {self.number_of_fit_passes_}")

        return self

    def transform(self, atlas: Atlas) -> None:
        """Project all cells by streaming ``X_batch @ components_.T``."""

        if self.components_ is None:
            raise RuntimeError("components_ is None. Please run fit first.")

        start = time.time()

        for batch in progress(
            atlas.get_minibatch_dense(
                batch_size=self.batch_size,
                pass_mode="single-pass",
                get_obs_col="atlas_cell_id",
            ),
            desc="Randomized PCA transform",
        ):
            X_batch = batch["X"]
            atlas_cell_ids = batch["atlas_cell_id"]
            X_pca = np.asarray(X_batch, dtype=np.float32) @ self.components_.T
            self._writer_obsm_x_pca(atlas, X_pca, atlas_cell_ids)

        self.transform_scan_time_ = time.time() - start
        logger.info(f"[RandomizedPCA] transform scan time = {self.transform_scan_time_:.2f} seconds")

    def fit_transform(self, atlas: Atlas) -> "StreamingRandomizedPCA":
        """Fit components and write full-cell PCA embeddings."""

        self.fit(atlas)
        self._writer_varm_pcs(atlas)
        self._writer_uns_pca_stats(atlas)
        self.transform(atlas)
        return self

    def run(self, atlas: Atlas) -> None:
        """Run the full randomized PCA workflow and write standard PCA tables."""

        self._create_pca_table(atlas, n_components=self.n_components)
        self._create_pcs_table(atlas, n_components=self.n_components)
        self._create_pca_stats_table(atlas)
        self.fit_transform(atlas)


def pca(
    atlas: Atlas,
    n_components: int = 30,
    batch_size: int = 2048,
    oversample: int = 200,
    random_state: int | None = 0,
    n_iter: int = 2,
) -> None:
    """Calculate PCA based on the Atlas expression matrix.

    This function is the public PCA entry point of scAtlasPy. It reads the
    expression matrix in batches from Atlas's dense minibatch reading interface,
    fits principal components with the selected backend, then projects all cells
    into PCA space and writes the results to the database.

    After running, the following result tables are generated or overwritten:

    - ``obsm_X_pca``: cell PCA coordinates;
    - ``varm_PCs``: gene PCA loadings;
    - ``uns_pca_stats``: PCA variance and variance explanation ratios.

    This workflow is similar to Scanpy's ``sc.tl.pca``, but to support large-scale
    data, both training and projection are completed through minibatch chunks.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and an index available for dense minibatch reading must already have been
        built through ``atlas.build_read_index(...)``.

    n_components
        Number of PCA principal components to calculate and save.
        The default value is ``30``.

    batch_size
        Number of cells contained in each minibatch. A larger value usually gives
        higher throughput, but increases per-batch memory usage.

    oversample
        Additional random projection dimensions. The internal projection
        dimension is ``n_components + oversample``.

    random_state
        Random seed used to generate the Gaussian projection matrix.

    n_iter
        Number of covariance subspace iterations. The default value is ``2``.

    Returns
    -------
    None
        PCA results are written directly to the Atlas database. No object is
        returned.

    Notes
    -----
    If you want PCA to run only on highly variable genes or filtered genes, you
    need to build the corresponding read index through ``atlas.build_read_index``
    before calling this function.

    Examples
    --------
    Calculate PCA on the default read index::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas)
    """

    t_start = time.time()

    pca_runner = StreamingRandomizedPCA(
        n_components=n_components,
        oversample=oversample,
        batch_size=batch_size,
        random_state=random_state,
        n_iter=n_iter,
    )

    pca_runner.run(atlas)

    t_end = time.time()
    logger.info(f" PCA Done, total time = {t_end - t_start:.2f} seconds")
