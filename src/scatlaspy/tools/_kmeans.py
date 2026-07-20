from sklearn.cluster import MiniBatchKMeans
from ..io import progress
from ..data import Atlas, duckdb_memory_limit
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger('Atlas')


def _q(name: str) -> str:
    """Quote a DuckDB identifier."""

    return '"' + name.replace('"', '""') + '"'


def _rep_to_obsm_table(use_rep: str) -> str:
    """Map a representation name such as ``X_umap`` to an ``obsm_*`` table."""

    if not isinstance(use_rep, str) or not use_rep:
        raise ValueError("use_rep must be a non-empty string")
    if use_rep.startswith("obsm_"):
        return use_rep
    if use_rep.startswith("X_"):
        return f"obsm_{use_rep}"
    raise ValueError(
        "use_rep must be an obsm representation name such as 'X_umap', "
        "'X_pca', 'obsm_X_umap', or 'obsm_X_pca'"
    )

class StreamingKMeans:

    """Streaming MiniBatchKMeans clusterer based on an ``obsm`` representation.

    This class reads a low-dimensional representation such as ``obsm_X_umap`` or
    ``obsm_X_pca`` in batches and uses sklearn ``MiniBatchKMeans`` for training
    and prediction.

    The complete workflow consists of two steps:

    - ``fit_kmeans``: reads representation minibatches and trains KMeans centers
      with ``partial_fit``;
    - ``predict_kmeans``: reads all representation minibatches again, predicts
      the cluster for each cell, and writes the results to an independent result
      table and an optional ``obs`` column.

    This class is the underlying implementation of ``sap.tl.kmeans``. Regular
    users usually call the public function ``kmeans`` directly.

    Parameters
    ----------
    use_rep
        Representation used for clustering. ``"X_umap"`` uses ``obsm_X_umap``;
        ``"X_pca"`` uses ``obsm_X_pca``.
    n_components
        Number of representation dimensions to use. If the selected
        representation has fewer dimensions, all available dimensions are used.
    n_clusters
        Number of K-means clusters.
    batch_size
        Number of cells in each minibatch.
    fit_batches
        Maximum number of minibatches used to train MiniBatchKMeans.

    Notes
    -----
    The selected ``obsm`` table must exist before running this class. The
    default ``use_rep="X_umap"`` expects ``sap.tl.umap(atlas)`` to have been run.

    Examples
    --------
    Recommended public API usage::

        sap.tl.umap(atlas)
        sap.tl.kmeans(atlas, n_clusters=20)
    """

    def __init__(
        self,
        use_rep: str = "X_umap",
        n_components: int = 30,
        n_clusters: int = 2,
        batch_size: int = 2048,
        fit_batches: int = 1000,
    ):
        """Initialize the streaming MiniBatchKMeans clusterer.

        This method stores the representation, number of clusters, and minibatch
        parameters, and creates the sklearn ``MiniBatchKMeans`` model.

        Parameters
        ----------
        use_rep
            Representation used for clustering. ``"X_umap"`` resolves to
            ``obsm_X_umap``.

        n_components
            Number of representation dimensions to use.

        n_clusters
            Number of KMeans clusters.

        batch_size
            Number of cells in each minibatch.

            A larger value usually trains faster, but increases the memory usage
            of projection and clustering for each batch.

        fit_batches
            Maximum number of minibatches used to train MiniBatchKMeans.

        Notes
        -----
        This object only initializes the model and parameters. It does not read
        the database or write cluster labels immediately.
        """

        self.use_rep = use_rep
        self.rep_table = _rep_to_obsm_table(use_rep)
        self.rep_columns: list[str] | None = None
        self.n_components = n_components
        self.n_clusters = n_clusters  # Target number of clusters
        self.batch_size = batch_size
        self.fit_batches = fit_batches  # Number of minibatches used for KMeans training
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init="k-means++",
            n_init="auto",
        )

    def _resolve_rep_columns(self, atlas: Atlas) -> list[str]:
        """Resolve value columns in the selected ``obsm`` table."""

        conn = atlas.connection
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
        """, [self.rep_table]).fetchone()[0]
        if table_exists == 0:
            raise ValueError(
                f"{self.rep_table} does not exist in the database. "
                f"Please run the workflow step that creates use_rep={self.use_rep!r} "
                "before running sap.tl.kmeans(atlas)."
            )

        table_columns = [
            row[0]
            for row in conn.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
            """, [self.rep_table]).fetchall()
        ]
        if "atlas_cell_id" not in table_columns:
            raise ValueError(f"The atlas_cell_id field does not exist in {self.rep_table}")

        value_columns = [c for c in table_columns if c != "atlas_cell_id"]
        if not value_columns:
            raise ValueError(f"{self.rep_table} has no representation value columns")

        if self.n_components is not None and self.n_components > 0:
            value_columns = value_columns[: min(int(self.n_components), len(value_columns))]

        self.n_components = len(value_columns)
        self.rep_columns = value_columns
        logger.debug(
            f"[KMeans] use_rep={self.use_rep!r}, table={self.rep_table}, "
            f"dimensions={self.n_components}"
        )
        return value_columns

    def _iter_rep_batches(self, atlas: Atlas):
        """Yield ``(atlas_cell_id, X)`` batches from the selected ``obsm`` table."""

        if self.rep_columns is None:
            self._resolve_rep_columns(atlas)

        select_cols = ["atlas_cell_id", *self.rep_columns]
        select_sql = ", ".join(_q(c) for c in select_cols)
        query = f"""
            SELECT {select_sql}
            FROM {_q(self.rep_table)}
        """
        reader = atlas.connection.execute(query).fetch_record_batch(
            rows_per_batch=self.batch_size
        )

        while True:
            try:
                record_batch = reader.read_next_batch()
            except StopIteration:
                break

            df = record_batch.to_pandas()
            if df.empty:
                continue

            cell_ids = df["atlas_cell_id"].to_numpy(dtype=np.int64, copy=False)
            X = df[self.rep_columns].to_numpy(dtype=np.float32, copy=False)
            X = np.ascontiguousarray(X, dtype=np.float32)
            yield cell_ids, X

    def _write_label_chunk(
        self,
        atlas: Atlas,
        cell_ids: np.ndarray,
        labels: np.ndarray,
        use_cluster_table: str,
        write_to_obs: bool,
        add_obs_col: str,
    ) -> None:
        """Write one accumulated label chunk to the result table and ``obs``."""

        conn = atlas.connection
        batch_df = pd.DataFrame({
            "atlas_cell_id": cell_ids,
            "cluster_id": labels,
        })

        conn.append(use_cluster_table, batch_df)

        if write_to_obs:
            conn.register("_kmeans_batch_tmp", batch_df)

            conn.execute(f"""
                UPDATE obs
                SET {add_obs_col} = t.cluster_id
                FROM _kmeans_batch_tmp t
                WHERE obs.atlas_cell_id = t.atlas_cell_id
            """)

            conn.unregister("_kmeans_batch_tmp")

    def _write_label_chunks(
        self,
        atlas: Atlas,
        cell_id_chunks: list[np.ndarray],
        label_chunks: list[np.ndarray],
        use_cluster_table: str,
        write_to_obs: bool,
        add_obs_col: str,
        write_chunk_rows: int = 1_000_000,
    ) -> None:
        """Write predicted labels in large chunks after representation reading ends."""

        pending_ids = []
        pending_labels = []
        pending_n = 0

        for cell_ids, labels in zip(cell_id_chunks, label_chunks):
            pending_ids.append(cell_ids)
            pending_labels.append(labels)
            pending_n += len(labels)

            if pending_n < write_chunk_rows:
                continue

            self._write_label_chunk(
                atlas=atlas,
                cell_ids=np.concatenate(pending_ids),
                labels=np.concatenate(pending_labels),
                use_cluster_table=use_cluster_table,
                write_to_obs=write_to_obs,
                add_obs_col=add_obs_col,
            )
            pending_ids.clear()
            pending_labels.clear()
            pending_n = 0

        if pending_n > 0:
            self._write_label_chunk(
                atlas=atlas,
                cell_ids=np.concatenate(pending_ids),
                labels=np.concatenate(pending_labels),
                use_cluster_table=use_cluster_table,
                write_to_obs=write_to_obs,
                add_obs_col=add_obs_col,
            )

    # Write obs_cluster
    def _write_clusters(self, atlas: Atlas, cell_ids: np.ndarray, labels: np.ndarray, table_name: str):

        """Write KMeans cluster labels for one batch to the result table.

        This internal function combines the ``atlas_cell_id`` values of the
        current batch and the ``cluster_id`` values predicted by KMeans into a
        DataFrame, and appends it to an ``obs_cluster``-style result table.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB
            database.
        cell_ids
            Array of ``atlas_cell_id`` values corresponding to the current batch.
        labels
            Array of cluster labels predicted for the current batch.
        table_name
            Name of the result table used to save cluster labels.

        Returns
        -------
        None
            Cluster labels are appended directly to the database table. No object
            is returned.

        Notes
        -----
        This helper currently only appends results to an independent result table;
        the logic for synchronizing back to ``obs`` is implemented in
        ``predict_kmeans``.
        """
        df = pd.DataFrame({
            "atlas_cell_id": cell_ids,
            "cluster_id": labels.astype(np.int32)
        })

        atlas.connection.append(table_name, df)


    # Write kmeans_centers
    def _write_centers(self, atlas: Atlas, table_name: str="kmeans_centers"):

        """Write KMeans cluster centers to a database table.

        This internal function reads ``self.kmeans.cluster_centers_`` and expands
        the center coordinate of each cluster on each PCA dimension into a long
        table, then writes it to a ``kmeans_centers``-style table. The output
        table contains three columns: ``cluster_id``, ``pc_index``, and ``value``.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB
            database.
        table_name
            Name of the table used to save KMeans centers. The default value is
            ``"kmeans_centers"``.

        Returns
        -------
        None
            Cluster centers are written directly to the database table. No object
            is returned.

        Notes
        -----
        ``fit_kmeans`` must be completed before calling this method; otherwise,
        ``cluster_centers_`` has not been generated yet.
        """
        conn = atlas.connection

        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        conn.execute(f"""
            CREATE TABLE {table_name} (
                cluster_id INTEGER,
                pc_index INTEGER,
                value FLOAT
            )
        """)

        C = self.kmeans.cluster_centers_

        rows = []
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                rows.append((i, j, float(C[i, j])))

        df = pd.DataFrame(
            rows,
            columns=["cluster_id", "pc_index", "value"]
        )

        conn.append(table_name, df)


    def fit_kmeans(self, atlas: Atlas) -> "StreamingKMeans":

        """Train the MiniBatchKMeans model in batches.

        This method reads the selected low-dimensional representation in batches
        and calls ``MiniBatchKMeans.partial_fit`` to update the cluster centers.

        This method only trains the KMeans model and does not write cell cluster
        labels. Label writing is handled by ``predict_kmeans`` or ``run``.

        Parameters
        ----------
        atlas
            Atlas object. It must already be connected to a DuckDB database, the
            selected ``obsm`` representation must exist.

        Returns
        -------
        StreamingKMeans
            The current ``StreamingKMeans`` object, enabling chained calls.

        Notes
        -----
        If the selected representation has fewer dimensions than ``n_components``,
        all available dimensions are used.

        """

        batch_count = 0

        with progress(total=self.fit_batches, desc="KMeans fit") as pbar:
            while batch_count < self.fit_batches:
                pass_batches = 0

                for _, X_batch in self._iter_rep_batches(atlas):
                    if not np.isfinite(X_batch).all():
                        raise ValueError(
                            f"NaN/Inf exists in {self.use_rep}: "
                            f"min={np.nanmin(X_batch)}, max={np.nanmax(X_batch)}"
                        )

                    self.kmeans.partial_fit(X_batch)

                    batch_count += 1
                    pass_batches += 1
                    pbar.update(1)

                    if batch_count % 10 == 0:
                        logger.debug(f"[KMeans] partial_fit batch = {batch_count}/{self.fit_batches}")

                    if batch_count >= self.fit_batches:
                        break

                if pass_batches == 0:
                    break

        if batch_count == 0:
            raise RuntimeError("[KMeans] No minibatch was obtained, so KMeans cannot be trained")

        return self


    def predict_kmeans(
        self,
        atlas: Atlas,
        use_cluster_table: str = "obs_cluster",
        write_to_obs: bool = True,
        add_obs_col: str = "kmeans"
    ) -> "StreamingKMeans":

        """Predict KMeans cluster labels for all cells and write them to the database.

        This method uses the trained ``MiniBatchKMeans`` model and reads the
        selected low-dimensional representation in batches. ``self.kmeans.predict``
        is called on each batch to obtain cluster labels.

        Prediction results are written to the independent result table specified
        by ``use_cluster_table``. When ``write_to_obs=True``, the results are also
        synchronized back to the ``add_obs_col`` column in the ``obs`` table. After
        prediction ends, the function also writes cluster centers to the
        ``kmeans_centers`` table.

        Parameters
        ----------
        atlas
            Atlas object. It must already be connected to a DuckDB database, and
            the dense minibatch reading workflow must be available.

        use_cluster_table
            Name of the database table used to save cell cluster labels. The
            default value is ``"obs_cluster"``.

        write_to_obs
            Whether to synchronize cluster labels to the ``obs`` table. The
            default value is ``True``.

        add_obs_col
            Name of the cluster label column written to the ``obs`` table. The
            default value is ``"kmeans"``.

        Returns
        -------
        StreamingKMeans
            The current ``StreamingKMeans`` object.

        Notes
        -----
        ``fit_kmeans`` must be run before calling this method. If
        ``fit_kmeans`` must be run before calling this method.

        Examples
        --------
        Write cluster labels using a trained model::

            model.fit_kmeans(atlas)
            model.predict_kmeans(atlas, add_obs_col="kmeans_20")
        """

        conn = atlas.connection

        conn.execute(f"DROP TABLE IF EXISTS {use_cluster_table}")
        conn.execute(f"""
            CREATE TABLE {use_cluster_table} (
                atlas_cell_id BIGINT,
                cluster_id INTEGER
            )
        """)

        # Add the kmeans column to obs
        if write_to_obs:
            obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
            if add_obs_col not in obs_cols:
                conn.execute(f"ALTER TABLE obs ADD COLUMN {add_obs_col} INTEGER")
            # Clear old results first
            conn.execute(f"UPDATE obs SET {add_obs_col} = NULL")

        predicted_cells = 0
        predict_batch_count = 0
        cell_id_chunks = []
        label_chunks = []

        for atlas_cell_ids, X_batch in progress(
            self._iter_rep_batches(atlas),
            desc="KMeans predict",
        ):
            labels = self.kmeans.predict(X_batch).astype(np.int32)
            n = len(labels)

            # Do not write to DuckDB while the Arrow reader is still open. A
            # write on the same connection can invalidate the pending read.
            cell_id_chunks.append(atlas_cell_ids.copy())
            label_chunks.append(labels.copy())

            predicted_cells += n
            predict_batch_count += 1

            if predict_batch_count % 20 == 0:
                logger.debug(
                    f"[KMeans] predicted cells = {predicted_cells:,}, "
                    f"batches = {predict_batch_count}"
                )

        self._write_label_chunks(
            atlas=atlas,
            cell_id_chunks=cell_id_chunks,
            label_chunks=label_chunks,
            use_cluster_table=use_cluster_table,
            write_to_obs=write_to_obs,
            add_obs_col=add_obs_col,
        )

        # Save centers
        self._write_centers(atlas, table_name="kmeans_centers")

        return self


    # Run the main function
    def run(
        self,
        atlas: Atlas,
        use_cluster_table: str = "obs_cluster",
        write_to_obs: bool = True,
        add_obs_col: str = "kmeans"
    ) -> "StreamingKMeans":
        """Train and write streaming KMeans clustering results.

        This method is the main workflow entry point of ``StreamingKMeans``.
        It first calls ``fit_kmeans`` to train the model, and then calls
        ``predict_kmeans`` to predict cluster labels for all cells.

        By default, results are written to an independent clustering result table
        and can also be synchronized back to the specified column in the ``obs``
        table, making them convenient for later UMAP coloring, differential gene
        analysis, and cluster-level visualization.

        Parameters
        ----------
        atlas
            Atlas object.

            The database must already contain the selected low-dimensional
            representation table.

        use_cluster_table
            Name of the database table used to save clustering results.

        write_to_obs
            Whether to synchronize cluster labels back to the ``obs`` table.

        add_obs_col
            Column name used when writing to ``obs``.

        Returns
        -------
        self
            The current ``StreamingKMeans`` object.

        Examples
        --------
        Run the complete KMeans workflow::

            model.run(atlas, use_cluster_table="obs_cluster", add_obs_col="kmeans")
        """

        # KMeans training
        self.fit_kmeans(atlas)

        # KMeans transform
        self.predict_kmeans(
            atlas,
            use_cluster_table=use_cluster_table,
            write_to_obs=write_to_obs,
            add_obs_col=add_obs_col
        )

        return self


# Entry function
@duckdb_memory_limit("5G")
def kmeans(
    atlas: Atlas,
    use_rep: str = "X_umap",
    n_components: int = 30,
    n_clusters: int = 10,
    batch_size: int = 2048,
    fit_batches: int = 1000,
    add_obs_col: str = "kmeans",
    use_cluster_table: str = "obs_cluster",
) -> None:

    """Perform MiniBatch K-means clustering on a low-dimensional representation.

    This function is the public KMeans entry point of scAtlasPy. By default, it
    clusters the UMAP coordinates stored in ``obsm_X_umap``. The representation
    is read in batches, ``MiniBatchKMeans`` is updated in a streaming manner, and
    all cells are predicted in a second pass.

    After running, three types of results are usually generated or updated:

    - ``use_cluster_table``: ``cluster_id`` for each cell;
    - ``obs[add_obs_col]``: optional obs cluster label column;
    - ``kmeans_centers``: center coordinates of each cluster in the selected
      representation.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the selected representation table must already exist in the database.

    use_rep
        Representation used for clustering. ``"X_umap"`` resolves to
        ``obsm_X_umap`` and is the default. ``"X_pca"`` resolves to
        ``obsm_X_pca``.

    n_components
        Number of representation dimensions used for KMeans clustering. The
        default value is ``30``. If the selected representation has fewer
        dimensions, all available dimensions are used.

    n_clusters
        Number of KMeans clusters. The default value is ``10``.

    batch_size
        Number of cells in each minibatch. A larger value is usually faster, but
        increases memory usage for each batch.

    fit_batches
        Maximum number of minibatches to read during the training stage. A larger
        value usually makes cluster centers more stable, but increases runtime.

    add_obs_col
        Name of the cluster label column written to the ``obs`` table. The
        default value is ``"kmeans"``.

    use_cluster_table
        Name of the independent result table used to save cluster labels for each
        cell. The default value is ``"obs_cluster"``.

    Returns
    -------
    None
        Clustering results are written directly to the Atlas database. No object
        is returned.

    Notes
    -----
    This function does not run UMAP or PCA automatically. With the default
    ``use_rep="X_umap"``, run ``sap.tl.umap(atlas)`` before clustering. To
    cluster PCA coordinates instead, run ``sap.tl.pca(atlas)`` and pass
    ``use_rep="X_pca"``.

    Examples
    --------
    Cluster into 20 groups using UMAP coordinates::

        sap.tl.umap(atlas)
        sap.tl.kmeans(atlas, n_clusters=20)
    """

    t_start = time.time()

    runner = StreamingKMeans(
        use_rep=use_rep,
        n_components=n_components,
        n_clusters=n_clusters,
        batch_size=batch_size,
        fit_batches=fit_batches,
    )

    runner.run(
        atlas,
        use_cluster_table=use_cluster_table,
        add_obs_col=add_obs_col,
    )

    t_end = time.time()

    logger.info(f" KMeans Done, elapsed time = {t_end - t_start:.2f} seconds")
