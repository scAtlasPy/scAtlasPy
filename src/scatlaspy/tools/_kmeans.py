from sklearn.cluster import MiniBatchKMeans
from ..io import progress
from ..data import Atlas
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger('Atlas')

# MiniBatchKMeans
class StreamingKMeans:

    """Streaming MiniBatchKMeans clusterer based on PCA embeddings.

    This class reads the PCA loadings table ``varm_PCs`` from the Atlas database
    and reads the expression matrix in batches through ``atlas.get_minibatch_dense``.
    Each minibatch is first projected into PCA space and then used for streaming
    training or prediction with sklearn ``MiniBatchKMeans``.

    The complete workflow consists of two steps:

    - ``fit_kmeans``: reads minibatches, projects them into PCA space, and trains
      KMeans centers with ``partial_fit``;
    - ``predict_kmeans``: reads all minibatches again, predicts the cluster for
      each cell, and writes the results to an independent result table and an
      optional ``obs`` column.

    This class is the underlying implementation of ``sap.tl.kmeans``. Regular
    users usually call the public function ``kmeans`` directly.

    Parameters
    ----------
    n_components
        Number of PCA components to use. It must match the number of available
        PC columns in ``varm_PCs``.
    n_clusters
        Number of K-means clusters.
    batch_size
        Number of cells in each minibatch.
    fit_batches
        Maximum number of minibatches used to train MiniBatchKMeans.
    buffer_batch_num
        Number of batches cached in the shuffle buffer during ``multi-pass``
        reading.

    Notes
    -----
    PCA must be completed before running this class. Make sure the ``varm_PCs``
    table exists in the database and that the dense minibatch read index in Atlas
    has already been built.

    Examples
    --------
    Recommended public API usage::

        sap.tl.pca(atlas)
        sap.tl.kmeans(atlas, n_components=30, n_clusters=20)
    """

    def __init__(
        self,
        n_components: int=50,
        n_clusters: int=2,
        batch_size: int=2048,
        fit_batches: int = 1000,
        buffer_batch_num: int = 5,
    ):
        """Initialize the streaming MiniBatchKMeans clusterer.

        This method stores the PCA projection dimension, number of clusters, and
        minibatch parameters, and creates the sklearn ``MiniBatchKMeans`` model.

        Later, ``fit_kmeans`` reads PCA loadings and expression matrix minibatches
        from Atlas, first projects the expression matrix into PCA space, and then
        performs streaming training with ``partial_fit``.

        Parameters
        ----------
        n_components
            Number of PCA components to use.

            It needs to be consistent with the number of PC columns in the
            ``varm_PCs`` table.

        n_clusters
            Number of KMeans clusters.

        batch_size
            Number of cells in each minibatch.

            A larger value usually trains faster, but increases the memory usage
            of projection and clustering for each batch.

        fit_batches
            Maximum number of minibatches used to train MiniBatchKMeans.

        buffer_batch_num
            Number of batches cached in the shuffle buffer during ``multi-pass``
            reading.

        Notes
        -----
        This object only initializes the model and parameters. It does not read
        the database or write cluster labels immediately.
        """

        # PCA parameters from the trained PCA model; read components_ from varm_PCs
        self.components_ = None  # components_ = axes -> directions for projection
        self.n_components = n_components
        self.n_clusters = n_clusters  # Target number of clusters
        self.batch_size = batch_size
        self.fit_batches = fit_batches  # Number of minibatches used for KMeans training
        self.buffer_batch_num = buffer_batch_num # Number of batches in ShuffleBuffer during multi-pass
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init="k-means++",
            n_init="auto",
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


    # Transform with PCA + train minibatch KMeans clustering
    def fit_kmeans(self, atlas: Atlas) -> "StreamingKMeans":

        """Train the MiniBatchKMeans model in batches.

        This method first reads PCA loadings from the ``varm_PCs`` table, then
        reads the expression matrix in batches through
        ``atlas.get_minibatch_dense(pass_mode="multi-pass")``. Each batch is
        multiplied by the PCA loadings to transform it into PCA space, and then
        ``MiniBatchKMeans.partial_fit`` is called to update the cluster centers.

        This method only trains the KMeans model and does not write cell cluster
        labels. Label writing is handled by ``predict_kmeans`` or ``run``.

        Parameters
        ----------
        atlas
            Atlas object. It must already be connected to a DuckDB database, the
            database must contain the ``varm_PCs`` table, and the dense minibatch
            reading workflow must be available.

        Returns
        -------
        StreamingKMeans
            The current ``StreamingKMeans`` object, enabling chained calls.

        Notes
        -----
        If the PCA dimension in the database is inconsistent with the
        ``n_components`` value passed during initialization, the current
        implementation uses the PCA dimension actually read from the database.

        """

        # Read PCA components
        self.components_ = self.load_components(atlas)

        # If the user-provided n_components differs from the actual PCA dimension in the database, give a hint
        real_components = self.components_.shape[0]
        if real_components != self.n_components:
            self.n_components = real_components

        batch_count = 0

        # Minibatch KMeans clustering training
        for X_batch in progress(
                atlas.get_minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                ),
                total=self.fit_batches,
                desc="KMeans fit"
        ):

            t0 = time.time()

            X_pca = X_batch @ self.components_.T  # PCA transform

            X_pca = np.ascontiguousarray(X_pca, dtype=np.float32)

            if not np.isfinite(X_pca).all():
                raise ValueError(
                    f"NaN/Inf exists in X_pca: "
                    f"min={np.nanmin(X_pca)}, max={np.nanmax(X_pca)}"
                )

            t1 = time.time()
            self.kmeans.partial_fit(X_pca)   # KMeans training

            batch_count += 1

            if batch_count % 10 == 0:
                logger.info(f"[KMeans] partial_fit batch = {batch_count}/{self.fit_batches}")

        if batch_count == 0:
            raise RuntimeError("[KMeans] No minibatch was obtained, so KMeans cannot be trained")

        return self


    # Transform with PCA + predict minibatch KMeans clustering
    def predict_kmeans(
        self,
        atlas: Atlas,
        use_cluster_table: str = "obs_cluster",
        write_to_obs: bool = True,
        add_obs_col: str = "kmeans"
    ) -> "StreamingKMeans":

        """Predict KMeans cluster labels for all cells and write them to the database.

        This method uses the trained ``MiniBatchKMeans`` model and performs
        single-pass minibatch reading over the full expression matrix. Each batch
        is first projected into PCA space, and then ``self.kmeans.predict`` is
        called to obtain cluster labels.

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
        ``self.components_`` is empty, the function automatically reads PCA
        loadings from the ``varm_PCs`` table.

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

        # Read PCA components
        if self.components_ is None:
            self.components_ = self.load_components(atlas)

        cell_offset = 0
        predict_batch_count = 0

        # Use single-pass in the transform stage
        for X_batch in progress(
                atlas.get_minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="single-pass",
                ),
                desc="KMeans predict"
        ):

            # PCA transform
            X_pca = X_batch @ self.components_.T

            # KMeans transform
            labels = self.kmeans.predict(X_pca).astype(np.int32)

            # atlas_cell_id values corresponding to the current batch
            n = len(labels)
            atlas_cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int32)

            batch_df = pd.DataFrame({
                "atlas_cell_id": atlas_cell_ids,
                "cluster_id": labels
            })

            # Write table
            conn.append(use_cluster_table, batch_df)

            # Synchronize back to obs.kmeans
            if write_to_obs:
                conn.register("_kmeans_batch_tmp", batch_df)

                conn.execute(f"""
                    UPDATE obs
                    SET {add_obs_col} = t.cluster_id
                    FROM _kmeans_batch_tmp t
                    WHERE obs.atlas_cell_id = t.atlas_cell_id
                """)

                conn.unregister("_kmeans_batch_tmp")

            cell_offset += n
            predict_batch_count += 1

            if predict_batch_count % 20 == 0:
                logger.info(
                    f"[KMeans] predicted cells = {cell_offset:,}, "
                    f"batches = {predict_batch_count}"
                )

        # Save centers
        self._write_centers(atlas, table_name="kmeans_centers")

        return self


    # Read PCA components from the database and restore them to self.components_
    def load_components(self, atlas: Atlas, table_name: str = "varm_PCs") -> np.ndarray:

        """Read PCA loadings from the database for KMeans projection.

        This method reads the ``varm_PCs`` table, sorts it by ``atlas_gene_id``,
        removes the gene ID column, and transposes it into the shape
        ``(n_components, n_genes)``. The returned matrix is used for
        ``X_batch @ components_.T`` to project the expression matrix into PCA
        space.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        table_name
            Name of the PCA loadings table to read. The default value is ``"varm_PCs"``.

        Returns
        -------
        numpy.ndarray
            PCA loadings array with shape ``(n_components, n_genes)`` and type ``float32``.

        Notes
        -----
        This method depends on the ``varm_PCs`` table already generated by ``sap.tl.pca``.

        Examples
        --------
        Read PCA loadings::

            components = model.load_components(atlas)
        """
        conn = atlas.connection

        df = conn.execute(f"""
            SELECT *
            FROM {table_name}
            ORDER BY atlas_gene_id
        """).fetchdf()

        # Remove atlas_gene_id
        pcs = df.drop(columns=["atlas_gene_id"]).values

        # Transpose back to the original PCA format: (gene, pc) -> (pc, gene)
        components_ = pcs.T.astype(np.float32)

        return components_


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

            The database must already contain the PCA loadings table, and the
            filtering index and minibatch reading workflow must be available.

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
def kmeans(
    atlas: Atlas,
    n_components: int = 30,
    n_clusters: int = 10,
    batch_size: int = 2048,
    fit_batches: int = 1000,
    add_obs_col: str = "kmeans",
    use_cluster_table: str = "obs_cluster",
) -> None:

    """Perform MiniBatch K-means clustering based on PCA embeddings.

    This function is the public KMeans entry point of scAtlasPy. It reads PCA
    loadings saved in ``varm_PCs`` and reads the expression matrix in batches
    through the Atlas dense minibatch interface. During the training stage, each
    minibatch is projected into PCA space and ``MiniBatchKMeans`` is updated in a
    streaming manner. During the prediction stage, all cells are read again in
    batches, cluster labels are predicted, and the results are written to the
    database.

    After running, three types of results are usually generated or updated:

    - ``use_cluster_table``: ``cluster_id`` for each cell;
    - ``obs[add_obs_col]``: optional obs cluster label column;
    - ``kmeans_centers``: center coordinates of each cluster in PCA space.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the PCA loadings table ``varm_PCs`` must already exist in the database.

    n_components
        Number of PCA components used for KMeans clustering. The default value is
        ``30``. If the actual available dimension in ``varm_PCs`` differs from
        this value, the underlying class uses the PCA dimension actually read from the database.

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
    This function does not run PCA automatically. If ``varm_PCs`` does not exist
    in the database, the function raises an error and prompts the user to run
    ``sap.tl.pca(atlas)`` first.

    Examples
    --------
    Cluster into 20 groups using the first 30 principal components::

        sap.tl.pca(atlas)
        sap.tl.kmeans(atlas, n_components=30, n_clusters=20)
    """

    t_start = time.time()

    conn = atlas.connection

    # Check whether PCA components exist
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    if "varm_PCs" not in tables:
        raise ValueError(
            "varm_PCs does not exist in the database.\n"
            "Please run sap.tl.pca(atlas) before running sap.tl.kmeans(atlas)."
        )

    runner = StreamingKMeans(
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
