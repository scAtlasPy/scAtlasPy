from ..data import Atlas
from sklearn.decomposition import IncrementalPCA
import numpy as np
from ..io import progress
import pandas as pd
import time
import logging
logger = logging.getLogger('Atlas')


class StreamingPCA:
    """Streaming PCA model for the Atlas database.

    This class wraps sklearn's ``IncrementalPCA`` and is used to train PCA
    without loading the full expression matrix into memory at once. It obtains
    dense expression matrices in batches through the Atlas minibatch reading
    interface, first learns principal components with ``partial_fit``, and then
    projects all cells into PCA space in batches.

    After the run is completed, the results are written to three tables in the
    Atlas database:

    - ``obsm_X_pca``: PCA coordinates for each cell;
    - ``varm_PCs``: loadings of each gene on each principal component;
    - ``uns_pca_stats``: variance and variance explanation ratio for each
      principal component.

    This class is the underlying implementation of ``sap.tl.pca``. Regular
    users usually call the public function ``pca`` directly. This class only
    needs to be used directly when debugging the training process or customizing
    streaming parameters.

    Parameters
    ----------
    n_components
        Number of PCA principal components to calculate.

    fit_batches
        Maximum number of minibatches used to fit ``IncrementalPCA``.

    buffer_batch_num
        Number of batches cached in the shuffle buffer during ``multi-pass`` reading.

    batch_size
        Number of cells contained in each minibatch.

    Notes
    -----
    Before running, the read index needs to be built through
    ``atlas.build_read_index(...)`` so that ``atlas.get_minibatch_dense`` can
    read the expression matrix as expected.

    Examples
    --------
    Recommended public API usage::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas, n_components=50)
    """

    # Initialize
    def __init__(self,
                 n_components: int = 30,
                 fit_batches: int = 1000,
                 buffer_batch_num: int = 5,
                 batch_size: int = 2048,
                 ):

        """Initialize the streaming PCA calculator.

        This method only stores PCA parameters and creates the sklearn
        ``IncrementalPCA`` object. It does not immediately read the database or
        write any result tables. Actual training and database writing happen in
        the ``fit``, ``transform``, or ``run`` stages.

        Parameters
        ----------
        n_components
            Number of PCA principal components to calculate. This value
            determines the output dimensionality in ``obsm_X_pca``, ``varm_PCs``,
            and ``uns_pca_stats``.

        fit_batches
            Maximum number of minibatches to read during training for ``partial_fit``.

        buffer_batch_num
            Number of minibatches cached in the prefetch or shuffle buffer during ``multi-pass`` reading.

        batch_size
            Number of cells in each minibatch.

        Notes
        -----
        Larger ``batch_size`` and ``fit_batches`` values usually improve PCA
        stability, but they increase training time and per-batch memory usage.
        """

        self.n_components = n_components # Target PCA dimension
        self.ipca = IncrementalPCA(n_components=n_components) # Create sklearn's incremental PCA model
        self.fit_batches = fit_batches
        self.buffer_batch_num = buffer_batch_num
        self.batch_size = batch_size

        self.components_ = None                 # components_ = coordinate axes      -> directions for projection
        self.explained_variance_ = None         # variance = importance of each axis -> strength of this direction
        self.explained_variance_ratio_ = None   # ratio = proportion of total information -> amount of information explained


    # Create the obsm_X_pca table
    def _create_pca_table(self, atlas:Atlas, n_components: int = 30, table_name: str="obsm_X_pca"):

        """Create the cell PCA coordinate result table.

        This internal function is used to create an ``obsm_X_pca``-style result
        table. Each row in the table corresponds to one cell and contains
        ``atlas_cell_id`` as well as PCA coordinate columns such as ``pc0`` and
        ``pc1``. If a table with the same name already exists, the old table is
        dropped first and then recreated, avoiding old results with different PCA
        dimensions from being mixed into the new results.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        n_components
            Number of PCA coordinate columns to create.

        table_name
            Result table name. The default value is ``"obsm_X_pca"``.

        Returns
        -------
        None
            The table structure is created directly in the Atlas database.
            No object is returned.

        Notes
        -----
        This is an internal table-creation helper, usually called automatically
        by ``run``.
        """

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
            atlas_cell_id INTEGER,
            {cols}
        );
        """
        atlas.connection.execute(sql)


    # Create the varm_PCs table
    def _create_pcs_table(self, atlas:Atlas,  n_components: int = 30, table_name: str="varm_PCs"):

        """Create the gene PCA loadings result table.

        This internal function is used to create the ``varm_PCs`` table. Each row
        in the table corresponds to one gene and contains ``atlas_gene_id`` as
        well as principal component loading columns such as ``pc0`` and ``pc1``.
        This table can later be reused by KMeans, UMAP preprocessing, or other
        workflows that require PCA projection.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        n_components
            Number of PCA loading columns to create.

        table_name
            Result table name. The default value is ``"varm_PCs"``.

        Returns
        -------
        None
            The table structure is created directly in the Atlas database. No
            object is returned.

        Notes
        -----
        This is an internal table-creation helper, usually called automatically
        by ``run``.
        """

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            atlas_gene_id USMALLINT,
            {cols}
        );
        """
        atlas.connection.execute(sql)


    # Create the uns_pca_stats table
    def _create_pca_stats_table(self, atlas:Atlas, table_name: str="uns_pca_stats"):

        """Create the PCA variance statistics result table.

        This internal function is used to create the ``uns_pca_stats`` table.
        Each row in the table corresponds to one principal component and records
        ``pc_index``, the variance explained by that principal component
        ``variance``, and the variance explanation ratio ``variance_ratio``.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        table_name
            Result table name. The default value is ``"uns_pca_stats"``.

        Returns
        -------
        None
            The table structure is created directly in the Atlas database. No
            object is returned.

        Notes
        -----
        This is an internal table-creation helper, usually called automatically
        by ``run``.
        """

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pc_index USMALLINT,
            variance REAL, 
            variance_ratio REAL 
        );
        """
        atlas.connection.execute(sql)


    # Write the obsm_X_pca table
    def _writer_obsm_x_pca(self, atlas: Atlas, X_batch: np.ndarray, cell_offset: int, table_name: str= "obsm_X_pca"):

        """Write the PCA coordinates of one batch to the cell result table.

        This internal function converts the PCA projection results of the current
        batch into a ``float32`` DataFrame, generates continuous
        ``atlas_cell_id`` values based on ``cell_offset``, and appends the result
        to an ``obsm_X_pca``-style result table.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        X_batch
            PCA coordinate matrix of the current batch, with shape
            ``(n_cells_in_batch, n_components)``.

        cell_offset
            Starting global ``atlas_cell_id`` corresponding to the first cell of
            the current batch.

        table_name
            PCA coordinate table to write to. The default value is
            ``"obsm_X_pca"``.

        Returns
        -------
        int
            ``cell_offset`` that should be used by the next batch.

        Notes
        -----
        This function assumes that the minibatch reading order is consistent with
        the order of ``atlas_cell_id``.
        """

        n = X_batch.shape[0]

        cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int32) # atlas_cell_id

        X_batch = X_batch.astype(np.float32) # float32, which saves space

        # Build DataFrame
        df = pd.DataFrame(
            X_batch,
            columns=[f"pc{i}" for i in range(X_batch.shape[1])]
        )

        df.insert(0, "atlas_cell_id", cell_ids)

        atlas.connection.append(table_name, df)

        return cell_offset + n


    # Write the varm_PCs table
    def _writer_varm_pcs(self, atlas: Atlas, table_name: str= "varm_PCs"):

        """Write PCA loadings to the ``varm_PCs`` table.

        This internal function reads ``self.components_``, transposes the sklearn
        structure from ``(n_components, n_genes)`` to
        ``(n_genes, n_components)``, and then writes the result by gene to a
        ``varm_PCs``-style result table.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        table_name
            Loadings table to write to. The default value is ``"varm_PCs"``.

        Returns
        -------
        None
            PCA loadings are appended directly to the database table. No object
            is returned.

        Notes
        -----
        Before calling this method, make sure ``fit`` has completed and
        ``self.components_`` is not empty.
        """

        pcs = self.components_.T.astype(np.float32)  # (n_genes, n_components)
        df = pd.DataFrame(
            pcs,
            columns=[f"pc{i}" for i in range(pcs.shape[1])]
        )
        # Insert atlas_gene_id
        df.insert(0, "atlas_gene_id", np.arange(pcs.shape[0], dtype=np.int32))
        atlas.connection.append(table_name, df)


    # Write the uns_pca_stats table
    def _writer_uns_pca_stats(self, atlas: Atlas, table_name: str="uns_pca_stats"):

        """Write PCA variance explanation statistics to the ``uns_pca_stats`` table.

        This internal function organizes ``self.explained_variance_`` and
        ``self.explained_variance_ratio_`` into a DataFrame and appends it to the database.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        table_name
            PCA statistics table to write to. The default value is ``"uns_pca_stats"``.

        Returns
        -------
        None
            PCA statistics are appended directly to the database table. No object
            is returned.

        Notes
        -----
        Before calling this method, make sure ``fit`` has completed and the
        variance statistics arrays have been generated.
        """

        pc_index = np.arange(len(self.explained_variance_), dtype=np.int32)

        df = pd.DataFrame({
            "pc_index": np.arange(len(self.explained_variance_), dtype=np.int32),
            "variance": self.explained_variance_.astype(np.float32),
            "variance_ratio": self.explained_variance_ratio_.astype(np.float32)
        })

        atlas.connection.append(table_name, df)


    # Train PCA
    def fit(self, atlas: Atlas) -> "StreamingPCA":

        """Fit the IncrementalPCA model in batches.

        This method reads the dense expression matrix in batches through
        ``atlas.get_minibatch_dense(pass_mode="multi-pass")`` and performs
        streaming training with ``IncrementalPCA.partial_fit``. After training is
        completed, the principal components, variance, and variance explanation
        ratios are saved to the current object's attributes.

        This method only trains the model and does not write the ``obsm_X_pca``,
        ``varm_PCs``, or ``uns_pca_stats`` tables. Database writing is handled by
        ``fit_transform`` or ``run``.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to the database,
            and an index available for dense minibatch reading must have already been built.

        Returns
        -------
        StreamingPCA
            The current ``StreamingPCA`` object, enabling chained calls.

        Notes
        -----
        If no minibatch is read, the function raises ``RuntimeError`` to avoid
        generating an empty PCA model.

        """

        batch_count = 0

        for X_batch in progress(
                atlas.get_minibatch_dense(
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                    batch_size=self.batch_size,
                ),
                total=self.fit_batches,
                desc="PCA"
        ):
            self.ipca.partial_fit(X_batch)

            batch_count += 1

            if batch_count % 10 == 0:
                logger.info(f"[PCA] partial_fit batch = {batch_count}/{self.fit_batches}")

        if batch_count == 0:
            raise RuntimeError("[PCA] No minibatch was obtained, so PCA cannot be trained")

        # Save results
        self.components_ = self.ipca.components_.astype(np.float32)
        self.explained_variance_ = self.ipca.explained_variance_.astype(np.float32)
        self.explained_variance_ratio_ = self.ipca.explained_variance_ratio_.astype(np.float32)

        cum_ratio = np.cumsum(self.explained_variance_ratio_)

        logger.info("[PCA] Cumulative explained variance ratio (first {} principal components): {:.4f}".format(
            len(self.explained_variance_ratio_),
            self.explained_variance_ratio_.sum()
        ))

        logger.info("[PCA] Cumulative explained variance ratio of the first 10 principal components:")
        logger.info(cum_ratio[:10])

        logger.info("[PCA] Final cumulative explained variance ratio: {:.4f}".format(cum_ratio[-1]))

        total_ratio = cum_ratio[-1]

        if total_ratio < 0.1:
            logger.info(" PCA explanation ratio is low; you may need to check the data or increase the number of principal components")
        elif total_ratio < 0.2:
            logger.info(" PCA explanation ratio is moderate, which is common in single-cell data")
        elif total_ratio < 0.4:
            logger.info(" PCA explanation ratio is normal")
        else:
            logger.info(" PCA explanation ratio is high, and the structure is relatively clear")

        return self


    def transform(self, atlas: Atlas) -> None:
        """Project all cells into PCA space in batches.

        This method reads the full expression matrix through
        ``atlas.get_minibatch_dense(pass_mode="single-pass")`` and uses the
        fitted ``IncrementalPCA`` model to calculate PCA coordinates for each
        batch. The results of each batch are immediately appended to the
        ``obsm_X_pca`` table.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to the database,
            and an index available for dense minibatch reading must have already
            been built.

        Returns
        -------
        None
            PCA coordinates are written directly to the ``obsm_X_pca`` table. No
            object is returned.

        Notes
        -----
        Before calling this method, run ``fit`` first and make sure
        ``_create_pca_table`` has already created the target table.

        """

        cell_offset = 0  # Global increment

        for X_batch in progress(atlas.get_minibatch_dense(pass_mode="single-pass")):

            X_pca = self.ipca.transform(X_batch)

            # Only write obsm, one batch at a time
            cell_offset = self._writer_obsm_x_pca(
                atlas,
                X_pca,
                cell_offset
            )


    # Main function
    def fit_transform(self, atlas: Atlas) -> "StreamingPCA":

        """Train the PCA model and write all PCA results.

        This method first calls ``fit`` to complete IncrementalPCA training, then
        writes gene loadings and PCA variance statistics, and finally calls
        ``transform`` to project all cells into PCA space.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to the database,
            and the dense minibatch reading workflow must be available.

        Returns
        -------
        StreamingPCA
            The current ``StreamingPCA`` object.

        Notes
        -----
        Before calling this method, the ``obsm_X_pca``, ``varm_PCs``, and
        ``uns_pca_stats`` tables should be created first. The ``run`` method
        automatically completes these table-creation steps.

        """

        # Train
        self.fit(atlas)

        # Write model results once
        self._writer_varm_pcs(atlas)
        self._writer_uns_pca_stats(atlas)

        # transform, writing obsm
        self.transform(atlas)
        return self


    # Get results
    def get_results(self) -> dict[str, np.ndarray | None]:
        """Get the results currently stored in memory by the PCA model.

        This method returns the PCA principal components, variance, and variance
        explanation ratios saved in the current object after ``fit``. It does not
        access the database or read the ``obsm_X_pca``, ``varm_PCs``, or
        ``uns_pca_stats`` tables.

        Returns
        -------
        dict
            Contains the following keys:

            - ``"components"``: PCA loadings, with shape ``(n_components, n_genes)``;
            - ``"explained_variance"``: variance explained by each principal component;
            - ``"explained_variance_ratio"``: variance explanation ratio of each principal component.

        Notes
        -----
        ``fit`` or ``fit_transform`` must be run before calling this method;
        otherwise, the returned arrays may be ``None``.

        Examples
        --------
        View the explanation ratio of the first few principal components::

            model.fit(atlas)
            result = model.get_results()
            result["explained_variance_ratio"][:5]
        """
        return {
            "components": self.components_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_
        }


    # Read PCA components from the database and restore them to self.components_
    def load_components(self, atlas: Atlas, table_name: str = "varm_PCs") -> np.ndarray:

        """Read PCA loadings from the database.

        This method reads a ``varm_PCs``-style table, sorts it by
        ``atlas_gene_id``, removes the gene ID column, and transposes it back to
        the ``(n_components, n_genes)`` shape used by sklearn
        ``IncrementalPCA``.

        Parameters
        ----------
        atlas
            Atlas object. The object must already be connected to a DuckDB database.

        table_name
            PCA loadings table to read. The default value is ``"varm_PCs"``.

        Returns
        -------
        numpy.ndarray
            PCA loadings array, with shape ``(n_components, n_genes)`` and type
            ``float32``.

        Notes
        -----
        This method is often used to check whether the PCA loadings saved in the
        database are consistent with ``self.components_`` in memory. It can also
        be used by later workflows such as KMeans to read the PCA projection
        matrix.

        Examples
        --------
        Restore PCA loadings from the database::

            components = model.load_components(atlas)
        """

        conn = atlas.connection

        # Read the entire table
        df = conn.execute(f"""
            SELECT * FROM {table_name}
            ORDER BY atlas_gene_id
        """).fetchdf()

        # Remove atlas_gene_id
        pcs = df.drop(columns=["atlas_gene_id"]).values

        # Transpose back to the original PCA format: (gene, pc) -> (pc, gene)
        components_ = pcs.T.astype(np.float32)

        return components_


    def run(self, atlas: Atlas) -> None:
        """Execute the complete streaming PCA computation workflow.

        This method first creates the PCA coordinate table, PC loadings table,
        and PCA statistics table, then calls ``fit_transform`` to complete
        IncrementalPCA training and full-cell projection. After the run ends, the
        results are written to the Atlas database.

        Parameters
        ----------
        atlas
            Atlas object. Filtering index construction must have already been
            completed, and expression matrix minibatches must be readable through
            ``atlas.get_minibatch_dense``.

        Returns
        -------
        None
            PCA coordinates, loadings, and variance explanation ratios are
            written directly to Atlas database tables.

        Examples
        --------
        Run PCA through the public API::

            atlas.build_read_index(use_hvg=True)
            sap.tl.pca(atlas, n_components=50)
        """

        # Create tables; the table dimensions must align with self.n_components for this PCA output
        self._create_pca_table(
            atlas,
            n_components=self.n_components
        )

        # The dimensions of the varm_PCs table must match the number of columns in self.components_.T
        self._create_pcs_table(
            atlas,
            n_components=self.n_components
        )

        self._create_pca_stats_table(atlas)

        # Run PCA
        self.fit_transform(atlas)

        # Comparison information
        components = self.load_components(atlas)
        if np.array_equal(components, self.components_):
            logger.info(" components were extracted correctly")
        if np.allclose(components, self.components_):
            logger.info(" components were extracted correctly")


def pca(
    atlas: Atlas,
    n_components: int = 50,
    fit_batches: int = 1000,
    batch_size: int = 2048,
) -> None:
    """Calculate PCA based on the Atlas expression matrix.

    This function is the public PCA entry point of scAtlasPy. It reads the
    expression matrix in batches from Atlas's dense minibatch reading interface,
    uses sklearn ``IncrementalPCA`` to fit principal components in a streaming
    manner, then projects all cells into PCA space and writes the results to the
    database.

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
        The default value is ``50``.

    fit_batches
        Maximum number of minibatches used to fit ``IncrementalPCA``.
        A larger value is usually more stable, but training takes longer.

    batch_size
        Number of cells contained in each minibatch. A larger value usually gives
        higher throughput, but increases per-batch memory usage.

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
    Calculate 50 principal components on the default read index::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas, n_components=50)
    """

    t_start = time.time()

    pca_runner = StreamingPCA(
        n_components=n_components,
        fit_batches=fit_batches,
        batch_size=batch_size,
    )

    pca_runner.run(atlas)

    t_end = time.time()
    logger.info(f" PCA Done, total time = {t_end - t_start:.2f} seconds")
