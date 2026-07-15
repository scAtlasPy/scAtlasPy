import duckdb
import numpy as np
import threading
import queue
import time
from os import PathLike, fspath
import scipy.sparse as sp
import logging
from collections.abc import Iterator
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


class ShuffleBuffer:

    """Dense minibatch shuffle buffer.

    This class caches multiple dense minibatches in ``multi-pass`` minibatch reading mode.
    After reaching the specified capacity, it randomly shuffles the cell order and then
    outputs data batch by batch. It is used to reduce ordering bias during multi-round
    training and mainly serves streaming models such as PCA and K-means.

    Parameters
    ----------
    gene_num
        Number of genes in the dense minibatch, that is, the number of columns in
        the output matrix.
    batch_size
        Number of cells in each minibatch.
    buffer_batch_num
        Maximum number of minibatches cached in the buffer; the total capacity is
        ``batch_size * buffer_batch_num`` cells.

    Notes
    -----
    This is an internal utility class. Regular users usually use it indirectly through
    ``atlas.get_minibatch_dense(...)``.

    Examples
    --------
    Cache and sample dense minibatches in internal tests::

        buffer = ShuffleBuffer(gene_num=2000, batch_size=128, buffer_batch_num=2)
        buffer.add_batch(
            np.zeros((128, 2000), dtype=np.float32),
            np.arange(128, dtype=np.int64),
        )
        buffer.add_batch(
            np.ones((128, 2000), dtype=np.float32),
            np.arange(128, 256, dtype=np.int64),
        )
        X_batch, filter_cell_ids = buffer.sample_batch()

    Process remaining data that is smaller than a complete buffer::

        buffer = ShuffleBuffer(gene_num=100, batch_size=32, buffer_batch_num=4)
        buffer.add_batch(
            np.zeros((20, 100), dtype=np.float32),
            np.arange(20, dtype=np.int64),
        )
        remaining_batches = buffer.flush_remaining()"""

    def __init__(self, gene_num: int, batch_size: int, buffer_batch_num: int):

        """Initialize the object.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Parameters
        ----------
        gene_num
            Number of genes in the dense minibatch.

        batch_size
            Number of cells to read, write, or process in each batch; larger values
            are usually faster but consume more memory.

        buffer_batch_num
            Number of minibatches cached in the shuffle buffer.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        self.batch_size = batch_size
        self.gene_num = gene_num
        self.buffer_batch_num = buffer_batch_num

        # Maximum number of cells in the buffer
        self.buffer_cells = buffer_batch_num * batch_size

        # Actual buffer: expression matrix
        self.X = np.zeros((self.buffer_cells, gene_num), dtype=np.float32)

        # Synchronously save the filter_cell_id corresponding to each row, ensuring that cells can still be written back after shuffling
        self.filter_cell_ids = np.empty(self.buffer_cells, dtype=np.int64)

        # Write pointer
        self.write_ptr = 0

        # Current output batch id
        self.output_batch_id = 0

        # Whether the buffer has already been shuffled
        self.shuffled = False


    # Write one batch into the buffer
    def add_batch(
        self,
        X_batch: np.ndarray,
        filter_cell_ids: np.ndarray,
    ) -> None:
        """Write a dense minibatch into the shuffle buffer.

        This method appends the current dense expression matrix to the buffer. When the
        buffer accumulates ``batch_size * buffer_batch_num`` cells, it randomly shuffles
        the cell order in the buffer and switches to an output state that can be read
        through ``sample_batch``.

        Parameters
        ----------
        X_batch
            Current dense minibatch. Rows represent cells and columns represent genes;
            the number of columns must match ``gene_num`` used during initialization.
            The number of rows is usually ``batch_size``, and the last batch may be smaller.
        filter_cell_ids
            The ``filter_cell_id`` corresponding to each row of the current ``X_batch``.
            Its length must equal ``X_batch.shape[0]``.

        Returns
        -------
        None
            This method only updates the buffer state and does not directly return
            training data.

        Examples
        --------
        Write two batches and trigger shuffling::

            buffer = ShuffleBuffer(gene_num=50, batch_size=16, buffer_batch_num=2)
            buffer.add_batch(
                np.random.rand(16, 50).astype(np.float32),
                np.arange(16, dtype=np.int64),
            )
            buffer.add_batch(
                np.random.rand(16, 50).astype(np.float32),
                np.arange(16, 32, dtype=np.int64),
            )
            X_batch, filter_cell_ids = buffer.sample_batch()

        When the buffer is not yet full, ``sample_batch`` returns ``None``::

            buffer = ShuffleBuffer(gene_num=50, batch_size=16, buffer_batch_num=2)
            buffer.add_batch(
                np.random.rand(16, 50).astype(np.float32),
                np.arange(16, dtype=np.int64),
            )
            assert buffer.sample_batch() is None"""

        # If the buffer is already full and has entered the output stage, do not write more data
        if self.shuffled:
            return

        n = X_batch.shape[0]

        if len(filter_cell_ids) != n:
            raise RuntimeError(
                "ShuffleBuffer.add_batch: the length of filter_cell_ids must equal the number of rows in X_batch."
            )

        # Safety protection to prevent overflow
        if self.write_ptr + n > self.buffer_cells:
            raise RuntimeError("ShuffleBuffer overflow")

        # Write into the buffer; X and filter_cell_ids must maintain the same row correspondence
        self.X[self.write_ptr:self.write_ptr + n] = X_batch
        self.filter_cell_ids[self.write_ptr:self.write_ptr + n] = filter_cell_ids
        self.write_ptr += n

        # If the buffer is full
        if self.write_ptr == self.buffer_cells:
            # Random shuffle: X and filter_cell_ids use the same permutation
            perm = np.random.permutation(self.buffer_cells)
            self.X[:] = self.X[perm]
            self.filter_cell_ids[:] = self.filter_cell_ids[perm]

            # Enter the output stage
            self.output_batch_id = 0
            self.shuffled = True


    # Output one batch
    def sample_batch(self) -> tuple[np.ndarray, np.ndarray] | None:

        """Execute the core functionality of ``sample_batch``.

        Restore CSR or dense minibatches from filtered HyS sparse tables and serve PCA,
        KMeans, and large-scale training.

        The function directly reads from or writes to related tables in the Atlas
        database and reduces memory usage as much as possible through SQL, chunked
        reading, or streaming computation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            Returns ``None`` if the shuffle buffer has not yet been filled and
            shuffled.

            If the buffer has already been shuffled, returns a tuple
            ``(X_batch, filter_cell_ids)``.

            ``X_batch`` is the current dense minibatch with shape
            ``(batch_size, gene_num)``. ``filter_cell_ids`` contains the
            corresponding ``filter_cell_id`` for each row in the batch, ensuring
            that ``X_batch[i, :]`` matches ``filter_cell_ids[i]``.

        Examples
        --------
        Call this function::

            sap.sample_batch(...)
        """

        if not self.shuffled:  # If the buffer is not full yet
            return None

        start = self.output_batch_id * self.batch_size
        end = start + self.batch_size

        X_batch = self.X[start:end]
        filter_cell_ids = self.filter_cell_ids[start:end]

        self.output_batch_id += 1

        # If all batches have already been output
        if self.output_batch_id == self.buffer_batch_num:
            # Reset buffer state
            self.write_ptr = 0
            self.output_batch_id = 0
            self.shuffled = False

        return X_batch, filter_cell_ids


    # Output remaining batches that did not fill the buffer,
    # preventing zero output when the dataset has fewer batches than buffer_batch_num
    def flush_remaining(self) -> list[tuple[np.ndarray, np.ndarray]]:

        """Execute the core functionality of ``flush_remaining``.

        Restore CSR or dense minibatches from filtered HyS sparse tables and serve PCA,
        KMeans, and large-scale training.

        The function directly reads from or writes to related tables in the Atlas
        database and reduces memory usage as much as possible through SQL, chunked
        reading, or streaming computation.

        The function flushes any buffered minibatches that were not emitted by a
        full buffer cycle.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Examples
        --------
        Call this function::

            sap.flush_remaining(...)
        """

        if self.write_ptr == 0:
            return []

        n_cells = self.write_ptr

        # Shuffle only the cells that have already been written; X and filter_cell_ids use the same permutation
        perm = np.random.permutation(n_cells)
        X_remain = self.X[:n_cells][perm]
        filter_cell_ids_remain = self.filter_cell_ids[:n_cells][perm]

        batches = []

        start = 0
        while start < n_cells:
            end = min(start + self.batch_size, n_cells)
            batches.append((
                X_remain[start:end].copy(),
                filter_cell_ids_remain[start:end].copy(),
            ))
            start = end

        # reset
        self.write_ptr = 0
        self.output_batch_id = 0
        self.shuffled = False

        return batches


class MultiThreadedMinibatchFetcher:

    """Multithreaded minibatch reader.

    This class restores CSR or dense expression matrices batch by batch from the
    filtered HyS tables in Atlas, uses a producer/queue structure to prefetch data,
    and ensures that the consumer receives results in batch order.
    It is the underlying implementation of ``atlas.get_minibatch_csr`` and
    ``atlas.get_minibatch_dense``.

    Parameters
    ----------
    file_path
        Path to the Atlas ``.sasql`` database file.

    batch_size
        Number of cells in each minibatch.

    x_type
        Output matrix type. Common values are ``"CSR"`` or ``"dense"``.

    pass_mode
        Traversal mode. ``"single-pass"`` traverses once in order, while
        ``"multi-pass"`` can be combined with ``ShuffleBuffer`` for randomized
        multi-batch training.

    buffer_batch_num
        Number of batches cached by the shuffle buffer in ``multi-pass`` mode.

    max_batches
        Maximum number of minibatches to output; if ``None``, there is no limit.

    return_cell_ids
        Whether to carry the row-wise corresponding ``filter_cell_id`` in the output.
        Disabled by default to maintain compatibility with the old matrix-only output.

    Notes
    -----
    This is an internal streaming reader. Regular users usually read minibatches through
    Atlas object methods.

    Examples
    --------
    Read data as dense batches through an Atlas object::

        atlas.build_read_index(use_hvg=True)
        for X_batch in atlas.get_minibatch_dense(batch_size=2048):
            print(X_batch.shape)
            break

    Directly create a reader for low-level debugging::

        fetcher = MultiThreadedMinibatchFetcher(
            atlas.file_path,
            batch_size=1024,
            x_type="CSR",
            pass_mode="single-pass",
            max_batches=10,
        )
        for batch in fetcher.run():
            X_batch = batch["X"] if isinstance(batch, dict) else batch
            print(X_batch.shape)"""

    def __init__(self, file_path: PathLike[str] | str,
                 batch_size: int=2048,
                 x_type: str= "CSR",
                 pass_mode: str="multi-pass",
                 buffer_batch_num: int=5,
                 max_batches: int | None=None,  # Maximum number of batches to output
                 return_cell_ids: bool=False,
                 ):

        """Initialize the object.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Parameters
        ----------
        file_path
            Input file path or Atlas ``.sasql`` database file path.

        batch_size
            Number of cells to read, write, or process in each batch; larger values
            are usually faster but consume more memory.

        x_type
            Output matrix type, usually ``"CSR"`` or ``"dense"``.

        pass_mode
            Minibatch traversal mode, usually ``"single-pass"`` or ``"multi-pass"``.

        buffer_batch_num
            Number of minibatches cached in the shuffle buffer.

        max_batches
            Maximum number of minibatches to output; if ``None``, there is no limit.

        return_cell_ids
            Whether to return ``filter_cell_ids`` together with the output batch.
            Defaults to ``False``, preserving the old behavior of returning only the
            matrix. If ``True``, returns
            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids}``.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        self.X_type = x_type  # Output X table format: "CSR" or "dense" (wide table)
        self.file_path = fspath(file_path)  # Absolute path to the sasql file
        self.batch_size = batch_size
        self.producer_num = 10  # Number of threads
        self.gene_num = self._get_gene_num()  # Get the number of genes
        self.index_data = self._get_index_data()  # Read the information saved by build_read_index(use_data=...) from the database, such as "data_log1p" or "data_scale"
        self.zero_scale_transform = self._get_zero_scale_transform()
        # Get zero_scale_transform from the var table, that is, (0 - g.mean) / g.std for each gene

        # Number of batches to output in this round
        self.max_batches = max_batches
        self.return_cell_ids = return_cell_ids
        self.stop_event = threading.Event()  # Early stop signal

        self.out_queue = queue.Queue(maxsize=20)  # Output queue

        self.fetch_size = 500_0000  # Size for streaming reading with fetch_record_batch
        self.pass_mode = pass_mode  # single-pass: single traversal; multi-pass: multiple traversals
        self.buffer_batch_num = buffer_batch_num  # For multiple traversals, buffer capacity; n means batch_size * n

        self.indptr_queue = self._prepare_indptr()  # Get the rb data stream for reading indptr

        self.batch_cell_counts, self.batch_nnz = self._prepare_batch_info_sql()  # Get batch_nnz, the number of nonzero values per cell batch
        self.batch_idx = 0  # Batch index
        self.batch_num = len(self.batch_nnz)  # Number of batches

        self.queue = queue.Queue(maxsize=self.producer_num * 5)  # Data cache queue: Queue (core)

        # Ring Buffer: circular buffer pool used to split batches
        self.pool_size = self.fetch_size * 10  # Capacity
        self.pool_gene_id = np.empty(self.pool_size, dtype=np.uint16)
        self.pool_data = np.empty(self.pool_size, dtype=np.float32)

        self.read_ptr = 0  # Read pointer
        self.write_ptr = 0  # Write pointer
        self.used_size = 0  # Current nnz data in the Ring Buffer
        self.total_batches = 0  # Counter

        # Output speed statistics
        self.output_start_time = None
        self.output_last_time = None
        self.output_cells = 0
        self.speed_log_every = 5  # Print speed every N output batches; set to 1 if you want to print every batch


    def _get_zero_scale_transform(self):
        """Get the fill values for sparse zero positions in the dense matrix.

        If the current build_read_index uses data_scale, the data has been scaled,
        and the original 0 values in the sparse matrix need to be filled with
        var.zero_scale_transform.

        If the current data is data, data_normalize, data_log1p, or other unscaled data,
        the original 0 positions in the sparse matrix should be filled with 0.0.
        """

        logger.info(
            f"[Minibatch] index_data = {self.index_data!r}; "
        )

        # zero_scale_transform only needs to be read for data_scale
        if self.index_data != "data_scale":
            logger.info(
                f"[Minibatch] read index use_data={self.index_data!r}; "
                "use 0.0 as dense fill value."
            )
            return np.zeros(self.gene_num, dtype=np.float32)

        conn = duckdb.connect(self.file_path)

        try:
            # New: even for data_scale, first check whether the field exists
            var_columns = conn.execute("PRAGMA table_info('var')").fetchdf()["name"].tolist()

            if "zero_scale_transform" not in var_columns:
                logger.info(
                    "[Minibatch] read index use_data='data_scale', "
                    "but var.zero_scale_transform not found; "
                    "use 0.0 as dense fill value."
                )
                return np.zeros(self.gene_num, dtype=np.float32)

            arr = conn.execute("""
                   SELECT zero_scale_transform
                   FROM var
                   WHERE filter_gene_id IS NOT NULL
                   ORDER BY filter_gene_id
               """).fetchnumpy()["zero_scale_transform"]

            return arr.astype("float32")

        finally:
            conn.close()


    def _get_index_data(self) -> str | None:
        """Read the expression value field used by the current read index from the database."""

        conn = duckdb.connect(self.file_path)

        try:
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

            if "atlas_read_index_meta" not in tables:
                return None

            row = conn.execute("""
                SELECT value
                FROM atlas_read_index_meta
                WHERE key = 'use_data'
            """).fetchone()

            if row is None:
                return None

            return row[0]

        finally:
            conn.close()


    def _get_gene_num(self):
        """Get internal information from the database or object.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Key tables accessed or generated by the current implementation include ``var``.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        conn = duckdb.connect(self.file_path)
        gene_num = conn.execute(
            "SELECT COUNT(*) FROM var WHERE filter_gene_id IS NOT NULL"
        ).fetchone()[0]
        ("gene_num:", gene_num)
        conn.close()
        return gene_num


    def _prepare_indptr(self):
        """Prepare indptr data read in ``filter_cell_id`` order.

        The returned record batch has ``filter_cell_id`` as column 0 and cumulative
        ``indptr`` as column 1.
        The consumer uses both to ensure that ``X_batch[i, :]`` corresponds to
        ``filter_cell_ids[i]``.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Key tables accessed or generated by the current implementation include
        ``X_HyS_indptr_filtered``.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        fetch_record_indptr = conn.execute(
            """
            SELECT
                filter_cell_id,
                indptr
            FROM X_HyS_indptr_filtered
            ORDER BY filter_cell_id
            """

        ).fetch_record_batch(rows_per_batch=self.batch_size)

        q = queue.Queue()
        for rb in fetch_record_indptr:
            q.put(rb)
        conn.close()
        return q


    def _prepare_batch_info_sql(self):
        """Execute the core functionality of ``_prepare_batch_info_sql``.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Key tables accessed or generated by the current implementation include
        ``X_HyS_indptr_filtered``.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        query = f"""
        WITH t AS (
            SELECT
                filter_cell_id,
                indptr,
                ROW_NUMBER() OVER (ORDER BY filter_cell_id) - 1 AS rn
            FROM X_HyS_indptr_filtered
        ),
        b AS (
            SELECT
                rn // {self.batch_size} AS batch_id,
                COUNT(*) AS n_cells,
                MAX(indptr) AS end_indptr
            FROM t
            GROUP BY batch_id
        )
        SELECT
            batch_id,
            CAST(n_cells AS INTEGER) AS n_cells,
            CAST(
                end_indptr - LAG(end_indptr, 1, 0) OVER (ORDER BY batch_id)
                AS BIGINT
            ) AS batch_nnz
        FROM b
        ORDER BY batch_id
        """

        rows = conn.execute(query).fetchall()
        conn.close()

        batch_cell_counts = [int(r[1]) for r in rows]
        batch_nnz = [int(r[2]) for r in rows]

        return batch_cell_counts, batch_nnz


    def _producer(self, tid: int):
        """Execute the core functionality of ``_producer``.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Key tables accessed or generated by the current implementation include
        ``X_HyS_data_filtered``.

        Parameters
        ----------
        tid
            Producer thread or data shard ID.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        query = f"""
            SELECT
                rowid,
                filter_gene_id,
                data
            FROM X_HyS_data_filtered
            WHERE tid = {tid}
            ORDER BY rowid
        """

        result = conn.execute(query).fetch_record_batch(
            rows_per_batch=self.fetch_size
        )

        rb_count = 0
        nnz_count = 0

        try:
            for rb in result:

                # New: stop early
                if self.stop_event.is_set():
                    break

                # Read rowid / gene_id / data
                rowids = rb.column(0).to_numpy().astype(np.int64)
                gene_id = rb.column(1).to_numpy().astype(np.uint16)
                data = rb.column(2).to_numpy().astype(np.float32)

                if len(rowids) == 0:
                    continue

                # Calculate seq_id using the real rowid
                seq_start = int(rowids[0] // self.fetch_size)  # Which block the first rowid of the current rb belongs to
                seq_end = int(rowids[-1] // self.fetch_size)  # Which block the last rowid of the current rb belongs to

                # Safety check: in theory, one rb should belong to only one seq block
                # If it crosses blocks, rows_per_batch / tid sharding does not match
                if seq_start != seq_end:
                    raise RuntimeError(
                        f"[Producer-{tid}] one record batch spans multiple seq blocks: "
                        f"{seq_start} -> {seq_end}, "
                        f"rowid_start={rowids[0]}, rowid_end={rowids[-1]}, "
                        f"fetch_size={self.fetch_size}"
                    )

                seq_id = seq_start

                while not self.stop_event.is_set():
                    try:
                        self.queue.put((seq_id, gene_id, data), timeout=0.5)
                        break
                    except queue.Full:
                        continue

                rb_count += 1
                nnz_count += len(gene_id)

        finally:
            conn.close()

            # Notify the consumer that this producer has finished
            try:
                self.queue.put(None, timeout=0.5)
            except queue.Full:
                pass


    def _consumer(self):
        """Execute the core functionality of ``_consumer``.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """

        reorder_buffer = {}  # Out-of-order data cache: reorder_buffer[seq_id] = (gene_id, data); its size is dynamic

        expected_seq = 0  # The next desired batch sequence number
        global_indptr_offset = 0  # Used to correct the cumulative offset of indptr

        prepared_batches = 0  # Number of original batches already read, constructed as dense, and placed into ShuffleBuffer

        # Build the output buffer for the wide table
        shuffle_buffer = ShuffleBuffer(
            gene_num=self.gene_num,
            batch_size=self.batch_size,
            buffer_batch_num=self.buffer_batch_num
        )

        # Current batch index < number of batches
        while self.batch_idx < self.batch_num:

            # If enough max_batches have already been prepared/output, end this round early
            if self._read_limit_reached(prepared_batches):
                logger.debug(
                    f"[Consumer] read limit reached, "
                    f"batch_idx={self.batch_idx}, "
                    f"prepared_batches={prepared_batches}, "
                    f"output_batches={self.total_batches}"
                )
                self.stop_event.set()
                break

            need = self.batch_nnz[self.batch_idx]  # nnz required by the current batch
            current_batch_cells = self.batch_cell_counts[
                self.batch_idx]  # Real number of cells in the current batch; the last batch may be smaller than self.batch_size

            # The data in RingBuffer is not enough; fill RingBuffer until it is enough for one batch
            while self.used_size < need:

                # If no more reading is needed, break to avoid waiting on the queue forever
                if self._read_limit_reached(prepared_batches):
                    self.stop_event.set()
                    break

                # Add timeout to avoid the consumer being permanently blocked after producers stop early
                try:
                    item = self.queue.get(timeout=0.5)  # Get the next item from Queue, blocking wait, with built-in lock
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if item is None:  # A producer has completed its task -> sentinel None, not stored in buffer
                    continue

                seq_id, gene_id, data = item  # Parse data from queue: item = (seq_id, gene_id, data)
                reorder_buffer[seq_id] = (gene_id, data)  # Store into the out-of-order data cache

                # Take data in order. The currently needed batch sequence number is expected_seq,
                # and the data is in the out-of-order data cache reorder_buffer
                while expected_seq in reorder_buffer:

                    gene_id, data = reorder_buffer.pop(expected_seq)  # Take out the needed data
                    length = len(gene_id)

                    # Write into Ring Buffer: circular buffer pool used to split batches
                    end_space = self.pool_size - self.write_ptr

                    if length <= end_space:  # Sequential write
                        self.pool_gene_id[self.write_ptr:self.write_ptr + length] = gene_id
                        self.pool_data[self.write_ptr:self.write_ptr + length] = data

                    else:  # Cross-boundary write
                        self.pool_gene_id[self.write_ptr:] = gene_id[:end_space]
                        self.pool_gene_id[:length - end_space] = gene_id[end_space:]

                        self.pool_data[self.write_ptr:] = data[:end_space]
                        self.pool_data[:length - end_space] = data[end_space:]

                    self.write_ptr = (self.write_ptr + length) % self.pool_size
                    self.used_size += length
                    expected_seq += 1

            # If the loop exits because of max_batches / stop_event and RingBuffer is still not enough for the current batch, end the main loop
            if self.used_size < need:
                break

            # The data in RingBuffer is enough for one batch
            if self.used_size >= need:

                end_space = self.pool_size - self.read_ptr

                if need <= end_space:  # Sequential read
                    vals = self.pool_data[self.read_ptr:self.read_ptr + need]
                    cols = self.pool_gene_id[self.read_ptr:self.read_ptr + need]

                else:  # Cross-boundary read in two parts
                    first_len = end_space
                    second_len = need - first_len

                    vals = np.empty(need, dtype=self.pool_data.dtype)
                    cols = np.empty(need, dtype=self.pool_gene_id.dtype)

                    # Tail part
                    vals[:first_len] = self.pool_data[self.read_ptr:]
                    cols[:first_len] = self.pool_gene_id[self.read_ptr:]

                    # Head part
                    vals[first_len:] = self.pool_data[:second_len]
                    cols[first_len:] = self.pool_gene_id[:second_len]

                self.read_ptr = (self.read_ptr + need) % self.pool_size
                self.used_size -= need

                # Build filter_cell_ids and indptr for the current batch
                indptr_rb = self.indptr_queue.get()

                # The filter_cell_id corresponding to each row of the current X
                filter_cell_ids = np.array(indptr_rb.column(0), dtype=np.int64)

                # The original cumulative indptr is in column 1
                indptr_raw = np.array(indptr_rb.column(1), dtype=np.int64)
                last_val = indptr_raw[-1]
                indptr_now = np.concatenate(([0], indptr_raw - global_indptr_offset))
                global_indptr_offset = last_val

                # Check whether the number of indptr rows equals the number of cells in the current batch
                if len(indptr_now) != current_batch_cells + 1:
                    raise RuntimeError(
                        f"[Consumer] indptr length mismatch: "
                        f"len(indptr_now)={len(indptr_now)}, "
                        f"current_batch_cells={current_batch_cells}, "
                        f"batch_idx={self.batch_idx}"
                    )

                if len(filter_cell_ids) != current_batch_cells:
                    raise RuntimeError(
                        f"[Consumer] filter_cell_ids length mismatch: "
                        f"len(filter_cell_ids)={len(filter_cell_ids)}, "
                        f"current_batch_cells={current_batch_cells}, "
                        f"batch_idx={self.batch_idx}"
                    )

                if self.X_type == "CSR":
                    # Output type 1: CSR format
                    X = sp.csr_matrix((current_batch_cells, self.gene_num), dtype=np.float32)

                    X.data = vals.copy()
                    X.indices = cols.copy()
                    X.indptr = indptr_now

                    self._put_output(
                        X,
                        filter_cell_ids.copy(),
                    )

                if self.X_type == "dense":
                    # Output type 2: dense format

                    X_dense = np.empty((current_batch_cells, self.gene_num), dtype=np.float32)
                    X_dense[:] = self.zero_scale_transform  # Fill by gene_id using self.zero_scale_transform
                    # zero_scale_transform stores each gene's (0 - g.mean) / g.std in the corresponding field of the var table for future use
                    # If scale() has not been run, the current data is still uncentered data such as normalize/log1p,
                    # so self.zero_scale_transform is filled with all zeros
                    # X_dense =
                    # [ Fill each gene with zero_scale_transform
                    #  [-0.5, 0.2, -1.1, ...],
                    #  [-0.5, 0.2, -1.1, ...],
                    #  ...
                    # ]
                    rows = np.repeat(  # [0,0, 1, 2,2,2] Row index corresponding to each nonzero element
                        np.arange(current_batch_cells),  # [0,1,2,...] Represents each cell (row)
                        np.diff(indptr_now)  # [2, 1, 3, ...] Number of nonzero values (nnz) for each cell
                    )
                    X_dense[rows, cols] = vals
                    # Write nonzero values into the corresponding rows and columns
                    # X_dense[0,1] = 10
                    # X_dense[0,3] = 20

                    if self.pass_mode == "single-pass":  # Single traversal
                        self._put_output(
                            X_dense.copy(),
                            filter_cell_ids.copy(),
                        )

                    if self.pass_mode == "multi-pass":  # Multiple traversals; add to buffer to ensure randomness across passes

                        # In multi-pass mode, count prepared_batches first
                        # This batch has already entered ShuffleBuffer,
                        # so even if it is not output temporarily, it will be output later by flush_remaining.
                        shuffle_buffer.add_batch(
                            X_dense,
                            filter_cell_ids,
                        )  # Write into the output cache shuffle buffer
                        prepared_batches += 1

                        # Once ShuffleBuffer is full, immediately output all batches from the shuffled buffer.
                        while True:
                            batch_random = shuffle_buffer.sample_batch()  # Randomly sample a batch from the output buffer to ensure randomness across multiple passes

                            if batch_random is None:
                                break

                            X_dense_random, filter_cell_ids_random = batch_random

                            ok = self._put_output(
                                X_dense_random.copy(),  # The same buffer will be reused every round; without copy, it will be overwritten
                                filter_cell_ids_random.copy(),
                            )

                            if not ok or self._output_limit_reached():
                                break

                        # If enough max_batches have already been prepared, do not continue reading new batches;
                        # let tail flush output the remaining data.
                        if self._read_limit_reached(prepared_batches):
                            self.stop_event.set()

                self.batch_idx += 1

        # multi-pass mode: output tail batches in ShuffleBuffer that did not fill the buffer
        if self.X_type == "dense" and self.pass_mode == "multi-pass":

            remain_batches = shuffle_buffer.flush_remaining()

            for X_remain, filter_cell_ids_remain in remain_batches:

                # Prevent tail output from exceeding max_batches
                if self._output_limit_reached():
                    self.stop_event.set()
                    break

                self._put_output(
                    X_remain.copy(),
                    filter_cell_ids_remain.copy(),
                )

        logger.info(
            f"[Done] processed_batches={self.batch_idx}, "
            f"output_batches={self.total_batches}"
        )

        # Notify run() to finish
        self.out_queue.put(None)


    def run(self) -> Iterator[sp.csr_matrix | np.ndarray | dict[str, np.ndarray]]:
        """Execute the core functionality of ``run``.

        Restore CSR or dense minibatches from filtered HyS sparse tables and serve PCA,
        KMeans, and large-scale training.

        The function directly reads from or writes to related tables in the Atlas
        database and reduces memory usage as much as possible through SQL, chunked
        reading, or streaming computation.

        The function yields minibatches restored from Atlas database tables for
        reuse in downstream steps.

        Yields
        -------
        batch
            When ``return_cell_ids=False``, CSR or dense matrices are generated batch by batch;
            when ``return_cell_ids=True``, dictionaries are generated batch by batch:

            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids}``.

        Examples
        --------
        Call this function::

            sap.run(...)
        """

        # producers: multithreaded
        producers = []
        for i in range(self.producer_num):
            t = threading.Thread(target=self._producer, args=(i,))
            t.start()
            producers.append(t)

        # consumer: single-threaded
        consumer = threading.Thread(target=self._consumer)
        consumer.start()

        # Yield uniformly from out_queue
        while True:
            batch = self.out_queue.get()  # Blocking
            if batch is None:  # Sentinel received, indicating all batches have been output
                break
            yield batch  # Normal batch continues to be yielded outward

        for t in producers:
            t.join()

        consumer.join()


    # Helper function 1: whether the output limit has been reached
    def _output_limit_reached(self):

        """Execute the core functionality of ``_output_limit_reached``.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """
        return (
                self.max_batches is not None
                and self.total_batches >= self.max_batches
        )


    # Helper function 2: whether reading new batches should stop
    def _read_limit_reached(self, prepared_batches: int):

        """Execute the core functionality of ``_read_limit_reached``.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Parameters
        ----------
        prepared_batches
            Number of batches that have already been prepared and put into the shuffle buffer.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """
        if self.max_batches is None:
            return False

        # In dense + multi-pass mode, batches enter ShuffleBuffer first
        # and may not be output immediately, so prepared_batches is checked
        if self.X_type == "dense" and self.pass_mode == "multi-pass":
            return prepared_batches >= self.max_batches

        # In other cases, reading is basically followed by output, so total_batches is checked
        return self.total_batches >= self.max_batches


    # Helper function 3: unified batch output
    def _put_output(
        self,
        X_batch: sp.csr_matrix | np.ndarray,
        filter_cell_ids: np.ndarray,
    ):

        """Execute the core functionality of ``_put_output``.

        This internal function belongs to the minibatch streaming reading module and
        supports public APIs in the same module.

        It restores CSR or dense minibatches from filtered HyS sparse tables and serves
        PCA, KMeans, and large-scale training.

        It is usually not called directly as a user-facing entry point. When called
        directly, the caller must ensure that the input object, database connection,
        and related temporary tables have already been prepared by upstream steps.

        Parameters
        ----------
        X_batch
            Expression matrix or embedding matrix of the current batch.

        filter_cell_ids
            The ``filter_cell_id`` corresponding to each row in the current batch.

        Returns
        -------
        result
            Function return value. The specific type depends on the parameter settings
            and internal execution path.

        Notes
        -----
        This is an internal helper. Unless extending the internal workflow of
        scAtlasPy, it is generally not recommended to call it directly in user code.
        """
        if self._output_limit_reached():
            self.stop_event.set()
            return False

        if len(filter_cell_ids) != X_batch.shape[0]:
            raise RuntimeError(
                f"the length of filter_cell_ids must equal the number of rows in X_batch: "
                f"len(filter_cell_ids)={len(filter_cell_ids)}, "
                f"X_batch.shape[0]={X_batch.shape[0]}"
            )

        if self.return_cell_ids:
            batch = {
                "X": X_batch,
                "filter_cell_ids": np.asarray(filter_cell_ids, dtype=np.int64),
            }
        else:
            batch = X_batch

        self.out_queue.put(batch)
        self.total_batches += 1

        # Current speed + average speed
        now = time.perf_counter()

        if self.output_start_time is None:
            self.output_start_time = now
            self.output_last_time = now

        n_cells = X_batch.shape[0]
        self.output_cells += n_cells

        interval = now - self.output_last_time
        elapsed = now - self.output_start_time

        current_batch_speed = 1.0 / interval if interval > 0 else 0.0
        current_cell_speed = n_cells / interval if interval > 0 else 0.0

        avg_batch_speed = self.total_batches / elapsed if elapsed > 0 else 0.0
        avg_cell_speed = self.output_cells / elapsed if elapsed > 0 else 0.0

        if self.total_batches % self.speed_log_every == 0:
            logger.info(
                f"[Speed] output_batches={self.total_batches}, "
                f"[ current={current_batch_speed:.2f} batch/s, "
                f"{current_cell_speed:.0f} cells/s, ]"
                f"[ avg={avg_batch_speed:.2f} batch/s, "
                f"{avg_cell_speed:.0f} cells/s ]"
            )

        self.output_last_time = now

        if self._output_limit_reached():
            logger.info(f"[Consumer] reach max_batches={self.max_batches}, stop")
            self.stop_event.set()

        return True
