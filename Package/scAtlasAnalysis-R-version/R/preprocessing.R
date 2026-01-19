# =============================================================================
# preprocessing.R - 数据预处理函数
# =============================================================================
#
# 功能：单细胞数据的预处理操作
# 包括：筛选、归一化、转换、高变基因识别等
#
# 不可变操作（返回新对象）：
#   - filter_cells(), filter_genes()
#
# 原地操作（修改自身，返回invisible）：
#   - normalize_total(), log1p(), scale(), highly_variable_genes()
#   - calculate_qc_metrics(), calculate_cell_total_counts()
#   - calculate_gene_total_counts(), exp1(), normalize_and_log1p()
#
# =============================================================================

# -----------------------------------------------------------------------------
# Filter Cells
# -----------------------------------------------------------------------------
#' Filter Low-Quality Cells
#'
#' Filters cells based on total counts and number of expressed genes.
#' Returns a NEW atlas object with filtered cell IDs (immutable style).
#'
#' @param a An atlas object.
#' @param min_counts Integer. Minimum total counts per cell.
#' @param min_genes Integer. Minimum number of genes expressed per cell.
#' @param max_counts Integer. Maximum total counts per cell.
#' @param max_genes Integer. Maximum number of genes expressed per cell.
#' @param col_name Character. Column name for filter flag in obs table.
#'
#' @return A NEW atlas object (view with filtered cells). The original atlas
#'   is not modified.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas_filtered <- filter_cells(atlas, min_counts = 100, min_genes = 200)
filter_cells <- function(a, ...) {
  UseMethod("filter_cells")
}

#' @export
filter_cells.atlas <- function(a,
                               min_counts = NULL,
                               min_genes = NULL,
                               max_counts = NULL,
                               max_genes = NULL,
                               col_name = "filter_cells_1",
                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  # Validate at least one filter is specified
  if (all(sapply(list(min_counts, min_genes, max_counts, max_genes), is.null))) {
    stop("At least one filter condition must be specified")
  }

  con <- a$con

  # Get total cell count
  total <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs")$n[1]
  message(sprintf("[filter_cells] Total cells: %s", format(total, big.mark = ",")))

  # Add filter column to obs table
  DBI::dbExecute(con, sprintf("
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS %s BOOLEAN DEFAULT FALSE
  ", col_name))

  # Build filter conditions
  conds <- c()
  if (!is.null(min_counts)) conds <- c(conds, sprintf("sum_expr >= %d", min_counts))
  if (!is.null(max_counts)) conds <- c(conds, sprintf("sum_expr <= %d", max_counts))
  if (!is.null(min_genes)) conds <- c(conds, sprintf("nonzero >= %d", min_genes))
  if (!is.null(max_genes)) conds <- c(conds, sprintf("nonzero <= %d", max_genes))
  condition <- paste(conds, collapse = " AND ")

  # Find cells meeting criteria (using CSR data for efficiency)
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _filter_keep AS
    SELECT cell_index
    FROM (
      SELECT cell_index, SUM(data) AS sum_expr, COUNT(*) AS nonzero
      FROM X_CSR_data
      GROUP BY cell_index
    ) WHERE %s
  ", condition))

  # Update filter column
  DBI::dbExecute(con, sprintf("UPDATE obs SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE obs SET %s = TRUE
    WHERE id IN (SELECT cell_index FROM _filter_keep)
  ", col_name))

  # Get filtered cell IDs
  keep_ids <- DBI::dbGetQuery(con, sprintf(
    "SELECT cell_id FROM obs WHERE %s = TRUE", col_name
  ))$cell_id

  # Report statistics
  n_keep <- length(keep_ids)
  message(sprintf("[filter_cells] Kept: %s (%.1f%%)",
         format(n_keep, big.mark = ","), 100 * n_keep / total))

  # Clean up
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")

  # Return new atlas object (view with filtered cells)
  .new_atlas_view(a, obs_cell_id = keep_ids)
}


# -----------------------------------------------------------------------------
# Filter Genes
# -----------------------------------------------------------------------------
#' Filter Low-Quality Genes
#'
#' Filters genes based on total counts and number of cells expressing them.
#' Returns a NEW atlas object with filtered gene IDs (immutable style).
#'
#' @param a An atlas object.
#' @param min_counts Integer. Minimum total counts per gene.
#' @param min_cells Integer. Minimum number of cells expressing the gene.
#' @param max_counts Integer. Maximum total counts per gene.
#' @param max_cells Integer. Maximum number of cells expressing the gene.
#' @param col_name Character. Column name for filter flag in var table.
#'
#' @return A NEW atlas object (view with filtered genes).
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas_filtered <- filter_genes(atlas, min_counts = 10, min_cells = 3)
filter_genes <- function(a, ...) {
  UseMethod("filter_genes")
}

#' @export
filter_genes.atlas <- function(a,
                               min_counts = NULL,
                               min_cells = NULL,
                               max_counts = NULL,
                               max_cells = NULL,
                               col_name = "filter_genes_1",
                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  if (all(sapply(list(min_counts, min_cells, max_counts, max_cells), is.null))) {
    stop("At least one filter condition must be specified")
  }

  con <- a$con
  total <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM var")$n[1]
  message(sprintf("[filter_genes] Total genes: %s", format(total, big.mark = ",")))

  # Add filter column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS %s BOOLEAN DEFAULT FALSE
  ", col_name))

  # Build conditions
  conds <- c()
  if (!is.null(min_counts)) conds <- c(conds, sprintf("sum_expr >= %d", min_counts))
  if (!is.null(max_counts)) conds <- c(conds, sprintf("sum_expr <= %d", max_counts))
  if (!is.null(min_cells)) conds <- c(conds, sprintf("nonzero >= %d", min_cells))
  if (!is.null(max_cells)) conds <- c(conds, sprintf("nonzero <= %d", max_cells))
  condition <- paste(conds, collapse = " AND ")

  # Find genes meeting criteria
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _filter_keep AS
    SELECT indices AS gene_index
    FROM (
      SELECT indices, SUM(data) AS sum_expr, COUNT(*) AS nonzero
      FROM X_CSR_data
      GROUP BY indices
    ) WHERE %s
  ", condition))

  # Update filter column
  DBI::dbExecute(con, sprintf("UPDATE var SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE var SET %s = TRUE
    WHERE id IN (SELECT gene_index FROM _filter_keep)
  ", col_name))

  keep_ids <- DBI::dbGetQuery(con, sprintf(
    "SELECT gene_id FROM var WHERE %s = TRUE", col_name
  ))$gene_id

  n_keep <- length(keep_ids)
  message(sprintf("[filter_genes] Kept: %s (%.1f%%)",
         format(n_keep, big.mark = ","), 100 * n_keep / total))

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")

  .new_atlas_view(a, var_gene_id = keep_ids)
}


# -----------------------------------------------------------------------------
# Normalize Total
# -----------------------------------------------------------------------------
#' Normalize Total Counts Per Cell
#'
#' Computes scale factors for each cell to normalize to a target sum.
#' Stores scale factors in the obs table. Modifies the atlas in-place.
#'
#' @param a An atlas object.
#' @param target_sum Numeric. Target sum for normalization (default: 10000).
#'   If NULL, uses the median of cell totals.
#' @param col_name Character. Column name for scale factors in obs table.
#' @param field Character. Data field to use for calculation.
#'
#' @return The atlas object (modified in-place, returned for chaining).
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- normalize_total(atlas, target_sum = 10000)
normalize_total <- function(a, ...) {
  UseMethod("normalize_total")
}

#' @export
normalize_total.atlas <- function(a,
                                  target_sum = 10000,
                                  col_name = "scale_factor",
                                  field = "data",
                                  ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[normalize_total] Computing scale factors...")

  # Compute total counts per cell
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_sum")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _cell_sum AS
    SELECT cell_index, SUM(%s) AS total
    FROM X_CSR_data
    GROUP BY cell_index
  ", field))

  # Use median if target_sum is NULL
  if (is.null(target_sum)) {
    target_sum <- DBI::dbGetQuery(con, "SELECT median(total) AS m FROM _cell_sum")$m[1]
    message(sprintf("  Using median: %s", format(target_sum, digits = 6)))
  }

  # Add or update scale factor column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  DBI::dbExecute(con, sprintf("
    UPDATE obs
    SET %s = CASE WHEN s.total > 0 THEN %f / s.total ELSE 0 END
    FROM _cell_sum AS s
    WHERE obs.id = s.cell_index
  ", col_name, as.numeric(target_sum)))

  n <- DBI::dbGetQuery(con, sprintf(
    "SELECT COUNT(*) AS n FROM obs WHERE %s > 0", col_name
  ))$n[1]

  message(sprintf("[normalize_total] Scale factors computed: %s cells", format(n, big.mark = ",")))

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_sum")
  invisible(a)
}


# -----------------------------------------------------------------------------
# Log1p Transform
# -----------------------------------------------------------------------------
#' Log-Transform Expression Values
#'
#' Applies log(1 + x) transformation to expression values.
#' Adds a new column to X_CSR_data table. Modifies in-place.
#'
#' @param a An atlas object.
#' @param base Numeric. Logarithm base. NULL = natural log.
#' @param col_name Character. Output column name in X_CSR_data.
#' @param field Character. Input field to transform.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- log1p(atlas)
log1p <- function(a, ...) {
  UseMethod("log1p")
}

#' @export
log1p.atlas <- function(a,
                        base = NULL,
                        col_name = "log1p_factor",
                        field = "data",
                        chunk_size = 100000000,
                        ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[log1p] Applying log(1+x) transformation...")

  # Enable DuckDB parallel writes (use all available cores)
  tryCatch({
    n_threads <- parallel::detectCores()
    DBI::dbExecute(con, sprintf("SET threads = %d", n_threads))
    message(sprintf("  DuckDB threads set to %d", n_threads))
  }, error = function(e) {
    message("  Could not set DuckDB threads: ", e$message)
  })

  # Check if input field exists
  field_exists <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS cnt
    FROM information_schema.columns
    WHERE table_name = 'X_CSR_data'
      AND column_name = '%s'
  ", field))$cnt[1]

  if (field_exists == 0) {
    stop(sprintf("Field '%s' does not exist in X_CSR_data", field))
  }

  # Determine log expression
  if (is.null(base)) {
    log_expr <- sprintf("ln(1.0 + %s)", field)
    message("  Using natural logarithm (ln)")
  } else {
    log_expr <- sprintf("log(%f, 1.0 + %s)", as.numeric(base), field)
    message(sprintf("  Using base = %s", base))
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get record count for progress
  n_total <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM X_CSR_data")$n[1]
  if (n_total == 0) {
    message("  X_CSR_data is empty, skipping")
    return(invisible(a))
  }

  # Get id range for chunked updates
  id_range <- DBI::dbGetQuery(con, "SELECT MIN(id) AS min, MAX(id) AS max FROM X_CSR_data")
  min_id <- id_range$min[1]
  max_id <- id_range$max[1]
  n_chunks <- ceiling((max_id - min_id + 1) / chunk_size)

  message(sprintf("  Processing %s records in %d chunks (chunk_size = %s)...",
         format(n_total, big.mark = ","), n_chunks, format(chunk_size, big.mark = ",")))

  # Update in chunks to avoid memory issues
  for (i in seq_len(n_chunks)) {
    start_id <- min_id + (i - 1) * chunk_size
    end_id <- min(start_id + chunk_size - 1, max_id)

    message(sprintf("  Chunk %s/%s: [%s, %s]",
           i, n_chunks, format(start_id, big.mark = ","), format(end_id, big.mark = ",")))

    DBI::dbExecute(con, sprintf("
      UPDATE X_CSR_data
      SET %s = %s
      WHERE id BETWEEN %s AND %s
        AND %s IS NOT NULL
    ", col_name, log_expr, as.character(start_id), as.character(end_id), field))
  }

  message(sprintf("[log1p] Transformation complete: %s", col_name))
  invisible(a)
}


# -----------------------------------------------------------------------------
# Highly Variable Genes
# -----------------------------------------------------------------------------
#' Identify Highly Variable Genes
#'
#' Identifies highly variable genes based on variance or coefficient of variation.
#' Adds a boolean column to the var table.
#'
#' @param a An atlas object.
#' @param flavor Character. "var" = by variance, "cv" = by coefficient of variation.
#' @param n_top Integer. Number of top genes to mark (NULL = all).
#' @param col_name Character. Output column name in var table.
#' @param field Character. Data field to use for calculation.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- highly_variable_genes(atlas, n_top = 2000)
highly_variable_genes <- function(a, ...) {
  UseMethod("highly_variable_genes")
}

#' @export
highly_variable_genes.atlas <- function(a,
                                        flavor = c("var", "cv"),
                                        n_top = NULL,
                                        col_name = "highly_variable_genes",
                                        field = "data",
                                        ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  flavor <- match.arg(flavor)
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message(sprintf("[highly_variable_genes] flavor = %s", flavor))

  # Step 1: Compute gene-level statistics
  message("  Step 1: Computing gene statistics...")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_stats")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _hvg_stats AS
    SELECT
      indices AS gene_index,
      AVG(%s) AS mean,
      VAR_POP(%s) AS var,
      STDDEV_POP(%s) AS std
    FROM X_CSR_data
    WHERE %s IS NOT NULL
    GROUP BY indices
  ", field, field, field, field))

  # Step 2: Compute score
  message("  Step 2: Computing scores...")
  if (flavor == "var") {
    score_expr <- "var"
  } else {
    score_expr <- "CASE WHEN mean > 0 THEN std / mean ELSE 0 END"
  }

  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _hvg_score AS
    SELECT gene_index, %s AS score FROM _hvg_stats
  ", score_expr))

  # Step 3: Select top genes
  if (!is.null(n_top)) {
    message(sprintf("  Step 3: Selecting top %s genes...", format(n_top, big.mark = ",")))
    DBI::dbExecute(con, sprintf("
      CREATE TEMP TABLE _hvg_top AS
      SELECT gene_index FROM _hvg_score
      ORDER BY score DESC LIMIT %d
    ", as.integer(n_top)))
  } else {
    message("  Step 3: Marking all genes (n_top = NULL)")
    DBI::dbExecute(con, "
      CREATE TEMP TABLE _hvg_top AS
      SELECT gene_index FROM _hvg_score
    ")
  }

  # Step 4: Write to var table
  message(sprintf("  Step 4: Writing to var.%s", col_name))
  DBI::dbExecute(con, sprintf("
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS %s BOOLEAN
  ", col_name))

  DBI::dbExecute(con, sprintf("UPDATE var SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE var SET %s = TRUE
    WHERE id IN (SELECT gene_index FROM _hvg_top)
  ", col_name))

  # Count
  n_hvg <- DBI::dbGetQuery(con, sprintf(
    "SELECT SUM(CASE WHEN %s THEN 1 ELSE 0 END) AS n FROM var", col_name
  ))$n[1]

  message(sprintf("[highly_variable_genes] Found %s HVG", format(n_hvg, big.mark = ",")))

  # Clean up
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_stats")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_score")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_top")

  invisible(a)
}


# -----------------------------------------------------------------------------
# Scale (Z-score)
# -----------------------------------------------------------------------------
#' Z-Score Standardization
#'
#' Standardizes expression values to have zero mean and unit variance per gene.
#' Optionally clips values to a specified range.
#'
#' @param a An atlas object.
#' @param max_value Numeric. Clip values to [-max_value, max_value] (NULL = no clipping).
#' @param col_name Character. Output column name in X_CSR_data.
#' @param field Character. Input field to scale.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- scale(atlas, max_value = 10)
scale <- function(a, ...) {
  UseMethod("scale")
}

#' @export
scale.atlas <- function(a,
                        max_value = NULL,
                        col_name = "X_scale",
                        field = "data",
                        gene_chunk_size = 512,
                        use_hvg = FALSE,
                        hvg_key = "highly_variable",
                        ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[scale] Computing z-scores (DuckDB OLAP optimized)...")

  # Enable DuckDB parallel writes
  tryCatch({
    n_threads <- parallel::detectCores()
    DBI::dbExecute(con, sprintf("SET threads = %d", n_threads))
    message(sprintf("  DuckDB threads set to %d", n_threads))
  }, error = function(e) {
    message("  Could not set DuckDB threads: ", e$message)
  })

  # Check if input field exists
  field_exists <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS cnt
    FROM information_schema.columns
    WHERE table_name = 'X_CSR_data'
      AND column_name = '%s'
  ", field))$cnt[1]

  if (field_exists == 0) {
    stop(sprintf("Field '%s' does not exist in X_CSR_data", field))
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get gene list
  if (use_hvg) {
    message("  Using HVG gene subset")
    gene_ids <- DBI::dbGetQuery(con, sprintf("
      SELECT id FROM var
      WHERE %s = TRUE
      ORDER BY id
    ", hvg_key))$id
  } else {
    gene_ids <- DBI::dbGetQuery(con, "
      SELECT DISTINCT indices
      FROM X_CSR_data
      ORDER BY indices
    ")$indices
  }

  n_genes <- length(gene_ids)
  if (n_genes == 0) {
    message("  No genes found, skipping")
    return(invisible(a))
  }

  n_chunks <- ceiling(n_genes / gene_chunk_size)
  message(sprintf("  Total genes: %s", format(n_genes, big.mark = ",")))
  message(sprintf("  Gene chunk size: %s", format(gene_chunk_size, big.mark = ",")))
  message(sprintf("  Total chunks: %s", n_chunks))

  # Create temporary table for scaled values
  DBI::dbExecute(con, "
    CREATE TEMP TABLE _X_scale_tmp (
      id BIGINT,
      indices INTEGER,
      val REAL
    )
  ")

  # Process genes in chunks
  for (i in seq_len(n_chunks)) {
    chunk_start <- (i - 1) * gene_chunk_size
    chunk_genes <- gene_ids[(chunk_start + 1):min(chunk_start + gene_chunk_size, n_genes)]
    gene_list_sql <- paste(chunk_genes, collapse = ",")

    message(sprintf("  Chunk %s/%s: %s genes", i, n_chunks, length(chunk_genes)))

    # Compute gene-wise statistics
    DBI::dbExecute(con, sprintf("
      CREATE OR REPLACE TEMP TABLE _gene_stat AS
      SELECT
        indices,
        AVG(%s) AS mean,
        STDDEV_POP(%s) AS std
      FROM X_CSR_data
      WHERE indices IN (%s)
        AND %s IS NOT NULL
      GROUP BY indices
    ", field, field, gene_list_sql, field))

    # Compute z-scores with optional clipping
    if (is.null(max_value)) {
      DBI::dbExecute(con, sprintf("
        INSERT INTO _X_scale_tmp
        SELECT
          x.id,
          x.indices,
          CASE
            WHEN g.std > 0 THEN (x.%s - g.mean) / g.std
            ELSE 0
          END
        FROM X_CSR_data x
        JOIN _gene_stat g ON x.indices = g.indices
        WHERE x.indices IN (%s)
          AND x.%s IS NOT NULL
      ", field, gene_list_sql, field))
    } else {
      DBI::dbExecute(con, sprintf("
        INSERT INTO _X_scale_tmp
        SELECT
          x.id,
          x.indices,
          CASE
            WHEN g.std > 0 THEN
              LEAST(%.6f, GREATEST(-%.6f, (x.%s - g.mean) / g.std))
            ELSE 0
          END
        FROM X_CSR_data x
        JOIN _gene_stat g ON x.indices = g.indices
        WHERE x.indices IN (%s)
          AND x.%s IS NOT NULL
      ", as.numeric(max_value), as.numeric(max_value), field, gene_list_sql, field))
    }
  }

  # Merge scaled values back to X_CSR_data
  message("  Merging scaled values back to X_CSR_data...")
  DBI::dbExecute(con, sprintf("
    UPDATE X_CSR_data x
    SET %s = t.val
    FROM _X_scale_tmp t
    WHERE x.id = t.id AND x.indices = t.indices
  ", col_name))

  # Cleanup
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _X_scale_tmp")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _gene_stat")

  message(sprintf("[scale] Z-score complete: %s", col_name))
  invisible(a)
}


# -----------------------------------------------------------------------------
# Exp1 Transform (Inverse of log1p)
# -----------------------------------------------------------------------------
#' Inverse Log-Transform (exp(x) - 1)
#'
#' Applies exp(x) - 1 transformation, which is the inverse of log(1 + x).
#' Useful for converting log-transformed data back to original scale.
#' Adds a new column to X_CSR_data table. Modifies in-place.
#'
#' @param a An atlas object.
#' @param base Numeric. Exponential base. NULL = natural exp.
#' @param col_name Character. Output column name in X_CSR_data.
#' @param field Character. Input field to transform (default: "X_log1p").
#' @param chunk_size Integer. Number of records per chunk.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- log1p(atlas)
#' atlas <- exp1(atlas)  # Back to original scale
exp1 <- function(a, ...) {
  UseMethod("exp1")
}

#' @export
exp1.atlas <- function(a,
                       base = NULL,
                       col_name = "exp1_factor",
                       field = "log1p_factor",
                       chunk_size = 100000000,
                       ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[exp1] Applying exp(x) - 1 transformation...")

  # Enable DuckDB parallel writes
  tryCatch({
    n_threads <- parallel::detectCores()
    DBI::dbExecute(con, sprintf("SET threads = %d", n_threads))
    message(sprintf("  DuckDB threads set to %d", n_threads))
  }, error = function(e) {
    message("  Could not set DuckDB threads: ", e$message)
  })

  # Check if field exists
  col_exists <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'X_CSR_data' AND column_name = '%s'
  ", field))$n[1]

  if (col_exists == 0) {
    stop(sprintf("Field '%s' does not exist in X_CSR_data", field))
  }

  # Determine exp expression
  if (is.null(base)) {
    exp_expr <- sprintf("exp(%s) - 1.0", field)
    message("  Using natural exponential (exp)")
  } else {
    exp_expr <- sprintf("pow(%f, %s) - 1.0", as.numeric(base), field)
    message(sprintf("  Using base = %s", base))
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get id range
  id_range <- DBI::dbGetQuery(con, "SELECT MIN(id) AS min, MAX(id) AS max FROM X_CSR_data")
  min_id <- id_range$min[1]
  max_id <- id_range$max[1]

  if (is.null(min_id) || is.na(min_id)) {
    message("  X_CSR_data is empty, skipping")
    return(invisible(a))
  }

  n_total <- max_id - min_id + 1
  n_chunks <- ceiling(n_total / chunk_size)
  message(sprintf("  Processing %s records in %s chunks", format(n_total, big.mark = ","), n_chunks))

  # Update in chunks
  for (i in seq_len(n_chunks)) {
    start_id <- min_id + (i - 1) * chunk_size
    end_id <- min(start_id + chunk_size - 1, max_id)

    message(sprintf("  Chunk %s/%s: [%s, %s]",
           i, n_chunks, format(start_id, big.mark = ","), format(end_id, big.mark = ",")))

    DBI::dbExecute(con, sprintf("
      UPDATE X_CSR_data
      SET %s = %s
      WHERE id BETWEEN %s AND %s
        AND %s IS NOT NULL
    ", col_name, exp_expr, as.character(start_id), as.character(end_id), field))
  }

  message(sprintf("[exp1] Transformation complete: %s", col_name))
  invisible(a)
}


# -----------------------------------------------------------------------------
# Calculate QC Metrics
# -----------------------------------------------------------------------------
#' Calculate Quality Control Metrics
#'
#' Computes per-cell and per-gene QC metrics including:
#' - Cell: total_counts, n_genes_by_counts, total_counts_mt, pct_counts_mt
#' - Gene: total_counts, n_cells_by_counts
#' Also marks mitochondrial genes based on prefix.
#'
#' @param a An atlas object.
#' @param mt_prefix Character. Prefix for mitochondrial genes (default: "MT-").
#' @param mt_key Character. Column name for mt flag in var (default: "mt").
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- calculate_qc_metrics(atlas, mt_prefix = "MT-")
calculate_qc_metrics <- function(a, ...) {
  UseMethod("calculate_qc_metrics")
}

#' @export
calculate_qc_metrics.atlas <- function(a,
                                       mt_prefix = "MT-",
                                       mt_key = "mt",
                                       ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[calculate_qc_metrics] Computing QC metrics...")

  start_time <- Sys.time()

  # =================================================
  # Step 1: Mark mitochondrial genes in var
  # =================================================
  message("  Step 1: Marking mitochondrial genes...")
  DBI::dbExecute(con, sprintf("
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS %s BOOLEAN
  ", mt_key))

  DBI::dbExecute(con, sprintf("
    UPDATE var
    SET %s = CASE
      WHEN gene_id LIKE '%s%%' THEN TRUE
      ELSE FALSE
    END
  ", mt_key, mt_prefix))

  n_mt <- DBI::dbGetQuery(con, sprintf(
    "SELECT COUNT(*) AS n FROM var WHERE %s = TRUE", mt_key
  ))$n[1]
  message(sprintf("    Found %s mitochondrial genes", format(n_mt, big.mark = ",")))

  # =================================================
  # Step 2: Cell-wise QC (total_counts, n_genes_by_counts)
  # =================================================
  message("  Step 2: Computing cell-wise QC metrics...")

  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _cell_basic AS
    SELECT
      cell_index AS id,
      SUM(data) AS total_counts,
      COUNT(*) AS n_genes_by_counts
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY cell_index
  ")

  # Add columns to obs
  DBI::dbExecute(con, "
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS total_counts REAL
  ")
  DBI::dbExecute(con, "
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
  ")

  # Update obs
  DBI::dbExecute(con, "
    UPDATE obs
    SET
      total_counts = c.total_counts,
      n_genes_by_counts = c.n_genes_by_counts
    FROM _cell_basic c
    WHERE obs.id = c.id
  ")

  n_cells <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs")$n[1]
  message(sprintf("    Computed for %s cells", format(n_cells, big.mark = ",")))

  # =================================================
  # Step 3: Mitochondrial QC (total_counts_mt, pct_counts_mt)
  # =================================================
  message("  Step 3: Computing mitochondrial QC metrics...")

  if (n_mt > 0) {
    DBI::dbExecute(con, sprintf("
      CREATE OR REPLACE TEMP TABLE _cell_mt AS
      SELECT
        x.cell_index AS id,
        SUM(x.data) AS total_counts_mt
      FROM X_CSR_data x
      JOIN var v ON x.indices = v.id
      WHERE v.%s = TRUE
      GROUP BY x.cell_index
    ", mt_key))

    DBI::dbExecute(con, "
      ALTER TABLE obs
      ADD COLUMN IF NOT EXISTS total_counts_mt REAL
    ")
    DBI::dbExecute(con, "
      ALTER TABLE obs
      ADD COLUMN IF NOT EXISTS pct_counts_mt REAL
    ")

    DBI::dbExecute(con, "
      UPDATE obs
      SET
        total_counts_mt = COALESCE(m.total_counts_mt, 0),
        pct_counts_mt = CASE
          WHEN obs.total_counts > 0
          THEN 100.0 * COALESCE(m.total_counts_mt, 0) / obs.total_counts
          ELSE 0
        END
      FROM _cell_mt m
      WHERE obs.id = m.id
    ")

    message(sprintf("    Computed mt metrics for %s cells", format(n_cells, big.mark = ",")))
  } else {
    message("    No mitochondrial genes found, skipping mt metrics")
  }

  # =================================================
  # Step 4: Gene-wise QC (total_counts, n_cells_by_counts)
  # =================================================
  message("  Step 4: Computing gene-wise QC metrics...")

  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _gene_qc AS
    SELECT
      indices AS id,
      SUM(data) AS total_counts,
      COUNT(DISTINCT cell_index) AS n_cells_by_counts
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY indices
  ")

  DBI::dbExecute(con, "
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS total_counts REAL
  ")
  DBI::dbExecute(con, "
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
  ")

  DBI::dbExecute(con, "
    UPDATE var
    SET
      total_counts = g.total_counts,
      n_cells_by_counts = g.n_cells_by_counts
    FROM _gene_qc g
    WHERE var.id = g.id
  ")

  n_genes <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM var")$n[1]
  message(sprintf("    Computed for %s genes", format(n_genes, big.mark = ",")))

  # =================================================
  # Clean up
  # =================================================
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_basic")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_mt")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _gene_qc")

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[calculate_qc_metrics] Complete in %.2f seconds", elapsed))

  invisible(a)
}


# -----------------------------------------------------------------------------
# Calculate Cell Total Counts
# -----------------------------------------------------------------------------
#' Calculate Total UMI Counts Per Cell
#'
#' Computes total expression counts for each cell and stores in obs table.
#' Uses minibatch processing for large datasets.
#'
#' @param a An atlas object.
#' @param col_name Character. Column name for total counts in obs.
#' @param batch_size Integer. Batch size for processing.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- calculate_cell_total_counts(atlas, col_name = "total_counts")
calculate_cell_total_counts <- function(a, ...) {
  UseMethod("calculate_cell_total_counts")
}

#' @export
calculate_cell_total_counts.atlas <- function(a,
                                               col_name = "total_counts",
                                               batch_size = 2048,
                                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[calculate_cell_total_counts] Computing total counts per cell...")

  start_time <- Sys.time()

  # Add column if not exists
  col_check <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'obs' AND column_name = '%s'
  ", col_name))$n[1]

  if (col_check == 0) {
    DBI::dbExecute(con, sprintf("
      ALTER TABLE obs ADD COLUMN %s REAL
    ", col_name))
  }

  # Compute total counts using SQL aggregation
  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _cell_total AS
    SELECT
      cell_index AS id,
      SUM(data) AS total
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY cell_index
  ")

  DBI::dbExecute(con, sprintf("
    UPDATE obs
    SET %s = COALESCE(t.total, 0)
    FROM _cell_total t
    WHERE obs.id = t.id
  ", col_name))

  n_cells <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs WHERE NOT isnan(total_counts)")$n[1]

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_total")

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[calculate_cell_total_counts] Computed for %s cells in %.2f seconds",
         format(n_cells, big.mark = ","), elapsed))

  invisible(a)
}


# -----------------------------------------------------------------------------
# Calculate Gene Total Counts
# -----------------------------------------------------------------------------
#' Calculate Total Expression Per Gene
#'
#' Computes total expression and mean expression for each gene.
#' Stores results in var table.
#'
#' @param a An atlas object.
#' @param col_total Character. Column name for total counts in var.
#' @param col_mean Character. Column name for mean counts in var.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- calculate_gene_total_counts(atlas)
calculate_gene_total_counts <- function(a, ...) {
  UseMethod("calculate_gene_total_counts")
}

#' @export
calculate_gene_total_counts.atlas <- function(a,
                                               col_total = "total_counts",
                                               col_mean = "mean_counts",
                                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[calculate_gene_total_counts] Computing total and mean per gene...")

  start_time <- Sys.time()

  # Add columns if not exist
  col_check <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'var' AND column_name = '%s'
  ", col_total))$n[1]

  if (col_check == 0) {
    DBI::dbExecute(con, sprintf("
      ALTER TABLE var ADD COLUMN %s REAL DEFAULT 0.0
    ", col_total))
  }

  col_check <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'var' AND column_name = '%s'
  ", col_mean))$n[1]

  if (col_check == 0) {
    DBI::dbExecute(con, sprintf("
      ALTER TABLE var ADD COLUMN %s REAL DEFAULT 0.0
    ", col_mean))
  }

  # Compute gene statistics using SQL
  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _gene_stats AS
    SELECT
      indices AS id,
      SUM(data) AS total,
      AVG(data) AS mean_val
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY indices
  ")

  DBI::dbExecute(con, sprintf("
    UPDATE var
    SET
      %s = COALESCE(s.total, 0),
      %s = COALESCE(s.mean_val, 0)
    FROM _gene_stats s
    WHERE var.id = s.id
  ", col_total, col_mean))

  # Get total cells for stats
  total_cells <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs")$n[1]

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _gene_stats")

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[calculate_gene_total_counts] Computed for all genes in %.2f seconds", elapsed))

  invisible(a)
}


# -----------------------------------------------------------------------------
# Normalize and Log1p Combined
# -----------------------------------------------------------------------------
#' Normalize Total and Apply Log Transform
#'
#' Combines normalize_total and log1p operations:
#' 1. Computes scale factors (stores in obs)
#' 2. Applies log(1 + data * scale_factor)
#'
#' This is equivalent to Scanpy's pp.normalize_total + pp.log1p pipeline.
#'
#' @param a An atlas object.
#' @param target_sum Numeric. Target sum for normalization (default: 10000).
#' @param scale_key Character. Column name for scale factors in obs.
#' @param col_name Character. Output column name in X_CSR_data.
#' @param field Character. Input field to use.
#' @param base Numeric. Logarithm base. NULL = natural log.
#' @param chunk_size Integer. Chunk size for log transform.
#'
#' @return The atlas object.
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data", mode = "r+")
#' atlas <- normalize_and_log1p(atlas, target_sum = 10000)
normalize_and_log1p <- function(a, ...) {
  UseMethod("normalize_and_log1p")
}

#' @export
normalize_and_log1p.atlas <- function(a,
                                       target_sum = 10000,
                                       scale_key = "scale_factor",
                                       col_name = "log1p_factor",
                                       field = "data",
                                       base = NULL,
                                       chunk_size = 100000000,
                                       ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[normalize_and_log1p] Normalizing and log-transforming...")

  start_time <- Sys.time()

  # Step 1: Compute scale factors (normalize_total)
  message("  Step 1: Computing scale factors...")
  normalize_total.atlas(a, target_sum = target_sum, col_name = scale_key, field = field)

  # Step 2: Apply log(1 + data * scale_factor)
  message("  Step 2: Applying log1p with scale factors...")

  # Determine log expression
  if (is.null(base)) {
    log_expr <- sprintf("ln(1.0 + x.%s * o.%s)", field, scale_key)
  } else {
    log_expr <- sprintf("log(%f, 1.0 + x.%s * o.%s)", as.numeric(base), field, scale_key)
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get id range
  id_range <- DBI::dbGetQuery(con, "SELECT MIN(id) AS min, MAX(id) AS max FROM X_CSR_data")
  min_id <- id_range$min[1]
  max_id <- id_range$max[1]

  if (is.null(min_id) || is.na(min_id)) {
    message("  X_CSR_data is empty, skipping")
    return(invisible(a))
  }

  n_total <- max_id - min_id + 1
  n_chunks <- ceiling(n_total / chunk_size)
  message(sprintf("  Processing %s records in %s chunks", format(n_total, big.mark = ","), n_chunks))

  # Update in chunks with JOIN to obs for scale factors
  for (i in seq_len(n_chunks)) {
    start_id <- min_id + (i - 1) * chunk_size
    end_id <- min(start_id + chunk_size - 1, max_id)

    if (i %% 10 == 1 || i == n_chunks) {
      message(sprintf("  Chunk %s/%s: [%s, %s]",
             i, n_chunks, format(start_id, big.mark = ","), format(end_id, big.mark = ",")))
    }

    DBI::dbExecute(con, sprintf("
      UPDATE X_CSR_data AS x
      SET %s = %s
      FROM obs AS o
      WHERE x.cell_index = o.id
        AND x.id BETWEEN %s AND %s
        AND x.%s IS NOT NULL
    ", col_name, log_expr, as.character(start_id), as.character(end_id), field))
  }

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[normalize_and_log1p] Complete in %.2f seconds", elapsed))

  invisible(a)
}


# -----------------------------------------------------------------------------
# Internal: Create Atlas View (for immutable operations)
# -----------------------------------------------------------------------------
# Creates a new atlas object that shares the connection but has different filters.
# Uses the same structure as atlas.R: obs_cell_id, var_gene_id, obs_mask, var_mask
.new_atlas_view <- function(a, obs_cell_id = NULL, var_gene_id = NULL) {
  # Start with original filters
  if (is.null(obs_cell_id)) {
    obs_cell_id <- a$obs_cell_id
  } else if (!is.null(a$obs_cell_id)) {
    obs_cell_id <- intersect(a$obs_cell_id, obs_cell_id)
  }

  if (is.null(var_gene_id)) {
    var_gene_id <- a$var_gene_id
  } else if (!is.null(a$var_gene_id)) {
    var_gene_id <- intersect(a$var_gene_id, var_gene_id)
  }

  new_obj <- list(
    name = a$name,
    path = a$path,
    mode = a$mode,
    con = a$con,
    obs_cell_id = obs_cell_id,
    var_gene_id = var_gene_id,
    obs_mask = NULL,  # Views don't use mask, only IDs
    var_mask = NULL
  )

  class(new_obj) <- "atlas"
  new_obj
}
