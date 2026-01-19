# zzz.R - 包初始化
# =============================================================================

.onLoad <- function(libname, pkgname) {
  options(
    scatlas.threads = parallel::detectCores(),
    scatlas.memory_limit = "250GB"
  )
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("scAtlas R版本 - 单细胞图谱分析工具")
}

# Utility: null coalescing operator
`%||%` <- function(a, b) if (is.null(a)) b else a
