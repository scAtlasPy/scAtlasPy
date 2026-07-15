def default_point_size(n_points: int) -> float:
    """Estimate scatter point size from the number of plotted cells."""
    return 120_000 / max(int(n_points), 1)
