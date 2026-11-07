"""KAPE: Kernel Adaptive Plasma Estimator."""

from .bandwidth import compute_bandwidth_from_array, opt_alph
from .core import KAPE, get_domain_bounds, in_spherical_domain
from .moments import EV_TO_K, compute_moments
from .plotting import (gyro_cont,gyro_contour,gyro_fhat_grid_spherical_domain,plot_gyro_scatter_original)

__version__ = "0.0.1"

__all__ = [
    # Original KAPE algoritihm for actual "reconstruction" f(v) surface.
    "KAPE",
    "compute_bandwidth_from_array",
    "opt_alph",
    "get_domain_bounds",
    "in_spherical_domain",

    # moment calculation functions
    "compute_moments",
    "EV_TO_K",

    # gyrotropic plotting functions, 3 options: original scatter-plot, contour plots, or continuous space plot.
    "gyro_cont",
    "gyro_contour",
    "gyro_fhat_grid_spherical_domain",
    "plot_gyro_scatter_original",
]