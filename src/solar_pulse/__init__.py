"""SOLAR_PULSE: Spherical Grid Adaptive Bandwidth Nadaraya-Watson Estimator."""

from .bandwidth import compute_bandwidth_from_array, opt_alph
from .core import NAE, get_domain_bounds, in_spherical_domain
from .moments import EV_TO_K, compute_moments

__version__ = "0.1.0"
__all__ = [
    "NAE",
    "compute_bandwidth_from_array",
    "opt_alph",
    "get_domain_bounds",
    "in_spherical_domain",
    "compute_moments",
    "EV_TO_K",
]