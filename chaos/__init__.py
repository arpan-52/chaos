"""
CHAOS - Chain-based Algebraic Optimal Solver

A calibration framework for radio interferometry.

Algorithm:
    1. Initial guess via algebraic chaining from reference antenna
    2. Least-squares polish over all baselines (always runs)

Features:
    - All standard Jones terms: K, B, G, G[p], D, P, X, KCROSS
    - Linear (X,Y) and circular (R,L) feed support
    - Parallactic angle correction
    - Config-based calibration pipelines
    - Memory-efficient chunked I/O
    - Optional GPU acceleration

Author: Arpan Pal
"""

__version__ = "2.0.0"
__author__ = "Arpan Pal"

from chaos.core.solver import solve_jones
from chaos.pipeline.runner import run_pipeline
from chaos.io.applycal import applycal

__all__ = [
    "solve_jones",
    "run_pipeline",
    "applycal",
    "__version__",
]
