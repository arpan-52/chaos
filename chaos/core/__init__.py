"""
Core Solver Components.

The CHAOS algorithm:
1. Single-chain algebraic solution for initial guess
2. Least-squares polish over all baselines (always runs)
"""

from chaos.core.solver import solve_jones
from chaos.core.chain_solver import ChainSolver
from chaos.core.polish import polish_jones

__all__ = [
    "solve_jones",
    "ChainSolver", 
    "polish_jones",
]
