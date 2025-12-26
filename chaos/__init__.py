"""
CHAOS - Chain-based Algebraic Optimal Solver

Multi-reference weighted Jones calibration for radio interferometry.

Core Innovation:
- Solve from EVERY antenna as reference
- Align phases to user-specified reference
- Weighted average using geometric mean of chain quality
- Robust against noise through redundancy

Modes:
- phase_only: J_ref = I (fixed), phases only
- diagonal: J_ref = diag(g_X, g_Y), amplitudes optimized
- full: J_ref = 2x2 complex, 8 real params optimized

Author: Arpan Pal
Date: December 2025
"""

__version__ = "0.1.0"
__author__ = "Arpan Pal"
__algorithm__ = "CHAOS - Chain-based Algebraic Optimal Solver"

from .calibrate import calibrate_ms

__all__ = ['calibrate_ms']
