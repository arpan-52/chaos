"""
CHAOS - Chain-based Algebraic Optimal Solver

Multi-reference weighted Jones calibration for radio interferometry.

Features:
- Algebraic chaining from every antenna as reference
- Phase alignment and weighted averaging
- Least squares polish for CASA-level accuracy
- Full calibration framework with config-based pipeline
- Support for linear (X,Y) and circular (R,L) feeds
- Parallactic angle correction
- Memory-efficient chunked I/O

Jones Terms Supported:
- K: Delay
- B: Bandpass  
- G: Gain
- P: Parallactic angle
- D: Leakage (D-terms)
- X: Cross-hand phase

Usage:
    # Simple MS calibration
    chaos mydata.ms --polish
    
    # Config-based pipeline
    chaos run calibration.yaml --ref-ant 0 --polish
    
    # Apply solutions
    chaos applycal mydata.ms --jones chaos_cal.npy

Author: Arpan Pal
"""

__version__ = "0.2.0"
__author__ = "Arpan Pal"
__algorithm__ = "CHAOS - Chain-based Algebraic Optimal Solver"

# Core calibration
from .calibrate import calibrate_ms
from .applycal import applycal

# Framework components
from .runner import CalibrationRunner, run_from_config
from .config_parser import load_config, CalConfig

# Jones matrices
from .jones_terms import (
    P_jones_linear, P_jones_circular,
    G_jones, B_jones, K_jones,
    D_jones_linear, D_jones_circular,
    X_jones, T_jones, I_jones,
    composite_jones, apply_jones, unapply_jones,
    detect_feed_type
)

# Table I/O
from .table_io import (
    save_jones_term, load_jones_term,
    list_jones_terms, get_jones_info
)

# Interpolation
from .interpolator import (
    interpolate_jones_diagonal,
    interpolate_jones_full,
    interpolate_multi_field
)

# Parallactic angle
from .parallactic import (
    compute_parallactic_angle,
    compute_parallactic_angles_from_ms,
    get_mount_type
)

__all__ = [
    # Core
    'calibrate_ms',
    'applycal',
    
    # Framework
    'CalibrationRunner',
    'run_from_config',
    'load_config',
    'CalConfig',
    
    # Jones matrices
    'P_jones_linear', 'P_jones_circular',
    'G_jones', 'B_jones', 'K_jones',
    'D_jones_linear', 'D_jones_circular',
    'X_jones', 'T_jones', 'I_jones',
    'composite_jones', 'apply_jones', 'unapply_jones',
    'detect_feed_type',
    
    # Table I/O
    'save_jones_term', 'load_jones_term',
    'list_jones_terms', 'get_jones_info',
    
    # Interpolation
    'interpolate_jones_diagonal',
    'interpolate_jones_full',
    'interpolate_multi_field',
    
    # Parallactic
    'compute_parallactic_angle',
    'compute_parallactic_angles_from_ms',
    'get_mount_type',
]
