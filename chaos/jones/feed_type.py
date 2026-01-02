"""
Feed Type Detection.

Detect whether an MS uses linear (X,Y) or circular (R,L) feeds
from the POLARIZATION table.
"""

import numpy as np
from typing import Union, List

# Feed type constants
FEED_LINEAR = "linear"
FEED_CIRCULAR = "circular"

# CASA correlation type codes
# From casacore Stokes.h
CORR_CODES = {
    # Undefined
    0: "Undefined",
    # Stokes
    1: "I", 2: "Q", 3: "U", 4: "V",
    # Circular
    5: "RR", 6: "RL", 7: "LR", 8: "LL",
    # Linear
    9: "XX", 10: "XY", 11: "YX", 12: "YY",
    # Mixed
    13: "RX", 14: "RY", 15: "LX", 16: "LY",
    17: "XR", 18: "XL", 19: "YR", 20: "YL",
    # Single dish
    21: "PP", 22: "PQ", 23: "QP", 24: "QQ",
    25: "RCircular", 26: "LCircular",
    27: "Linear", 28: "Ptotal", 29: "Plinear", 30: "PFtotal", 31: "PFlinear",
    32: "Pangle",
}

# Linear correlation types
LINEAR_CORRS = {9, 10, 11, 12}  # XX, XY, YX, YY

# Circular correlation types
CIRCULAR_CORRS = {5, 6, 7, 8}  # RR, RL, LR, LL


def detect_feed_type(corr_types: Union[np.ndarray, List[int]]) -> str:
    """
    Detect feed type from correlation types.
    
    Parameters
    ----------
    corr_types : array-like
        CORR_TYPE values from POLARIZATION table
    
    Returns
    -------
    feed_type : str
        'linear' or 'circular'
    
    Raises
    ------
    ValueError
        If correlation types are unknown or mixed
    """
    corr_set = set(corr_types)
    
    has_linear = bool(corr_set & LINEAR_CORRS)
    has_circular = bool(corr_set & CIRCULAR_CORRS)
    
    if has_linear and has_circular:
        raise ValueError(
            f"Mixed linear and circular correlations: {corr_set}. "
            "This is not supported."
        )
    
    if has_linear:
        return FEED_LINEAR
    elif has_circular:
        return FEED_CIRCULAR
    else:
        raise ValueError(f"Unknown correlation types: {corr_set}")


def get_feed_type_from_ms(ms_path: str) -> str:
    """
    Get feed type from MeasurementSet.
    
    Parameters
    ----------
    ms_path : str
        Path to MeasurementSet
    
    Returns
    -------
    feed_type : str
        'linear' or 'circular'
    """
    from casacore.tables import table
    
    with table(f"{ms_path}/POLARIZATION", ack=False) as tb:
        corr_types = tb.getcol("CORR_TYPE")
        # corr_types shape: (n_pol_setup, n_corr)
        # Use first row
        return detect_feed_type(corr_types[0])


def get_corr_labels(feed_type: str) -> tuple:
    """
    Get correlation labels for a feed type.
    
    Parameters
    ----------
    feed_type : str
        'linear' or 'circular'
    
    Returns
    -------
    labels : tuple of str
        ('XX', 'XY', 'YX', 'YY') or ('RR', 'RL', 'LR', 'LL')
    """
    if feed_type == FEED_LINEAR:
        return ("XX", "XY", "YX", "YY")
    elif feed_type == FEED_CIRCULAR:
        return ("RR", "RL", "LR", "LL")
    else:
        raise ValueError(f"Unknown feed type: {feed_type}")


def get_pol_labels(feed_type: str) -> tuple:
    """
    Get polarization labels for a feed type.
    
    Parameters
    ----------
    feed_type : str
        'linear' or 'circular'
    
    Returns
    -------
    labels : tuple of str
        ('X', 'Y') or ('R', 'L')
    """
    if feed_type == FEED_LINEAR:
        return ("X", "Y")
    elif feed_type == FEED_CIRCULAR:
        return ("R", "L")
    else:
        raise ValueError(f"Unknown feed type: {feed_type}")
