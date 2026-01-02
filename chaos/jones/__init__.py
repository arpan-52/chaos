"""
Jones Matrix Definitions and Operations.

This module provides all Jones matrix types used in radio interferometry
calibration, with proper implementations for both linear and circular feeds.
"""

from chaos.jones.terms import (
    # Delay
    K_jones,
    K_jones_from_delay,
    KCROSS_jones,
    # Bandpass and Gain
    B_jones,
    G_jones,
    G_jones_phase_only,
    # Parallactic angle
    P_jones_linear,
    P_jones_circular,
    # Leakage
    D_jones,
    # Cross-hand phase
    X_jones,
    # Utility
    I_jones,
)

from chaos.jones.operations import (
    apply_jones,
    unapply_jones,
    composite_jones,
    jones_to_mueller,
)

from chaos.jones.feed_type import (
    detect_feed_type,
    FEED_LINEAR,
    FEED_CIRCULAR,
)

__all__ = [
    # Terms
    "K_jones",
    "K_jones_from_delay",
    "KCROSS_jones",
    "B_jones",
    "G_jones",
    "G_jones_phase_only",
    "P_jones_linear",
    "P_jones_circular",
    "D_jones",
    "X_jones",
    "I_jones",
    # Operations
    "apply_jones",
    "unapply_jones",
    "composite_jones",
    "jones_to_mueller",
    # Feed type
    "detect_feed_type",
    "FEED_LINEAR",
    "FEED_CIRCULAR",
]
