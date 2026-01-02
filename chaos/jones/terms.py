"""
Jones Matrix Definitions.

All Jones matrices for radio interferometry calibration.
Supports both linear (X,Y) and circular (R,L) feeds.

Convention:
    - Jones matrices are 2x2 complex
    - For linear feeds: index 0 = X, index 1 = Y
    - For circular feeds: index 0 = R, index 1 = L
    - All functions support broadcasting
"""

import numpy as np
from typing import Union, Optional

# Type alias for array-like inputs
ArrayLike = Union[float, np.ndarray]


def I_jones(shape: Optional[tuple] = None) -> np.ndarray:
    """
    Identity Jones matrix.
    
    I = | 1  0 |
        | 0  1 |
    
    Parameters
    ----------
    shape : tuple, optional
        Leading dimensions for broadcasting, e.g., (n_ant,) or (n_time, n_ant)
    
    Returns
    -------
    I : ndarray, shape (*shape, 2, 2) or (2, 2)
    """
    if shape is None:
        return np.eye(2, dtype=np.complex128)
    
    I = np.zeros(shape + (2, 2), dtype=np.complex128)
    I[..., 0, 0] = 1.0
    I[..., 1, 1] = 1.0
    return I


# =============================================================================
# K-Jones: Delay
# =============================================================================

def K_jones(phase_0: ArrayLike, phase_1: ArrayLike) -> np.ndarray:
    """
    Delay Jones matrix from phases.
    
    K = | exp(i·φ_0)      0       |
        |     0       exp(i·φ_1)  |
    
    For linear feeds: φ_0 = φ_X, φ_1 = φ_Y
    For circular feeds: φ_0 = φ_R, φ_1 = φ_L
    
    Parameters
    ----------
    phase_0 : float or ndarray
        Phase for first polarization (radians)
    phase_1 : float or ndarray  
        Phase for second polarization (radians)
    
    Returns
    -------
    K : ndarray, shape (..., 2, 2)
    """
    phase_0 = np.asarray(phase_0, dtype=np.float64)
    phase_1 = np.asarray(phase_1, dtype=np.float64)
    
    shape = np.broadcast_shapes(phase_0.shape, phase_1.shape)
    
    if len(shape) == 0:
        K = np.zeros((2, 2), dtype=np.complex128)
    else:
        K = np.zeros(shape + (2, 2), dtype=np.complex128)
    
    K[..., 0, 0] = np.exp(1j * phase_0)
    K[..., 1, 1] = np.exp(1j * phase_1)
    
    return K


def K_jones_from_delay(
    tau_0: ArrayLike, 
    tau_1: ArrayLike, 
    freq: ArrayLike
) -> np.ndarray:
    """
    Delay Jones matrix from delays and frequency.
    
    K = | exp(2πi·ν·τ_0)        0          |
        |       0         exp(2πi·ν·τ_1)   |
    
    Parameters
    ----------
    tau_0 : float or ndarray
        Delay for first polarization (seconds)
    tau_1 : float or ndarray
        Delay for second polarization (seconds)
    freq : float or ndarray
        Frequency (Hz)
    
    Returns
    -------
    K : ndarray, shape (..., 2, 2)
    """
    tau_0 = np.asarray(tau_0, dtype=np.float64)
    tau_1 = np.asarray(tau_1, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    
    phase_0 = 2 * np.pi * freq * tau_0
    phase_1 = 2 * np.pi * freq * tau_1
    
    return K_jones(phase_0, phase_1)


def KCROSS_jones(delta_tau: ArrayLike, freq: ArrayLike) -> np.ndarray:
    """
    Cross-hand delay Jones matrix (GLOBAL - same for all antennas).
    
    K_cross = | 1                  0              |
              | 0          exp(2πi·ν·Δτ)          |
    
    This represents the residual delay difference between the two
    polarizations on the reference antenna.
    
    Parameters
    ----------
    delta_tau : float or ndarray
        Cross-hand delay difference (seconds)
    freq : float or ndarray
        Frequency (Hz)
    
    Returns
    -------
    K_cross : ndarray, shape (..., 2, 2)
    """
    delta_tau = np.asarray(delta_tau, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    
    phase = 2 * np.pi * freq * delta_tau
    
    shape = np.broadcast_shapes(delta_tau.shape, freq.shape)
    
    if len(shape) == 0:
        K = np.zeros((2, 2), dtype=np.complex128)
    else:
        K = np.zeros(shape + (2, 2), dtype=np.complex128)
    
    K[..., 0, 0] = 1.0
    K[..., 1, 1] = np.exp(1j * phase)
    
    return K


# =============================================================================
# B-Jones: Bandpass
# =============================================================================

def B_jones(b_0: ArrayLike, b_1: ArrayLike) -> np.ndarray:
    """
    Bandpass Jones matrix (diagonal).
    
    B = | b_0    0  |
        |  0    b_1 |
    
    For linear feeds: b_0 = b_X, b_1 = b_Y
    For circular feeds: b_0 = b_R, b_1 = b_L
    
    Parameters
    ----------
    b_0 : complex or ndarray
        Complex bandpass for first polarization
    b_1 : complex or ndarray
        Complex bandpass for second polarization
    
    Returns
    -------
    B : ndarray, shape (..., 2, 2)
    """
    b_0 = np.asarray(b_0, dtype=np.complex128)
    b_1 = np.asarray(b_1, dtype=np.complex128)
    
    shape = np.broadcast_shapes(b_0.shape, b_1.shape)
    
    if len(shape) == 0:
        B = np.zeros((2, 2), dtype=np.complex128)
    else:
        B = np.zeros(shape + (2, 2), dtype=np.complex128)
    
    B[..., 0, 0] = b_0
    B[..., 1, 1] = b_1
    
    return B


# =============================================================================
# G-Jones: Gain
# =============================================================================

def G_jones(g_0: ArrayLike, g_1: ArrayLike) -> np.ndarray:
    """
    Gain Jones matrix (diagonal, complex).
    
    G = | g_0    0  |
        |  0    g_1 |
    
    For linear feeds: g_0 = g_X, g_1 = g_Y
    For circular feeds: g_0 = g_R, g_1 = g_L
    
    Parameters
    ----------
    g_0 : complex or ndarray
        Complex gain for first polarization
    g_1 : complex or ndarray
        Complex gain for second polarization
    
    Returns
    -------
    G : ndarray, shape (..., 2, 2)
    """
    return B_jones(g_0, g_1)  # Same structure as B


def G_jones_phase_only(phase_0: ArrayLike, phase_1: ArrayLike) -> np.ndarray:
    """
    Phase-only gain Jones matrix G[p] (amplitude = 1).
    
    G[p] = | exp(i·φ_0)      0        |
           |     0       exp(i·φ_1)   |
    
    Parameters
    ----------
    phase_0 : float or ndarray
        Phase for first polarization (radians)
    phase_1 : float or ndarray
        Phase for second polarization (radians)
    
    Returns
    -------
    G : ndarray, shape (..., 2, 2)
    """
    return K_jones(phase_0, phase_1)  # Same structure as K


# =============================================================================
# P-Jones: Parallactic Angle
# =============================================================================

def P_jones_linear(psi: ArrayLike) -> np.ndarray:
    """
    Parallactic angle rotation matrix for LINEAR feeds (X, Y).
    
    P = | cos(ψ)   -sin(ψ) |
        | sin(ψ)    cos(ψ) |
    
    This is a real rotation matrix (orthogonal, det = 1).
    
    Parameters
    ----------
    psi : float or ndarray
        Parallactic angle in radians
    
    Returns
    -------
    P : ndarray, shape (..., 2, 2)
    """
    psi = np.asarray(psi, dtype=np.float64)
    
    c = np.cos(psi)
    s = np.sin(psi)
    
    if psi.ndim == 0:
        P = np.zeros((2, 2), dtype=np.complex128)
    else:
        P = np.zeros(psi.shape + (2, 2), dtype=np.complex128)
    
    P[..., 0, 0] = c
    P[..., 0, 1] = -s
    P[..., 1, 0] = s
    P[..., 1, 1] = c
    
    return P


def P_jones_circular(psi: ArrayLike) -> np.ndarray:
    """
    Parallactic angle rotation matrix for CIRCULAR feeds (R, L).
    
    P = | exp(-i·ψ)      0       |
        |     0      exp(+i·ψ)   |
    
    For circular feeds, parallactic angle appears as a diagonal phase.
    This makes P commute with other diagonal Jones terms.
    
    Parameters
    ----------
    psi : float or ndarray
        Parallactic angle in radians
    
    Returns
    -------
    P : ndarray, shape (..., 2, 2)
    """
    psi = np.asarray(psi, dtype=np.float64)
    
    if psi.ndim == 0:
        P = np.zeros((2, 2), dtype=np.complex128)
    else:
        P = np.zeros(psi.shape + (2, 2), dtype=np.complex128)
    
    P[..., 0, 0] = np.exp(-1j * psi)
    P[..., 1, 1] = np.exp(1j * psi)
    
    return P


# =============================================================================
# D-Jones: Leakage (D-terms)
# =============================================================================

def D_jones(d_01: ArrayLike, d_10: ArrayLike) -> np.ndarray:
    """
    Leakage (D-term) Jones matrix.
    
    D = |  1    d_01 |
        | d_10   1   |
    
    For linear feeds:
        d_01 = d_X = leakage from Y into X
        d_10 = d_Y = leakage from X into Y
    
    For circular feeds:
        d_01 = d_R = leakage from L into R
        d_10 = d_L = leakage from R into L
    
    Parameters
    ----------
    d_01 : complex or ndarray
        Leakage from pol 1 into pol 0
    d_10 : complex or ndarray
        Leakage from pol 0 into pol 1
    
    Returns
    -------
    D : ndarray, shape (..., 2, 2)
    """
    d_01 = np.asarray(d_01, dtype=np.complex128)
    d_10 = np.asarray(d_10, dtype=np.complex128)
    
    shape = np.broadcast_shapes(d_01.shape, d_10.shape)
    
    if len(shape) == 0:
        D = np.zeros((2, 2), dtype=np.complex128)
    else:
        D = np.zeros(shape + (2, 2), dtype=np.complex128)
    
    D[..., 0, 0] = 1.0
    D[..., 0, 1] = d_01
    D[..., 1, 0] = d_10
    D[..., 1, 1] = 1.0
    
    return D


# =============================================================================
# X-Jones: Cross-hand Phase
# =============================================================================

def X_jones(phi: ArrayLike) -> np.ndarray:
    """
    Cross-hand phase Jones matrix (GLOBAL - same for all antennas).
    
    X = | 1           0          |
        | 0      exp(i·φ)        |
    
    This represents the residual X-Y (or R-L) phase on the reference antenna.
    Applied identically to all antennas.
    
    Parameters
    ----------
    phi : float or ndarray
        Cross-hand phase (radians)
    
    Returns
    -------
    X : ndarray, shape (..., 2, 2)
    """
    phi = np.asarray(phi, dtype=np.float64)
    
    if phi.ndim == 0:
        X = np.zeros((2, 2), dtype=np.complex128)
    else:
        X = np.zeros(phi.shape + (2, 2), dtype=np.complex128)
    
    X[..., 0, 0] = 1.0
    X[..., 1, 1] = np.exp(1j * phi)
    
    return X


# =============================================================================
# Full Jones (for completeness)
# =============================================================================

def full_jones(
    j_00: ArrayLike, 
    j_01: ArrayLike, 
    j_10: ArrayLike, 
    j_11: ArrayLike
) -> np.ndarray:
    """
    Arbitrary full 2x2 Jones matrix.
    
    J = | j_00  j_01 |
        | j_10  j_11 |
    
    Parameters
    ----------
    j_00, j_01, j_10, j_11 : complex or ndarray
        Matrix elements
    
    Returns
    -------
    J : ndarray, shape (..., 2, 2)
    """
    j_00 = np.asarray(j_00, dtype=np.complex128)
    j_01 = np.asarray(j_01, dtype=np.complex128)
    j_10 = np.asarray(j_10, dtype=np.complex128)
    j_11 = np.asarray(j_11, dtype=np.complex128)
    
    shape = np.broadcast_shapes(j_00.shape, j_01.shape, j_10.shape, j_11.shape)
    
    if len(shape) == 0:
        J = np.zeros((2, 2), dtype=np.complex128)
    else:
        J = np.zeros(shape + (2, 2), dtype=np.complex128)
    
    J[..., 0, 0] = j_00
    J[..., 0, 1] = j_01
    J[..., 1, 0] = j_10
    J[..., 1, 1] = j_11
    
    return J
