"""
Jones Matrix Definitions for Radio Interferometry.

Measurement Equation (from signal path order):
    V_ij = M_ij B_ij G_ij D_ij E_ij P_ij T_ij V_ij^IDEAL

Where:
    T = Tropospheric/atmospheric effects (opacity, path-length)
    P = Parallactic angle rotation
    E = Primary beam / elevation effects
    D = Instrumental polarization (leakage/D-terms)
    G = Electronic gain (complex, time-variable)
    B = Bandpass (frequency-dependent)
    M = Baseline-based correlator errors (non-factorable)

For antenna-based calibration, we have:
    V_ij = J_i M_ij J_j^H

Where J_i is the composite Jones matrix for antenna i.

Both linear (X,Y) and circular (R,L) feeds use the same chain structure,
but the matrix forms differ.
"""

import numpy as np


# =============================================================================
# Parallactic Angle (P-Jones)
# =============================================================================

def P_jones_linear(psi):
    """
    Parallactic angle rotation matrix for LINEAR feeds (X, Y).
    
    Rotates the polarization frame as the antenna tracks a source.
    
    P = | cos(psi)  -sin(psi) |
        | sin(psi)   cos(psi) |
    
    Parameters
    ----------
    psi : float or array
        Parallactic angle in radians
    
    Returns
    -------
    P : ndarray (..., 2, 2)
        P-Jones matrix
    """
    psi = np.asarray(psi)
    c = np.cos(psi)
    s = np.sin(psi)
    
    if psi.ndim == 0:
        return np.array([[c, -s], [s, c]], dtype=complex)
    else:
        P = np.zeros(psi.shape + (2, 2), dtype=complex)
        P[..., 0, 0] = c
        P[..., 0, 1] = -s
        P[..., 1, 0] = s
        P[..., 1, 1] = c
        return P


def P_jones_circular(psi):
    """
    Parallactic angle rotation matrix for CIRCULAR feeds (R, L).
    
    For circular feeds, parallactic angle appears as a diagonal phase.
    
    P = | exp(-i*psi)    0        |
        |     0      exp(+i*psi)  |
    
    Parameters
    ----------
    psi : float or array
        Parallactic angle in radians
    
    Returns
    -------
    P : ndarray (..., 2, 2)
        P-Jones matrix
    """
    psi = np.asarray(psi)
    
    if psi.ndim == 0:
        return np.array([[np.exp(-1j * psi), 0], 
                         [0, np.exp(1j * psi)]], dtype=complex)
    else:
        P = np.zeros(psi.shape + (2, 2), dtype=complex)
        P[..., 0, 0] = np.exp(-1j * psi)
        P[..., 1, 1] = np.exp(1j * psi)
        return P


# =============================================================================
# Gain (G-Jones)
# =============================================================================

def G_jones(g_x, g_y):
    """
    Electronic gain matrix (diagonal).
    
    G = | g_x   0  |
        |  0   g_y |
    
    Parameters
    ----------
    g_x, g_y : complex or array
        Complex gains for X/R and Y/L polarizations
    
    Returns
    -------
    G : ndarray (..., 2, 2)
        G-Jones matrix
    """
    g_x = np.asarray(g_x)
    g_y = np.asarray(g_y)
    
    if g_x.ndim == 0:
        return np.array([[g_x, 0], [0, g_y]], dtype=complex)
    else:
        shape = np.broadcast_shapes(g_x.shape, g_y.shape)
        G = np.zeros(shape + (2, 2), dtype=complex)
        G[..., 0, 0] = g_x
        G[..., 1, 1] = g_y
        return G


def G_jones_full(g_xx, g_xy, g_yx, g_yy):
    """
    Full 2x2 electronic gain matrix.
    
    G = | g_xx  g_xy |
        | g_yx  g_yy |
    
    Parameters
    ----------
    g_xx, g_xy, g_yx, g_yy : complex or array
        Full Jones matrix elements
    
    Returns
    -------
    G : ndarray (..., 2, 2)
        G-Jones matrix
    """
    g_xx = np.asarray(g_xx)
    
    if g_xx.ndim == 0:
        return np.array([[g_xx, g_xy], [g_yx, g_yy]], dtype=complex)
    else:
        G = np.zeros(g_xx.shape + (2, 2), dtype=complex)
        G[..., 0, 0] = g_xx
        G[..., 0, 1] = g_xy
        G[..., 1, 0] = g_yx
        G[..., 1, 1] = g_yy
        return G


# =============================================================================
# Bandpass (B-Jones)
# =============================================================================

def B_jones(b_x, b_y):
    """
    Bandpass matrix (diagonal, frequency-dependent).
    
    B = | b_x   0  |
        |  0   b_y |
    
    Parameters
    ----------
    b_x, b_y : complex or array
        Complex bandpass for each polarization
    
    Returns
    -------
    B : ndarray (..., 2, 2)
        B-Jones matrix
    """
    return G_jones(b_x, b_y)  # Same form as G


# =============================================================================
# Delay (K-Jones)
# =============================================================================

def K_jones(tau_x, tau_y, freq):
    """
    Delay matrix (converts delay to phase vs frequency).
    
    K = | exp(2*pi*i*freq*tau_x)         0              |
        |         0              exp(2*pi*i*freq*tau_y) |
    
    Parameters
    ----------
    tau_x, tau_y : float or array
        Delays in seconds
    freq : float or array
        Frequency in Hz
    
    Returns
    -------
    K : ndarray (..., 2, 2)
        K-Jones matrix
    """
    tau_x = np.asarray(tau_x)
    tau_y = np.asarray(tau_y)
    freq = np.asarray(freq)
    
    phase_x = 2 * np.pi * freq * tau_x
    phase_y = 2 * np.pi * freq * tau_y
    
    return G_jones(np.exp(1j * phase_x), np.exp(1j * phase_y))


# =============================================================================
# Leakage / D-terms (D-Jones)
# =============================================================================

def D_jones_linear(d_x, d_y):
    """
    Leakage matrix for LINEAR feeds.
    
    D = |  1   d_x |
        | d_y   1  |
    
    d_x = leakage from Y into X
    d_y = leakage from X into Y
    
    Parameters
    ----------
    d_x, d_y : complex or array
        Leakage terms
    
    Returns
    -------
    D : ndarray (..., 2, 2)
        D-Jones matrix
    """
    d_x = np.asarray(d_x)
    d_y = np.asarray(d_y)
    
    if d_x.ndim == 0:
        return np.array([[1, d_x], [d_y, 1]], dtype=complex)
    else:
        shape = np.broadcast_shapes(d_x.shape, d_y.shape)
        D = np.zeros(shape + (2, 2), dtype=complex)
        D[..., 0, 0] = 1
        D[..., 0, 1] = d_x
        D[..., 1, 0] = d_y
        D[..., 1, 1] = 1
        return D


def D_jones_circular(d_r, d_l):
    """
    Leakage matrix for CIRCULAR feeds.
    
    D = |  1   d_r |
        | d_l   1  |
    
    d_r = leakage from L into R
    d_l = leakage from R into L
    
    Parameters
    ----------
    d_r, d_l : complex or array
        Leakage terms
    
    Returns
    -------
    D : ndarray (..., 2, 2)
        D-Jones matrix
    """
    return D_jones_linear(d_r, d_l)  # Same form


# =============================================================================
# Cross-hand phase/delay (X-Jones)
# =============================================================================

def X_jones(phi):
    """
    Cross-hand phase matrix.
    
    X = | 1         0      |
        | 0   exp(i*phi)   |
    
    Relative phase between polarizations.
    
    Parameters
    ----------
    phi : float or array
        Cross-hand phase in radians
    
    Returns
    -------
    X : ndarray (..., 2, 2)
        X-Jones matrix
    """
    phi = np.asarray(phi)
    return G_jones(1.0, np.exp(1j * phi))


# =============================================================================
# Troposphere (T-Jones) - scalar, same for both pols
# =============================================================================

def T_jones(amplitude, phase):
    """
    Tropospheric/atmospheric effects (scalar matrix).
    
    T = | a*exp(i*phi)       0        |
        |      0        a*exp(i*phi)  |
    
    Parameters
    ----------
    amplitude : float or array
        Amplitude factor (opacity)
    phase : float or array
        Phase delay in radians
    
    Returns
    -------
    T : ndarray (..., 2, 2)
        T-Jones matrix
    """
    t = amplitude * np.exp(1j * phase)
    return G_jones(t, t)


# =============================================================================
# Identity
# =============================================================================

def I_jones(shape=None):
    """
    Identity Jones matrix.
    
    Parameters
    ----------
    shape : tuple, optional
        Leading dimensions
    
    Returns
    -------
    I : ndarray (..., 2, 2)
        Identity matrix
    """
    if shape is None:
        return np.eye(2, dtype=complex)
    else:
        I = np.zeros(shape + (2, 2), dtype=complex)
        I[..., 0, 0] = 1
        I[..., 1, 1] = 1
        return I


# =============================================================================
# Composite Jones
# =============================================================================

def composite_jones(jones_list):
    """
    Compute composite Jones matrix from a list of Jones terms.
    
    J_composite = J_N @ J_{N-1} @ ... @ J_2 @ J_1
    
    Order: signal propagation order (first term is closest to sky)
    
    Parameters
    ----------
    jones_list : list of ndarray
        List of Jones matrices in signal order
    
    Returns
    -------
    J : ndarray (..., 2, 2)
        Composite Jones matrix
    """
    if len(jones_list) == 0:
        return I_jones()
    
    J = jones_list[-1].copy()
    for i in range(len(jones_list) - 2, -1, -1):
        J = J @ jones_list[i]
    
    return J


# =============================================================================
# Apply/Unapply Jones to visibilities
# =============================================================================

def apply_jones(V, J_i, J_j):
    """
    Apply Jones matrices to model visibility.
    
    V_obs = J_i @ V_model @ J_j^H
    
    Parameters
    ----------
    V : ndarray (..., 2, 2)
        Model visibility matrix
    J_i, J_j : ndarray (..., 2, 2)
        Jones matrices for antennas i and j
    
    Returns
    -------
    V_obs : ndarray (..., 2, 2)
        Corrupted visibility
    """
    J_j_H = np.conj(np.swapaxes(J_j, -2, -1))
    return J_i @ V @ J_j_H


def unapply_jones(V_obs, J_i, J_j):
    """
    Remove Jones matrices from observed visibility.
    
    V_corrected = J_i^{-1} @ V_obs @ (J_j^{-1})^H
    
    Parameters
    ----------
    V_obs : ndarray (..., 2, 2)
        Observed visibility matrix
    J_i, J_j : ndarray (..., 2, 2)
        Jones matrices for antennas i and j
    
    Returns
    -------
    V_corrected : ndarray (..., 2, 2)
        Corrected visibility
    """
    J_i_inv = np.linalg.inv(J_i)
    J_j_inv = np.linalg.inv(J_j)
    J_j_inv_H = np.conj(np.swapaxes(J_j_inv, -2, -1))
    return J_i_inv @ V_obs @ J_j_inv_H


# =============================================================================
# Feed type detection
# =============================================================================

# CASA correlation type codes
CORR_TYPE_LINEAR = {9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY'}
CORR_TYPE_CIRCULAR = {5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL'}


def detect_feed_type(corr_types):
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
    """
    corr_types = set(corr_types)
    
    if corr_types & {9, 10, 11, 12}:
        return 'linear'
    elif corr_types & {5, 6, 7, 8}:
        return 'circular'
    else:
        raise ValueError(f"Unknown correlation types: {corr_types}")


__all__ = [
    'P_jones_linear', 'P_jones_circular',
    'G_jones', 'G_jones_full',
    'B_jones',
    'K_jones',
    'D_jones_linear', 'D_jones_circular',
    'X_jones',
    'T_jones',
    'I_jones',
    'composite_jones',
    'apply_jones', 'unapply_jones',
    'detect_feed_type',
    'CORR_TYPE_LINEAR', 'CORR_TYPE_CIRCULAR'
]
