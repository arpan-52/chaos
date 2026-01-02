"""
Jones Matrix Operations.

Apply/unapply Jones matrices to visibilities, build composite matrices, etc.
"""

import numpy as np
from typing import List, Optional


def apply_jones(
    V: np.ndarray, 
    J_i: np.ndarray, 
    J_j: np.ndarray
) -> np.ndarray:
    """
    Apply Jones matrices to model visibility.
    
    V_obs = J_i @ V_model @ J_j†
    
    Parameters
    ----------
    V : ndarray (..., 2, 2)
        Model visibility matrix
    J_i : ndarray (..., 2, 2)
        Jones matrix for antenna i
    J_j : ndarray (..., 2, 2)
        Jones matrix for antenna j
    
    Returns
    -------
    V_obs : ndarray (..., 2, 2)
        Corrupted visibility
    """
    J_j_H = np.conj(np.swapaxes(J_j, -2, -1))
    return J_i @ V @ J_j_H


def unapply_jones(
    V_obs: np.ndarray, 
    J_i: np.ndarray, 
    J_j: np.ndarray
) -> np.ndarray:
    """
    Remove Jones matrices from observed visibility (calibration correction).
    
    V_corrected = J_i^{-1} @ V_obs @ (J_j^{-1})†
    
    Parameters
    ----------
    V_obs : ndarray (..., 2, 2)
        Observed visibility matrix
    J_i : ndarray (..., 2, 2)
        Jones matrix for antenna i
    J_j : ndarray (..., 2, 2)
        Jones matrix for antenna j
    
    Returns
    -------
    V_corrected : ndarray (..., 2, 2)
        Corrected visibility
    """
    J_i_inv = np.linalg.inv(J_i)
    J_j_inv = np.linalg.inv(J_j)
    J_j_inv_H = np.conj(np.swapaxes(J_j_inv, -2, -1))
    return J_i_inv @ V_obs @ J_j_inv_H


def composite_jones(jones_list: List[np.ndarray]) -> np.ndarray:
    """
    Build composite Jones matrix from a list of Jones terms.
    
    The order follows signal propagation (sky to correlator):
        J_composite = J_N @ J_{N-1} @ ... @ J_1
    
    So jones_list[0] is closest to sky, jones_list[-1] is closest to correlator.
    
    Parameters
    ----------
    jones_list : list of ndarray
        List of Jones matrices in signal propagation order
    
    Returns
    -------
    J : ndarray (..., 2, 2)
        Composite Jones matrix
    """
    if len(jones_list) == 0:
        return np.eye(2, dtype=np.complex128)
    
    if len(jones_list) == 1:
        return jones_list[0].copy()
    
    # Multiply from right to left (last term first)
    J = jones_list[-1].copy()
    for i in range(len(jones_list) - 2, -1, -1):
        J = J @ jones_list[i]
    
    return J


def composite_jones_for_baseline(
    jones_per_antenna: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray
) -> tuple:
    """
    Get composite Jones for each baseline.
    
    Parameters
    ----------
    jones_per_antenna : ndarray (n_ant, 2, 2) or (n_time, n_ant, 2, 2)
        Jones matrices per antenna
    antenna1 : ndarray (n_bl,)
        First antenna indices
    antenna2 : ndarray (n_bl,)
        Second antenna indices
    
    Returns
    -------
    J_i : ndarray (n_bl, 2, 2) or (n_time, n_bl, 2, 2)
        Jones for antenna1
    J_j : ndarray (n_bl, 2, 2) or (n_time, n_bl, 2, 2)
        Jones for antenna2
    """
    if jones_per_antenna.ndim == 3:
        # (n_ant, 2, 2)
        J_i = jones_per_antenna[antenna1]
        J_j = jones_per_antenna[antenna2]
    elif jones_per_antenna.ndim == 4:
        # (n_time, n_ant, 2, 2)
        J_i = jones_per_antenna[:, antenna1]
        J_j = jones_per_antenna[:, antenna2]
    else:
        raise ValueError(f"Unexpected jones shape: {jones_per_antenna.shape}")
    
    return J_i, J_j


def jones_to_mueller(J: np.ndarray) -> np.ndarray:
    """
    Convert Jones matrix to Mueller matrix.
    
    The Mueller matrix M relates Stokes vectors:
        S_out = M @ S_in
    
    M = A @ (J ⊗ J*) @ A^{-1}
    
    where A is the transformation matrix between coherency and Stokes.
    
    Parameters
    ----------
    J : ndarray (..., 2, 2)
        Jones matrix
    
    Returns
    -------
    M : ndarray (..., 4, 4)
        Mueller matrix
    """
    # Transformation matrix (coherency to Stokes)
    # S = [I, Q, U, V]
    # Coherency = [XX, XY, YX, YY]
    A = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1j, 1j, 0]
    ], dtype=np.complex128) / 2
    
    A_inv = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1j],
        [0, 0, 1, -1j],
        [1, -1, 0, 0]
    ], dtype=np.complex128)
    
    # Kronecker product J ⊗ J*
    J_conj = np.conj(J)
    
    shape = J.shape[:-2]
    M_coh = np.zeros(shape + (4, 4), dtype=np.complex128)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    # Index in 4x4 matrix
                    row = 2 * i + j
                    col = 2 * k + l
                    M_coh[..., row, col] = J[..., i, k] * J_conj[..., j, l]
    
    # Transform to Stokes basis
    M = A @ M_coh @ A_inv
    
    return M


def jones_determinant(J: np.ndarray) -> np.ndarray:
    """
    Compute determinant of Jones matrices.
    
    Parameters
    ----------
    J : ndarray (..., 2, 2)
    
    Returns
    -------
    det : ndarray (...)
    """
    return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]


def jones_inverse(J: np.ndarray) -> np.ndarray:
    """
    Compute inverse of Jones matrices.
    
    For 2x2: J^{-1} = (1/det) * | j_11  -j_01 |
                                | -j_10  j_00 |
    
    Parameters
    ----------
    J : ndarray (..., 2, 2)
    
    Returns
    -------
    J_inv : ndarray (..., 2, 2)
    """
    det = jones_determinant(J)
    
    J_inv = np.zeros_like(J)
    J_inv[..., 0, 0] = J[..., 1, 1]
    J_inv[..., 0, 1] = -J[..., 0, 1]
    J_inv[..., 1, 0] = -J[..., 1, 0]
    J_inv[..., 1, 1] = J[..., 0, 0]
    
    J_inv = J_inv / det[..., np.newaxis, np.newaxis]
    
    return J_inv


def jones_hermitian(J: np.ndarray) -> np.ndarray:
    """
    Compute Hermitian (conjugate transpose) of Jones matrices.
    
    Parameters
    ----------
    J : ndarray (..., 2, 2)
    
    Returns
    -------
    J_H : ndarray (..., 2, 2)
    """
    return np.conj(np.swapaxes(J, -2, -1))
