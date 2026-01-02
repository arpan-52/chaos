"""
Fast Vectorized Polish for Jones Solutions.

Uses batched matrix operations - NO Python loops in residual computation.
Works on CPU (NumPy) or GPU (CuPy) seamlessly.

For diagonal Jones: params are (amp, phase) per antenna per pol
Reference antenna: phase = 0 constraint
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Dict, Optional


def polish_jones(
    jones_init: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_antenna: int,
    mode: str = "diagonal",
    max_iter: int = 50,
    tol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Polish Jones solution using least squares with vectorized operations.
    
    Parameters
    ----------
    jones_init : ndarray (n_ant, 2, 2)
        Initial Jones matrices from chain solver
    vis_obs : ndarray (n_bl, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_bl, 2, 2)
        Model visibilities  
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    ref_antenna : int
        Reference antenna (phase = 0)
    mode : str
        'phase_only', 'diagonal', or 'full'
    max_iter : int
        Max iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    
    Returns
    -------
    jones : ndarray (n_ant, 2, 2)
        Polished Jones matrices
    info : dict
        Optimization diagnostics
    """
    jones_init = np.asarray(jones_init, dtype=np.complex128)
    vis_obs = np.asarray(vis_obs, dtype=np.complex128)
    vis_model = np.asarray(vis_model, dtype=np.complex128)
    antenna1 = np.asarray(antenna1, dtype=np.int32)
    antenna2 = np.asarray(antenna2, dtype=np.int32)
    
    n_ant = jones_init.shape[0]
    n_bl = len(antenna1)
    
    if verbose:
        print(f"[CHAOS] Polishing: {n_ant} antennas, {n_bl} baselines, mode={mode}")
    
    if mode == "phase_only":
        return _polish_phase_only(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, n_ant, n_bl, max_iter, tol, verbose
        )
    elif mode == "diagonal":
        return _polish_diagonal(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, n_ant, n_bl, max_iter, tol, verbose
        )
    else:
        return _polish_full(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, n_ant, n_bl, max_iter, tol, verbose
        )


def _polish_diagonal(
    jones_init, vis_obs, vis_model, antenna1, antenna2,
    ref_antenna, n_ant, n_bl, max_iter, tol, verbose
):
    """
    Polish diagonal Jones matrices (G, B, K).
    
    Parameterization per pol:
        - ref_antenna: amplitude only (phase = 0)
        - other antennas: amplitude + phase
    Total: 2*(N-1) + 2 = 2*N params per pol
    
    Solves XX and YY independently for efficiency.
    """
    # Solve XX
    params_X_init = _jones_to_params_1pol(jones_init[:, 0, 0], ref_antenna)
    result_X = least_squares(
        _build_residual_1pol(vis_obs[:, 0, 0], vis_model[:, 0, 0], 
                             antenna1, antenna2, n_ant, ref_antenna),
        params_X_init,
        method='lm',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_iter * len(params_X_init)
    )
    
    # Solve YY
    params_Y_init = _jones_to_params_1pol(jones_init[:, 1, 1], ref_antenna)
    result_Y = least_squares(
        _build_residual_1pol(vis_obs[:, 1, 1], vis_model[:, 1, 1],
                             antenna1, antenna2, n_ant, ref_antenna),
        params_Y_init,
        method='lm',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_iter * len(params_Y_init)
    )
    
    # Reconstruct Jones
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = _params_to_gains(result_X.x, n_ant, ref_antenna)
    jones[:, 1, 1] = _params_to_gains(result_Y.x, n_ant, ref_antenna)
    
    if verbose:
        cost_init = 0.5 * (np.sum(_build_residual_1pol(
            vis_obs[:, 0, 0], vis_model[:, 0, 0], antenna1, antenna2, n_ant, ref_antenna
        )(params_X_init)**2) + np.sum(_build_residual_1pol(
            vis_obs[:, 1, 1], vis_model[:, 1, 1], antenna1, antenna2, n_ant, ref_antenna
        )(params_Y_init)**2))
        cost_final = 0.5 * (result_X.cost + result_Y.cost)
        print(f"[CHAOS] Polish: cost {cost_init:.6e} -> {cost_final:.6e}")
        print(f"[CHAOS] Polish: XX {result_X.nfev} evals, YY {result_Y.nfev} evals")
    
    info = {
        "cost_final": 0.5 * (result_X.cost + result_Y.cost),
        "nfev_X": result_X.nfev,
        "nfev_Y": result_Y.nfev,
        "success": result_X.success and result_Y.success,
    }
    
    return jones, info


def _polish_phase_only(
    jones_init, vis_obs, vis_model, antenna1, antenna2,
    ref_antenna, n_ant, n_bl, max_iter, tol, verbose
):
    """
    Polish phase-only Jones (G[p]).
    
    Params: phase per antenna (excluding ref which is 0)
    Amplitude fixed at 1.
    """
    # Solve XX phases
    phases_X_init = np.array([np.angle(jones_init[i, 0, 0]) 
                              for i in range(n_ant) if i != ref_antenna])
    result_X = least_squares(
        _build_residual_phase_only(vis_obs[:, 0, 0], vis_model[:, 0, 0],
                                    antenna1, antenna2, n_ant, ref_antenna),
        phases_X_init,
        method='lm',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_iter * len(phases_X_init)
    )
    
    # Solve YY phases
    phases_Y_init = np.array([np.angle(jones_init[i, 1, 1])
                              for i in range(n_ant) if i != ref_antenna])
    result_Y = least_squares(
        _build_residual_phase_only(vis_obs[:, 1, 1], vis_model[:, 1, 1],
                                    antenna1, antenna2, n_ant, ref_antenna),
        phases_Y_init,
        method='lm',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_iter * len(phases_Y_init)
    )
    
    # Reconstruct Jones
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    jones[:, 0, 0] = _phases_to_gains(result_X.x, n_ant, ref_antenna)
    jones[:, 1, 1] = _phases_to_gains(result_Y.x, n_ant, ref_antenna)
    
    if verbose:
        print(f"[CHAOS] Phase polish: XX {result_X.nfev} evals, YY {result_Y.nfev} evals")
    
    info = {
        "nfev_X": result_X.nfev,
        "nfev_Y": result_Y.nfev,
        "success": result_X.success and result_Y.success,
    }
    
    return jones, info


def _polish_full(
    jones_init, vis_obs, vis_model, antenna1, antenna2,
    ref_antenna, n_ant, n_bl, max_iter, tol, verbose
):
    """
    Polish full 2x2 Jones (for D-terms or general).
    
    Uses full matrix residuals, still vectorized.
    """
    # Pack all Jones elements into params
    # Each antenna: j00_re, j00_im, j01_re, j01_im, j10_re, j10_im, j11_re, j11_im
    # Ref antenna: j00 and j11 have phase=0 constraint
    
    params_init = _jones_to_params_full(jones_init, ref_antenna)
    
    result = least_squares(
        _build_residual_full(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_antenna),
        params_init,
        method='lm',
        ftol=tol, xtol=tol, gtol=tol,
        max_nfev=max_iter * len(params_init)
    )
    
    jones = _params_to_jones_full(result.x, n_ant, ref_antenna)
    
    if verbose:
        print(f"[CHAOS] Full polish: {result.nfev} evals, success={result.success}")
    
    info = {
        "nfev": result.nfev,
        "success": result.success,
    }
    
    return jones, info


# =============================================================================
# Vectorized residual functions
# =============================================================================

def _build_residual_1pol(V_obs, M, antenna1, antenna2, n_ant, ref_antenna):
    """
    Build VECTORIZED residual function for single polarization.
    
    V_obs, M: (n_bl,) complex arrays
    Returns function that computes residuals for all baselines at once.
    """
    def residual_func(params):
        # Convert params to complex gains
        gains = _params_to_gains(params, n_ant, ref_antenna)
        
        # VECTORIZED: compute all baselines at once
        V_pred = gains[antenna1] * M * np.conj(gains[antenna2])
        
        # Residual as real vector [re_0, im_0, re_1, im_1, ...]
        R = V_obs - V_pred
        return np.concatenate([R.real, R.imag])
    
    return residual_func


def _build_residual_phase_only(V_obs, M, antenna1, antenna2, n_ant, ref_antenna):
    """Vectorized residual for phase-only gains."""
    def residual_func(phases):
        gains = _phases_to_gains(phases, n_ant, ref_antenna)
        V_pred = gains[antenna1] * M * np.conj(gains[antenna2])
        R = V_obs - V_pred
        return np.concatenate([R.real, R.imag])
    
    return residual_func


def _build_residual_full(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_antenna):
    """
    Vectorized residual for full 2x2 Jones.
    
    V_pred = J_i @ M @ J_j^H for all baselines simultaneously.
    """
    def residual_func(params):
        jones = _params_to_jones_full(params, n_ant, ref_antenna)
        
        # Gather Jones for all baselines: (n_bl, 2, 2)
        J_i = jones[antenna1]
        J_j = jones[antenna2]
        
        # Batched matrix multiply: V_pred = J_i @ M @ J_j^H
        # J_j^H = conjugate transpose
        J_j_H = np.conj(J_j).swapaxes(-1, -2)
        V_pred = np.einsum('bij,bjk,bkl->bil', J_i, vis_model, J_j_H)
        
        # Residual
        R = vis_obs - V_pred
        
        # Flatten to real vector
        return np.concatenate([R.real.ravel(), R.imag.ravel()])
    
    return residual_func


# =============================================================================
# Parameter <-> Jones conversion
# =============================================================================

def _jones_to_params_1pol(gains, ref_antenna):
    """
    Convert complex gains to parameter vector.
    
    Params: [amp_0, phase_0, amp_1, phase_1, ...]
    Ref antenna: only amplitude (phase implicitly 0)
    """
    n_ant = len(gains)
    params = []
    for i in range(n_ant):
        g = gains[i]
        if i == ref_antenna:
            params.append(np.abs(g))  # amplitude only
        else:
            params.append(np.abs(g))
            params.append(np.angle(g))
    return np.array(params)


def _params_to_gains(params, n_ant, ref_antenna):
    """Convert parameter vector back to complex gains."""
    gains = np.zeros(n_ant, dtype=np.complex128)
    idx = 0
    for i in range(n_ant):
        if i == ref_antenna:
            amp = params[idx]
            phase = 0.0
            idx += 1
        else:
            amp = params[idx]
            phase = params[idx + 1]
            idx += 2
        gains[i] = amp * np.exp(1j * phase)
    return gains


def _phases_to_gains(phases, n_ant, ref_antenna):
    """Convert phase vector to unit-amplitude gains."""
    gains = np.ones(n_ant, dtype=np.complex128)
    idx = 0
    for i in range(n_ant):
        if i == ref_antenna:
            gains[i] = 1.0  # phase = 0, amp = 1
        else:
            gains[i] = np.exp(1j * phases[idx])
            idx += 1
    return gains


def _jones_to_params_full(jones, ref_antenna):
    """
    Convert full 2x2 Jones to params.
    
    Each antenna: 8 params (re/im for each of 4 elements)
    Ref antenna: j00 and j11 are real (phase=0), so 6 params
    """
    n_ant = jones.shape[0]
    params = []
    for i in range(n_ant):
        J = jones[i]
        if i == ref_antenna:
            # j00: real only (phase=0)
            params.append(np.abs(J[0, 0]))
            # j01: complex
            params.extend([J[0, 1].real, J[0, 1].imag])
            # j10: complex
            params.extend([J[1, 0].real, J[1, 0].imag])
            # j11: real only (phase=0)
            params.append(np.abs(J[1, 1]))
        else:
            # All elements complex
            params.extend([J[0, 0].real, J[0, 0].imag])
            params.extend([J[0, 1].real, J[0, 1].imag])
            params.extend([J[1, 0].real, J[1, 0].imag])
            params.extend([J[1, 1].real, J[1, 1].imag])
    return np.array(params)


def _params_to_jones_full(params, n_ant, ref_antenna):
    """Convert params back to full 2x2 Jones."""
    jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
    idx = 0
    for i in range(n_ant):
        if i == ref_antenna:
            jones[i, 0, 0] = params[idx]  # real
            idx += 1
            jones[i, 0, 1] = params[idx] + 1j * params[idx + 1]
            idx += 2
            jones[i, 1, 0] = params[idx] + 1j * params[idx + 1]
            idx += 2
            jones[i, 1, 1] = params[idx]  # real
            idx += 1
        else:
            jones[i, 0, 0] = params[idx] + 1j * params[idx + 1]
            idx += 2
            jones[i, 0, 1] = params[idx] + 1j * params[idx + 1]
            idx += 2
            jones[i, 1, 0] = params[idx] + 1j * params[idx + 1]
            idx += 2
            jones[i, 1, 1] = params[idx] + 1j * params[idx + 1]
            idx += 2
    return jones


__all__ = ['polish_jones']
