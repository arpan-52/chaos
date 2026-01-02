"""
Least Squares Polish for Jones Solutions.

Refines algebraic chain solution using least squares optimization
over all baselines simultaneously.

This is the key step that achieves CASA-level accuracy.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Tuple, Optional


def polish_jones(
    jones_init: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_antenna: int,
    mode: str = "diagonal",
    jones_type: str = "G",
    max_iter: int = 100,
    tol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Polish Jones solution using least squares.
    
    Minimizes:
        sum_ij |V_obs_ij - J_i @ M_ij @ J_j^H|^2
    
    subject to reference antenna constraints.
    
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
        Reference antenna (constrained)
    mode : str
        'phase_only', 'diagonal', or 'full'
    jones_type : str
        'G', 'G[p]', 'B', 'K', 'D' - affects reference constraint
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance (ftol, xtol, gtol)
    verbose : bool
        Print progress
    
    Returns
    -------
    jones_polished : ndarray (n_ant, 2, 2)
        Refined Jones matrices
    info : dict
        Optimization info
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
    
    # Choose polish function based on mode
    if mode == "phase_only":
        return _polish_phase_only(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, max_iter, tol, verbose
        )
    elif mode == "diagonal":
        return _polish_diagonal(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, jones_type, max_iter, tol, verbose
        )
    else:  # full
        return _polish_full(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, jones_type, max_iter, tol, verbose
        )


def _polish_phase_only(
    jones_init: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_antenna: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> Tuple[np.ndarray, Dict]:
    """
    Polish phase-only Jones (G[p]).
    
    Parameters: N-1 phases per polarization (ref antenna phase = 0)
    Total: 2*(N-1) real parameters
    
    Reference constraint: phi_X = phi_Y = 0 for ref antenna
    """
    n_ant = jones_init.shape[0]
    n_bl = len(antenna1)
    
    # Extract initial phases
    # params = [phi_X_0, phi_X_1, ..., phi_Y_0, phi_Y_1, ...]
    # excluding ref_antenna for each pol
    params_init = []
    for pol in range(2):
        for ant in range(n_ant):
            if ant == ref_antenna:
                continue
            phase = np.angle(jones_init[ant, pol, pol])
            params_init.append(phase)
    
    params_init = np.array(params_init)
    
    def params_to_jones(params):
        """Convert parameter vector to Jones matrices."""
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        
        idx = 0
        for pol in range(2):
            for ant in range(n_ant):
                if ant == ref_antenna:
                    jones[ant, pol, pol] = 1.0  # Phase = 0, Amp = 1
                else:
                    jones[ant, pol, pol] = np.exp(1j * params[idx])
                    idx += 1
        
        return jones
    
    def residual_func(params):
        """Compute residuals for all baselines."""
        jones = params_to_jones(params)
        
        residuals = []
        for bl_idx in range(n_bl):
            a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
            
            J_i = jones[a1]
            J_j = jones[a2]
            M = vis_model[bl_idx]
            V_obs = vis_obs[bl_idx]
            
            # V_pred = J_i @ M @ J_j^H
            V_pred = J_i @ M @ np.conj(J_j.T)
            
            # Residual for diagonal elements (parallel hand)
            R = V_obs - V_pred
            residuals.extend([R[0, 0].real, R[0, 0].imag])
            residuals.extend([R[1, 1].real, R[1, 1].imag])
        
        return np.array(residuals)
    
    # Initial cost
    r0 = residual_func(params_init)
    cost_init = 0.5 * np.sum(r0**2)
    
    # Optimize
    result = least_squares(
        residual_func,
        params_init,
        method='lm',
        max_nfev=max_iter * len(params_init),
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    
    jones_polished = params_to_jones(result.x)
    
    cost_final = 0.5 * np.sum(result.fun**2)
    
    if verbose:
        print(f"[CHAOS] Polish: cost {cost_init:.6e} -> {cost_final:.6e}")
        print(f"[CHAOS] Polish: {result.nfev} function evaluations, status={result.status}")
    
    info = {
        "cost_init": cost_init,
        "cost_final": cost_final,
        "n_iter": result.nfev,
        "status": result.status,
        "message": result.message,
    }
    
    return jones_polished, info


def _polish_diagonal(
    jones_init: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_antenna: int,
    jones_type: str,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> Tuple[np.ndarray, Dict]:
    """
    Polish diagonal Jones (G, B, K).
    
    Parameters per polarization:
        - Amplitude: N values (all antennas, including ref)
        - Phase: N-1 values (ref antenna phase = 0)
    
    Total: 2*(N + N-1) = 4N - 2 real parameters
    
    Reference constraint: phase = 0 for both pols
    """
    n_ant = jones_init.shape[0]
    n_bl = len(antenna1)
    
    # Extract initial parameters
    # params = [amp_X_0..amp_X_{N-1}, phase_X_0..phase_X_{N-2}, 
    #           amp_Y_0..amp_Y_{N-1}, phase_Y_0..phase_Y_{N-2}]
    # Note: phases exclude ref_antenna
    params_init = []
    
    for pol in range(2):
        # Amplitudes (all antennas)
        for ant in range(n_ant):
            amp = np.abs(jones_init[ant, pol, pol])
            params_init.append(amp)
        
        # Phases (exclude ref_antenna)
        for ant in range(n_ant):
            if ant == ref_antenna:
                continue
            phase = np.angle(jones_init[ant, pol, pol])
            params_init.append(phase)
    
    params_init = np.array(params_init)
    
    # Calculate parameter indices
    n_amp_per_pol = n_ant
    n_phase_per_pol = n_ant - 1
    n_per_pol = n_amp_per_pol + n_phase_per_pol
    
    def params_to_jones(params):
        """Convert parameter vector to Jones matrices."""
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        
        for pol in range(2):
            offset = pol * n_per_pol
            
            # Amplitudes
            amps = params[offset:offset + n_amp_per_pol]
            
            # Phases
            phase_offset = offset + n_amp_per_pol
            phases_partial = params[phase_offset:phase_offset + n_phase_per_pol]
            
            # Build full phase array (insert 0 at ref_antenna)
            phases = np.zeros(n_ant)
            idx = 0
            for ant in range(n_ant):
                if ant == ref_antenna:
                    phases[ant] = 0.0
                else:
                    phases[ant] = phases_partial[idx]
                    idx += 1
            
            # Construct diagonal elements
            for ant in range(n_ant):
                jones[ant, pol, pol] = amps[ant] * np.exp(1j * phases[ant])
        
        return jones
    
    def residual_func(params):
        """Compute residuals for all baselines."""
        jones = params_to_jones(params)
        
        residuals = []
        for bl_idx in range(n_bl):
            a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
            
            J_i = jones[a1]
            J_j = jones[a2]
            M = vis_model[bl_idx]
            V_obs = vis_obs[bl_idx]
            
            V_pred = J_i @ M @ np.conj(J_j.T)
            
            # Residuals for parallel hands
            R = V_obs - V_pred
            residuals.extend([R[0, 0].real, R[0, 0].imag])
            residuals.extend([R[1, 1].real, R[1, 1].imag])
        
        return np.array(residuals)
    
    # Initial cost
    r0 = residual_func(params_init)
    cost_init = 0.5 * np.sum(r0**2)
    
    # Optimize
    result = least_squares(
        residual_func,
        params_init,
        method='lm',
        max_nfev=max_iter * len(params_init),
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    
    jones_polished = params_to_jones(result.x)
    cost_final = 0.5 * np.sum(result.fun**2)
    
    if verbose:
        print(f"[CHAOS] Polish: cost {cost_init:.6e} -> {cost_final:.6e}")
        print(f"[CHAOS] Polish: {result.nfev} function evaluations, status={result.status}")
    
    info = {
        "cost_init": cost_init,
        "cost_final": cost_final,
        "n_iter": result.nfev,
        "status": result.status,
        "message": result.message,
    }
    
    return jones_polished, info


def _polish_full(
    jones_init: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_antenna: int,
    jones_type: str,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> Tuple[np.ndarray, Dict]:
    """
    Polish full 2x2 Jones (D-terms).
    
    For D-terms:
        D = | 1    d_01 |
            | d_10  1   |
    
    Parameters: d_01 and d_10 for each antenna (complex = 2 real each)
    Reference constraint: d_01 = 0 for ref antenna (to break degeneracy)
    
    Total: 2*(N-1) + 2*N = 4N - 2 real parameters for D-terms
    """
    n_ant = jones_init.shape[0]
    n_bl = len(antenna1)
    
    if jones_type == "D":
        return _polish_d_terms(
            jones_init, vis_obs, vis_model, antenna1, antenna2,
            ref_antenna, max_iter, tol, verbose
        )
    
    # General full Jones
    # Parameters: real and imag of all 4 elements for each antenna
    # Reference: diagonal phase = 0
    
    params_init = []
    for ant in range(n_ant):
        J = jones_init[ant]
        for i in range(2):
            for j in range(2):
                params_init.append(J[i, j].real)
                params_init.append(J[i, j].imag)
    
    params_init = np.array(params_init)
    
    def params_to_jones(params):
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        idx = 0
        for ant in range(n_ant):
            for i in range(2):
                for j in range(2):
                    jones[ant, i, j] = params[idx] + 1j * params[idx + 1]
                    idx += 2
        
        # Apply reference constraint: phase = 0 for diagonal
        if ref_antenna < n_ant:
            for p in range(2):
                val = jones[ref_antenna, p, p]
                if np.abs(val) > 1e-10:
                    phase = np.angle(val)
                    jones[ref_antenna, p, p] = np.abs(val)  # Remove phase
        
        return jones
    
    def residual_func(params):
        jones = params_to_jones(params)
        
        residuals = []
        for bl_idx in range(n_bl):
            a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
            
            J_i = jones[a1]
            J_j = jones[a2]
            M = vis_model[bl_idx]
            V_obs = vis_obs[bl_idx]
            
            V_pred = J_i @ M @ np.conj(J_j.T)
            R = V_obs - V_pred
            
            # All 4 correlations
            for i in range(2):
                for j in range(2):
                    residuals.extend([R[i, j].real, R[i, j].imag])
        
        return np.array(residuals)
    
    r0 = residual_func(params_init)
    cost_init = 0.5 * np.sum(r0**2)
    
    result = least_squares(
        residual_func,
        params_init,
        method='lm',
        max_nfev=max_iter * len(params_init),
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    
    jones_polished = params_to_jones(result.x)
    cost_final = 0.5 * np.sum(result.fun**2)
    
    if verbose:
        print(f"[CHAOS] Polish: cost {cost_init:.6e} -> {cost_final:.6e}")
        print(f"[CHAOS] Polish: {result.nfev} function evaluations, status={result.status}")
    
    info = {
        "cost_init": cost_init,
        "cost_final": cost_final,
        "n_iter": result.nfev,
        "status": result.status,
        "message": result.message,
    }
    
    return jones_polished, info


def _polish_d_terms(
    jones_init: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ref_antenna: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> Tuple[np.ndarray, Dict]:
    """
    Polish D-terms (leakage).
    
    D = | 1    d_01 |
        | d_10  1   |
    
    Reference constraint: d_01 = 0 for ref antenna
    
    Parameters:
        - d_01 for all antennas except ref (complex = 2 real)
        - d_10 for all antennas (complex = 2 real)
    
    Total: 2*(N-1) + 2*N = 4*N - 2 real parameters
    """
    n_ant = jones_init.shape[0]
    n_bl = len(antenna1)
    
    # Extract initial parameters
    params_init = []
    
    # d_01 (off-diagonal, row 0 col 1) - exclude ref antenna
    for ant in range(n_ant):
        if ant == ref_antenna:
            continue
        d = jones_init[ant, 0, 1]
        params_init.extend([d.real, d.imag])
    
    # d_10 (off-diagonal, row 1 col 0) - all antennas
    for ant in range(n_ant):
        d = jones_init[ant, 1, 0]
        params_init.extend([d.real, d.imag])
    
    params_init = np.array(params_init)
    
    n_d01 = 2 * (n_ant - 1)  # complex values for d_01
    
    def params_to_jones(params):
        jones = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        
        # Set diagonals to 1
        for ant in range(n_ant):
            jones[ant, 0, 0] = 1.0
            jones[ant, 1, 1] = 1.0
        
        # d_01
        idx = 0
        for ant in range(n_ant):
            if ant == ref_antenna:
                jones[ant, 0, 1] = 0.0  # Reference constraint
            else:
                jones[ant, 0, 1] = params[idx] + 1j * params[idx + 1]
                idx += 2
        
        # d_10
        for ant in range(n_ant):
            jones[ant, 1, 0] = params[n_d01 + 2*ant] + 1j * params[n_d01 + 2*ant + 1]
        
        return jones
    
    def residual_func(params):
        jones = params_to_jones(params)
        
        residuals = []
        for bl_idx in range(n_bl):
            a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
            
            J_i = jones[a1]
            J_j = jones[a2]
            M = vis_model[bl_idx]
            V_obs = vis_obs[bl_idx]
            
            V_pred = J_i @ M @ np.conj(J_j.T)
            R = V_obs - V_pred
            
            # All 4 correlations for D-terms
            for i in range(2):
                for j in range(2):
                    residuals.extend([R[i, j].real, R[i, j].imag])
        
        return np.array(residuals)
    
    r0 = residual_func(params_init)
    cost_init = 0.5 * np.sum(r0**2)
    
    result = least_squares(
        residual_func,
        params_init,
        method='lm',
        max_nfev=max_iter * len(params_init),
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )
    
    jones_polished = params_to_jones(result.x)
    cost_final = 0.5 * np.sum(result.fun**2)
    
    if verbose:
        print(f"[CHAOS] D-term polish: cost {cost_init:.6e} -> {cost_final:.6e}")
        print(f"[CHAOS] Polish: {result.nfev} function evaluations, status={result.status}")
    
    info = {
        "cost_init": cost_init,
        "cost_final": cost_final,
        "n_iter": result.nfev,
        "status": result.status,
        "message": result.message,
    }
    
    return jones_polished, info
