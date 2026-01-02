"""
Polish CHAOS solutions using least squares optimization.

Takes the algebraic CHAOS solution as initial guess and refines
using scipy.optimize.least_squares over all baselines.
"""

import numpy as np
from scipy.optimize import least_squares
from .array_ops import to_cpu


def jones_to_params(jones, ref_antenna, mode='diagonal'):
    """
    Convert Jones matrices to parameter vector for optimization.
    
    For phase_only mode (XX):
    - ref_antenna: 0 params (phase fixed to 0)
    - other antennas: 1 param each (phase only, amplitude=1)
    
    For diagonal mode (XX):
    - ref_antenna: 1 param (amplitude only, phase=0)
    - other antennas: 2 params each (amplitude, phase)
    
    Parameters
    ----------
    jones : ndarray (n_ant, 2, 2)
        Jones matrices
    ref_antenna : int
        Reference antenna index
    mode : str
        'phase_only' or 'diagonal'
    
    Returns
    -------
    params_X : ndarray
        Parameter vector for XX
    params_Y : ndarray
        Parameter vector for YY
    """
    n_ant = jones.shape[0]
    
    if mode == 'phase_only':
        # Phase only: all amplitudes = 1, only phases vary
        # ref_antenna has phase = 0 (no params)
        # other antennas have 1 param (phase)
        params_X = []
        for i in range(n_ant):
            g = jones[i, 0, 0]
            if i != ref_antenna:
                params_X.append(np.angle(g))
        
        params_Y = []
        for i in range(n_ant):
            g = jones[i, 1, 1]
            if i != ref_antenna:
                params_Y.append(np.angle(g))
        
        return np.array(params_X), np.array(params_Y)
    
    elif mode == 'diagonal':
        # XX parameters
        params_X = []
        for i in range(n_ant):
            g = jones[i, 0, 0]
            if i == ref_antenna:
                # Only amplitude (phase fixed to 0)
                params_X.append(np.abs(g))
            else:
                # Amplitude and phase
                params_X.append(np.abs(g))
                params_X.append(np.angle(g))
        
        # YY parameters
        params_Y = []
        for i in range(n_ant):
            g = jones[i, 1, 1]
            if i == ref_antenna:
                params_Y.append(np.abs(g))
            else:
                params_Y.append(np.abs(g))
                params_Y.append(np.angle(g))
        
        return np.array(params_X), np.array(params_Y)
    
    else:
        raise NotImplementedError("Full Jones polish not yet implemented")


def params_to_jones(params_X, params_Y, n_ant, ref_antenna, mode='diagonal'):
    """
    Convert parameter vectors back to Jones matrices.
    
    Parameters
    ----------
    params_X : ndarray
        XX parameters
    params_Y : ndarray
        YY parameters
    n_ant : int
        Number of antennas
    ref_antenna : int
        Reference antenna index
    mode : str
        'phase_only' or 'diagonal'
    
    Returns
    -------
    jones : ndarray (n_ant, 2, 2)
        Jones matrices
    """
    jones = np.zeros((n_ant, 2, 2), dtype=complex)
    
    if mode == 'phase_only':
        # XX: amplitude = 1 for all, phases from params (ref = 0)
        idx = 0
        for i in range(n_ant):
            if i == ref_antenna:
                phase = 0.0
            else:
                phase = params_X[idx]
                idx += 1
            jones[i, 0, 0] = np.exp(1j * phase)
        
        # YY
        idx = 0
        for i in range(n_ant):
            if i == ref_antenna:
                phase = 0.0
            else:
                phase = params_Y[idx]
                idx += 1
            jones[i, 1, 1] = np.exp(1j * phase)
    
    elif mode == 'diagonal':
        # XX
        idx = 0
        for i in range(n_ant):
            if i == ref_antenna:
                amp = params_X[idx]
                phase = 0.0
                idx += 1
            else:
                amp = params_X[idx]
                phase = params_X[idx + 1]
                idx += 2
            jones[i, 0, 0] = amp * np.exp(1j * phase)
        
        # YY
        idx = 0
        for i in range(n_ant):
            if i == ref_antenna:
                amp = params_Y[idx]
                phase = 0.0
                idx += 1
            else:
                amp = params_Y[idx]
                phase = params_Y[idx + 1]
                idx += 2
            jones[i, 1, 1] = amp * np.exp(1j * phase)
    
    return jones


def build_residual_func(vis_obs, vis_model, antenna1, antenna2, n_ant, ref_antenna, pol='XX', mode='diagonal'):
    """
    Build residual function for least squares.
    
    Parameters
    ----------
    vis_obs : ndarray (n_bl, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_bl, 2, 2)
        Model visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    n_ant : int
        Number of antennas
    ref_antenna : int
        Reference antenna
    pol : str
        'XX' or 'YY'
    mode : str
        'phase_only' or 'diagonal'
    
    Returns
    -------
    residual_func : callable
        Function that takes params and returns residual vector
    """
    pol_idx = 0 if pol == 'XX' else 1
    
    # Extract relevant visibilities
    V_obs = vis_obs[:, pol_idx, pol_idx]
    M = vis_model[:, pol_idx, pol_idx]
    
    if mode == 'phase_only':
        def residual_func(params):
            """Compute residual vector for phase_only mode."""
            # Convert params to gains (amplitude = 1 for all)
            gains = np.zeros(n_ant, dtype=complex)
            idx = 0
            for i in range(n_ant):
                if i == ref_antenna:
                    phase = 0.0
                else:
                    phase = params[idx]
                    idx += 1
                gains[i] = np.exp(1j * phase)
            
            # Compute residuals for all baselines
            residuals = []
            for bl_idx in range(len(antenna1)):
                a1 = antenna1[bl_idx]
                a2 = antenna2[bl_idx]
                
                V_pred = gains[a1] * M[bl_idx] * np.conj(gains[a2])
                r = V_obs[bl_idx] - V_pred
                
                # Split into real and imag for least_squares
                residuals.append(r.real)
                residuals.append(r.imag)
            
            return np.array(residuals)
        
        return residual_func
    
    else:  # diagonal
        def residual_func(params):
            """Compute residual vector for diagonal mode."""
            # Convert params to gains
            gains = np.zeros(n_ant, dtype=complex)
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
            
            # Compute residuals for all baselines
            residuals = []
            for bl_idx in range(len(antenna1)):
                a1 = antenna1[bl_idx]
                a2 = antenna2[bl_idx]
                
                V_pred = gains[a1] * M[bl_idx] * np.conj(gains[a2])
                r = V_obs[bl_idx] - V_pred
                
                # Split into real and imag for least_squares
                residuals.append(r.real)
                residuals.append(r.imag)
            
            return np.array(residuals)
        
        return residual_func


def polish_jones(jones_init, vis_obs, vis_model, antenna1, antenna2, 
                 ref_antenna, mode='diagonal', max_iter=10, tol=1e-10):
    """
    Polish CHAOS solution using least squares.
    
    Parameters
    ----------
    jones_init : ndarray (n_ant, 2, 2)
        Initial Jones matrices from CHAOS
    vis_obs : ndarray (n_bl, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_bl, 2, 2)
        Model visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    ref_antenna : int
        Reference antenna
    mode : str
        'phase_only', 'diagonal', or 'full'
    max_iter : int
        Maximum iterations for least squares
    tol : float
        Tolerance for convergence (ftol, xtol, gtol)
    
    Returns
    -------
    jones_polished : ndarray (n_ant, 2, 2)
        Polished Jones matrices
    info : dict
        Optimization info
    """
    jones_init = to_cpu(jones_init)
    vis_obs = to_cpu(vis_obs)
    vis_model = to_cpu(vis_model)
    antenna1 = to_cpu(antenna1)
    antenna2 = to_cpu(antenna2)
    
    n_ant = jones_init.shape[0]
    
    print(f"[CHAOS] Polishing with least squares (max_iter={max_iter}, tol={tol})...")
    
    if mode == 'phase_only' or mode == 'diagonal':
        # Get initial parameters
        params_X, params_Y = jones_to_params(jones_init, ref_antenna, mode)
        
        # Build residual functions
        res_func_X = build_residual_func(vis_obs, vis_model, antenna1, antenna2, 
                                          n_ant, ref_antenna, 'XX', mode)
        res_func_Y = build_residual_func(vis_obs, vis_model, antenna1, antenna2, 
                                          n_ant, ref_antenna, 'YY', mode)
        
        # Initial cost
        r0_X = res_func_X(params_X)
        r0_Y = res_func_Y(params_Y)
        cost_init = np.sum(r0_X**2) + np.sum(r0_Y**2)
        print(f"[CHAOS]   Initial cost: {cost_init:.6e}")
        
        # Optimize XX
        result_X = least_squares(
            res_func_X,
            params_X,
            method='lm',
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_iter * len(params_X)
        )
        
        # Optimize YY
        result_Y = least_squares(
            res_func_Y,
            params_Y,
            method='lm',
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_iter * len(params_Y)
        )
        
        # Convert back to Jones
        jones_polished = params_to_jones(result_X.x, result_Y.x, n_ant, ref_antenna, mode)
        
        # Final cost
        r_X = res_func_X(result_X.x)
        r_Y = res_func_Y(result_Y.x)
        cost_final = np.sum(r_X**2) + np.sum(r_Y**2)
        print(f"[CHAOS]   Final cost: {cost_final:.6e}")
        print(f"[CHAOS]   Improvement: {cost_init/cost_final:.2f}x")
        print(f"[CHAOS]   XX iterations: {result_X.nfev}, YY iterations: {result_Y.nfev}")
        
        info = {
            'cost_init': cost_init,
            'cost_final': cost_final,
            'nfev_X': result_X.nfev,
            'nfev_Y': result_Y.nfev,
            'success_X': result_X.success,
            'success_Y': result_Y.success
        }
        
        return jones_polished, info
    
    else:
        raise NotImplementedError("Full Jones polish not yet implemented")


__all__ = ['polish_jones']
