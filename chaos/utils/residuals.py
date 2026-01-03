"""
Residual Computation. 

Compute calibration residuals to assess solution quality.
"""

import numpy as np
from typing import Dict


def compute_residuals(
    jones: np.ndarray,
    vis_obs: np.ndarray,
    vis_model: np. ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
) -> Dict[str, float]:
    """
    Compute residual statistics after calibration (vectorized).
    
    Residual = V_obs - J_i @ M @ J_j^H
    
    Parameters
    ----------
    jones : ndarray (n_ant, 2, 2)
        Jones matrices
    vis_obs : ndarray (n_bl, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_bl, 2, 2)
        Model visibilities
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    
    Returns
    -------
    stats : dict
        rms:  overall RMS
        XX_rms, YY_rms, XY_rms, YX_rms:  per-correlation RMS
        max_residual: maximum absolute residual
    """
    # Gather Jones matrices for all baselines at once
    J_i = jones[antenna1]  # (n_bl, 2, 2)
    J_j = jones[antenna2]  # (n_bl, 2, 2)
    
    # Vectorized:  V_pred = J_i @ M @ J_j^H
    J_j_H = np.conj(J_j.swapaxes(-2, -1))  # (n_bl, 2, 2)
    V_pred = np.einsum('bij,bjk,bkl->bil', J_i, vis_model, J_j_H)
    
    residuals = vis_obs - V_pred
    
    # Compute statistics
    res_abs = np.abs(residuals)
    
    stats = {
        "rms": np. sqrt(np.mean(res_abs**2)),
        "XX_rms":  np.sqrt(np.mean(res_abs[: , 0, 0]**2)),
        "XY_rms":  np.sqrt(np.mean(res_abs[:, 0, 1]**2)),
        "YX_rms": np.sqrt(np.mean(res_abs[:, 1, 0]**2)),
        "YY_rms": np.sqrt(np.mean(res_abs[:, 1, 1]**2)),
        "max_residual": res_abs.max(),
        "mean_residual": res_abs.mean(),
    }
    
    return stats


def compute_phase_residuals(
    jones: np. ndarray,
    vis_obs: np.ndarray,
    vis_model:  np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
) -> Dict[str, float]: 
    """
    Compute phase residuals (in degrees) for diagonal terms (vectorized).
    
    Parameters
    ----------
    jones : ndarray (n_ant, 2, 2)
    vis_obs : ndarray (n_bl, 2, 2)
    vis_model : ndarray (n_bl, 2, 2)
    antenna1, antenna2 :  ndarray (n_bl,)
    
    Returns
    -------
    stats : dict
        XX_phase_rms, YY_phase_rms:  phase RMS in degrees
    """
    # Gather Jones matrices for all baselines
    J_i = jones[antenna1]  # (n_bl, 2, 2)
    J_j = jones[antenna2]  # (n_bl, 2, 2)
    
    # Vectorized: V_pred = J_i @ M @ J_j^H
    J_j_H = np.conj(J_j.swapaxes(-2, -1))
    V_pred = np.einsum('bij,bjk,bkl->bil', J_i, vis_model, J_j_H)
    
    # XX phase residuals
    V_pred_XX = V_pred[:, 0, 0]
    V_obs_XX = vis_obs[:, 0, 0]
    mask_XX = np.abs(V_pred_XX) > 1e-10
    phase_diff_XX = np. angle(V_obs_XX[mask_XX]) - np.angle(V_pred_XX[mask_XX])
    
    # YY phase residuals
    V_pred_YY = V_pred[:, 1, 1]
    V_obs_YY = vis_obs[:, 1, 1]
    mask_YY = np.abs(V_pred_YY) > 1e-10
    phase_diff_YY = np.angle(V_obs_YY[mask_YY]) - np.angle(V_pred_YY[mask_YY])
    
    stats = {
        "XX_phase_rms": np.std(np.degrees(phase_diff_XX)) if len(phase_diff_XX) > 0 else 0.0,
        "YY_phase_rms": np. std(np.degrees(phase_diff_YY)) if len(phase_diff_YY) > 0 else 0.0,
    }
    
    return stats