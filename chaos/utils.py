"""
Utility functions for CHAOS calibration.
"""

import numpy as np
from .array_ops import xp, to_cpu


def corr_to_jones(data):
    """
    Convert correlation vector [XX, XY, YX, YY] to Jones matrix [[XX, XY], [YX, YY]].

    Parameters
    ----------
    data : ndarray, shape (..., 4)
        Correlation data in CASA order

    Returns
    -------
    jones : ndarray, shape (..., 2, 2)
        Jones matrix format
    """
    shape = data.shape[:-1] + (2, 2)
    jones = np.zeros(shape, dtype=data.dtype)
    jones[..., 0, 0] = data[..., 0]  # XX
    jones[..., 0, 1] = data[..., 1]  # XY
    jones[..., 1, 0] = data[..., 2]  # YX
    jones[..., 1, 1] = data[..., 3]  # YY
    return jones


def jones_to_corr(jones):
    """
    Convert Jones matrix [[XX, XY], [YX, YY]] to correlation vector [XX, XY, YX, YY].

    Parameters
    ----------
    jones : ndarray, shape (..., 2, 2)
        Jones matrix format

    Returns
    -------
    corr : ndarray, shape (..., 4)
        Correlation data in CASA order
    """
    shape = jones.shape[:-2] + (4,)
    corr = np.zeros(shape, dtype=jones.dtype)
    corr[..., 0] = jones[..., 0, 0]  # XX
    corr[..., 1] = jones[..., 0, 1]  # XY
    corr[..., 2] = jones[..., 1, 0]  # YX
    corr[..., 3] = jones[..., 1, 1]  # YY
    return corr


def compute_residuals(jones, vis_obs, vis_model, antenna1, antenna2):
    """
    Compute calibration residuals.

    Parameters
    ----------
    jones : ndarray, shape (n_ant, 2, 2)
        Jones matrices
    vis_obs : ndarray, shape (n_bl, 2, 2)
        Observed visibilities
    vis_model : ndarray, shape (n_bl, 2, 2)
        Model visibilities
    antenna1, antenna2 : ndarray, shape (n_bl,)
        Antenna indices

    Returns
    -------
    stats : dict
        Residual statistics
    """
    residuals = []
    for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
        V_pred = jones[a1] @ vis_model[idx] @ jones[a2].conj().T
        res = vis_obs[idx] - V_pred
        residuals.append(to_cpu(res))

    residuals = np.array(residuals)
    res_abs = np.abs(residuals)

    return {
        'rms': float(np.sqrt(np.mean(res_abs**2))),
        'mean': float(res_abs.mean()),
        'max': float(res_abs.max()),
        'median': float(np.median(res_abs)),
        'XX_rms': float(np.sqrt(np.mean(res_abs[:, 0, 0]**2))),
        'YY_rms': float(np.sqrt(np.mean(res_abs[:, 1, 1]**2))),
        'XY_rms': float(np.sqrt(np.mean(res_abs[:, 0, 1]**2))),
        'YX_rms': float(np.sqrt(np.mean(res_abs[:, 1, 0]**2)))
    }


__all__ = ['corr_to_jones', 'jones_to_corr', 'compute_residuals']
