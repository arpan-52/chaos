"""
Baseline quality calculation for CHAOS.

Quality metric: median_amplitude / mad_spread
Higher = better SNR, more trustworthy baseline
"""

import numpy as np


def compute_baseline_quality(vis_obs, flags, min_points=1):
    """
    Compute baseline quality using amplitude statistics.

    Parameters
    ----------
    vis_obs : ndarray, shape (n_bl, 2, 2)
        Observed visibilities (Jones matrices)
    flags : ndarray, shape (n_bl, 2, 2) bool
        Flags (True = flagged)
    min_points : int
        Minimum unflagged points

    Returns
    -------
    quality : ndarray, shape (n_bl,)
        Baseline quality (0 = bad, higher = better)
    """
    n_bl = vis_obs.shape[0]
    quality = np.zeros(n_bl)

    for bl_idx in range(n_bl):
        bl_data = vis_obs[bl_idx]
        bl_flags = flags[bl_idx]

        # Get unflagged amplitudes
        amp = np.abs(bl_data)
        amp_unflagged = amp[~bl_flags]
        amp_unflagged = amp_unflagged[amp_unflagged > 0]

        if len(amp_unflagged) < min_points:
            quality[bl_idx] = 0.0
            continue

        # Robust median amplitude
        median_amp = np.median(amp_unflagged)

        # MAD spread
        mad_spread = np.median(np.abs(amp_unflagged - median_amp))

        if mad_spread == 0:
            # Perfect coherence
            quality[bl_idx] = median_amp
        else:
            # Quality = SNR-like
            quality[bl_idx] = median_amp / mad_spread

    return quality


def compute_quality_matrix(vis_obs, flags, antenna1, antenna2, n_ant):
    """
    Compute quality matrix Q[i,j] for all antenna pairs.

    Parameters
    ----------
    vis_obs : ndarray, shape (n_bl, 2, 2)
    flags : ndarray, shape (n_bl, 2, 2)
    antenna1, antenna2 : ndarray, shape (n_bl,)
    n_ant : int

    Returns
    -------
    Q : ndarray, shape (n_ant, n_ant)
        Quality matrix (symmetric)
    """
    quality = compute_baseline_quality(vis_obs, flags)

    Q = np.zeros((n_ant, n_ant))
    for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
        Q[a1, a2] = quality[idx]
        Q[a2, a1] = quality[idx]

    return Q


__all__ = ['compute_baseline_quality', 'compute_quality_matrix']
