"""
RFI Flagging Module.

Robust RFI detection and flagging using median-based statistics.
Uses Median Absolute Deviation (MAD) which is resistant to outliers.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FlagStats:
    """Statistics from flagging operation."""
    n_total: int
    n_flagged: int
    n_new_flags: int
    fraction_flagged: float
    threshold_used: float


def mad(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute Median Absolute Deviation.
    
    MAD = median(|x - median(x)|)
    
    This is a robust measure of spread, resistant to outliers.
    
    Parameters
    ----------
    x : ndarray
        Input data
    axis : int, optional
        Axis along which to compute
    
    Returns
    -------
    mad : ndarray or scalar
    """
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis)


def mad_to_sigma(mad_value: float) -> float:
    """
    Convert MAD to equivalent Gaussian sigma.
    
    For a Gaussian distribution:
        sigma = MAD * 1.4826
    
    Parameters
    ----------
    mad_value : float
        MAD value
    
    Returns
    -------
    sigma : float
        Equivalent Gaussian standard deviation
    """
    return mad_value * 1.4826


def flag_rfi_mad(
    vis: np.ndarray,
    sigma: float = 5.0,
    existing_flags: Optional[np.ndarray] = None,
    per_baseline: bool = True,
    per_correlation: bool = True,
) -> Tuple[np.ndarray, FlagStats]:
    """
    Flag RFI using robust MAD-based detection.
    
    Flags data points that deviate from the median by more than
    `sigma` times the MAD-derived standard deviation.
    
    This is robust because:
    - Median is not affected by outliers
    - MAD is a robust measure of spread
    - Even with 50% RFI, the median will still be valid
    
    Parameters
    ----------
    vis : ndarray (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
        Visibility data
    sigma : float
        Number of sigma for threshold (default: 5)
    existing_flags : ndarray, optional
        Existing flags to combine with
    per_baseline : bool
        Compute statistics per baseline (default: True)
    per_correlation : bool
        Compute statistics per correlation (default: True)
    
    Returns
    -------
    flags : ndarray (same shape as vis)
        Boolean flags (True = flagged)
    stats : FlagStats
        Flagging statistics
    """
    # Work with amplitude
    amp = np.abs(vis)
    
    # Initialize flags
    flags = np.zeros(amp.shape, dtype=bool)
    
    if existing_flags is not None:
        flags |= existing_flags
    
    n_flagged_before = flags.sum()
    
    if per_baseline and per_correlation:
        # Compute threshold per baseline per correlation
        if amp.ndim == 3:
            # (n_bl, 2, 2)
            for i in range(2):
                for j in range(2):
                    amp_corr = amp[:, i, j]
                    med = np.median(amp_corr)
                    mad_val = mad(amp_corr)
                    sigma_robust = mad_to_sigma(mad_val)
                    
                    if sigma_robust > 0:
                        threshold = sigma * sigma_robust
                        flags[:, i, j] |= np.abs(amp_corr - med) > threshold
        else:
            # (n_bl, n_chan, 2, 2)
            for bl in range(amp.shape[0]):
                for i in range(2):
                    for j in range(2):
                        amp_corr = amp[bl, :, i, j]
                        med = np.median(amp_corr)
                        mad_val = mad(amp_corr)
                        sigma_robust = mad_to_sigma(mad_val)
                        
                        if sigma_robust > 0:
                            threshold = sigma * sigma_robust
                            flags[bl, :, i, j] |= np.abs(amp_corr - med) > threshold
    
    elif per_correlation:
        # Global threshold per correlation
        for i in range(2):
            for j in range(2):
                if amp.ndim == 3:
                    amp_corr = amp[:, i, j]
                else:
                    amp_corr = amp[:, :, i, j].flatten()
                
                med = np.median(amp_corr)
                mad_val = mad(amp_corr)
                sigma_robust = mad_to_sigma(mad_val)
                
                if sigma_robust > 0:
                    threshold = sigma * sigma_robust
                    if amp.ndim == 3:
                        flags[:, i, j] |= np.abs(amp[:, i, j] - med) > threshold
                    else:
                        flags[:, :, i, j] |= np.abs(amp[:, :, i, j] - med) > threshold
    
    else:
        # Global threshold for all data
        amp_flat = amp.flatten()
        med = np.median(amp_flat)
        mad_val = mad(amp_flat)
        sigma_robust = mad_to_sigma(mad_val)
        
        if sigma_robust > 0:
            threshold = sigma * sigma_robust
            flags |= np.abs(amp - med) > threshold
    
    n_flagged_after = flags.sum()
    n_total = flags.size
    
    stats = FlagStats(
        n_total=n_total,
        n_flagged=n_flagged_after,
        n_new_flags=n_flagged_after - n_flagged_before,
        fraction_flagged=n_flagged_after / n_total,
        threshold_used=sigma,
    )
    
    return flags, stats


def flag_rfi_channels(
    vis: np.ndarray,
    sigma: float = 5.0,
    existing_flags: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, FlagStats]:
    """
    Flag entire channels based on channel statistics.
    
    Useful for detecting narrow-band RFI.
    
    Parameters
    ----------
    vis : ndarray (n_bl, n_chan, 2, 2)
        Visibility data
    sigma : float
        Threshold in sigma
    existing_flags : ndarray, optional
        Existing flags
    
    Returns
    -------
    flags : ndarray
        Boolean flags
    stats : FlagStats
    """
    if vis.ndim != 4:
        raise ValueError("flag_rfi_channels requires (n_bl, n_chan, 2, 2) data")
    
    n_bl, n_chan = vis.shape[:2]
    
    # Compute channel statistics (mean amplitude per channel)
    amp = np.abs(vis)
    channel_amp = amp.mean(axis=(0, 2, 3))  # (n_chan,)
    
    # Robust statistics
    med = np.median(channel_amp)
    mad_val = mad(channel_amp)
    sigma_robust = mad_to_sigma(mad_val)
    
    # Flag bad channels
    bad_channels = np.zeros(n_chan, dtype=bool)
    if sigma_robust > 0:
        threshold = sigma * sigma_robust
        bad_channels = np.abs(channel_amp - med) > threshold
    
    # Expand to full shape
    flags = np.zeros(vis.shape, dtype=bool)
    if existing_flags is not None:
        flags |= existing_flags
    
    n_flagged_before = flags.sum()
    
    for ch in range(n_chan):
        if bad_channels[ch]:
            flags[:, ch, :, :] = True
    
    n_flagged_after = flags.sum()
    
    stats = FlagStats(
        n_total=flags.size,
        n_flagged=n_flagged_after,
        n_new_flags=n_flagged_after - n_flagged_before,
        fraction_flagged=n_flagged_after / flags.size,
        threshold_used=sigma,
    )
    
    return flags, stats


def flag_rfi_baselines(
    vis: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    sigma: float = 5.0,
    existing_flags: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, FlagStats]:
    """
    Flag entire baselines based on baseline statistics.
    
    Useful for detecting baseline-based problems.
    
    Parameters
    ----------
    vis : ndarray (n_bl, ..., 2, 2)
        Visibility data
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    sigma : float
        Threshold in sigma
    existing_flags : ndarray, optional
        Existing flags
    
    Returns
    -------
    flags : ndarray
        Boolean flags
    stats : FlagStats
    """
    n_bl = len(antenna1)
    
    # Compute baseline statistics
    amp = np.abs(vis)
    if amp.ndim == 3:
        baseline_amp = amp.mean(axis=(1, 2))  # (n_bl,)
    else:
        baseline_amp = amp.mean(axis=(1, 2, 3))  # (n_bl,)
    
    # Robust statistics
    med = np.median(baseline_amp)
    mad_val = mad(baseline_amp)
    sigma_robust = mad_to_sigma(mad_val)
    
    # Flag bad baselines
    bad_baselines = np.zeros(n_bl, dtype=bool)
    if sigma_robust > 0:
        threshold = sigma * sigma_robust
        bad_baselines = np.abs(baseline_amp - med) > threshold
    
    # Expand to full shape
    flags = np.zeros(vis.shape, dtype=bool)
    if existing_flags is not None:
        flags |= existing_flags
    
    n_flagged_before = flags.sum()
    
    for bl in range(n_bl):
        if bad_baselines[bl]:
            flags[bl] = True
    
    n_flagged_after = flags.sum()
    
    stats = FlagStats(
        n_total=flags.size,
        n_flagged=n_flagged_after,
        n_new_flags=n_flagged_after - n_flagged_before,
        fraction_flagged=n_flagged_after / flags.size,
        threshold_used=sigma,
    )
    
    return flags, stats


def flag_zeros(vis: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Flag zero or near-zero visibilities.
    
    Parameters
    ----------
    vis : ndarray
        Visibility data
    
    Returns
    -------
    flags : ndarray
        Boolean flags
    n_flagged : int
        Number of zeros flagged
    """
    flags = np.abs(vis) < 1e-10
    return flags, flags.sum()


def flag_nans(vis: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Flag NaN and Inf visibilities.
    
    Parameters
    ----------
    vis : ndarray
        Visibility data
    
    Returns
    -------
    flags : ndarray
        Boolean flags
    n_flagged : int
        Number flagged
    """
    flags = ~np.isfinite(vis)
    return flags, flags.sum()


def flag_rfi_iterative(
    vis: np.ndarray,
    sigma: float = 5.0,
    max_iter: int = 3,
    convergence_threshold: float = 0.001,
    existing_flags: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, FlagStats]:
    """
    Iteratively flag RFI until convergence.
    
    Repeats MAD-based flagging until no new flags are added
    or max iterations reached.
    
    Parameters
    ----------
    vis : ndarray
        Visibility data
    sigma : float
        Threshold in sigma
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Stop if fraction of new flags < this
    existing_flags : ndarray, optional
        Starting flags
    verbose : bool
        Print progress
    
    Returns
    -------
    flags : ndarray
        Final flags
    stats : FlagStats
        Final statistics
    """
    flags = existing_flags.copy() if existing_flags is not None else np.zeros(vis.shape, dtype=bool)
    
    # First flag zeros and NaNs
    zero_flags, n_zeros = flag_zeros(vis)
    nan_flags, n_nans = flag_nans(vis)
    flags |= zero_flags | nan_flags
    
    if verbose:
        print(f"[FLAG] Flagged {n_zeros} zeros, {n_nans} NaN/Inf")
    
    total_new = 0
    
    for iteration in range(max_iter):
        # Mask flagged data for statistics
        vis_masked = np.where(flags, np.nan, vis)
        
        # Run MAD flagging
        new_flags, stats = flag_rfi_mad(
            vis, sigma=sigma, existing_flags=flags
        )
        
        n_new = stats.n_new_flags
        total_new += n_new
        
        if verbose:
            print(f"[FLAG] Iteration {iteration + 1}: {n_new} new flags "
                  f"({stats.fraction_flagged*100:.2f}% total)")
        
        flags = new_flags
        
        # Check convergence
        if n_new == 0:
            if verbose:
                print(f"[FLAG] Converged after {iteration + 1} iterations")
            break
        
        frac_new = n_new / (flags.size - flags.sum() + n_new)
        if frac_new < convergence_threshold:
            if verbose:
                print(f"[FLAG] Converged (new flags < {convergence_threshold*100:.1f}%)")
            break
    
    final_stats = FlagStats(
        n_total=flags.size,
        n_flagged=flags.sum(),
        n_new_flags=total_new,
        fraction_flagged=flags.sum() / flags.size,
        threshold_used=sigma,
    )
    
    return flags, final_stats


def apply_flags(
    vis: np.ndarray,
    flags: np.ndarray,
    fill_value: complex = 0.0,
) -> np.ndarray:
    """
    Apply flags to visibility data.
    
    Parameters
    ----------
    vis : ndarray
        Visibility data
    flags : ndarray
        Boolean flags
    fill_value : complex
        Value to fill flagged data
    
    Returns
    -------
    vis_flagged : ndarray
        Visibility with flags applied
    """
    return np.where(flags, fill_value, vis)
