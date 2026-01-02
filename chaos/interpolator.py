"""
Interpolation of Jones Calibration Solutions.

Handles interpolation in time and frequency, with proper handling
of the unitary ambiguity for full Jones matrices.

For diagonal Jones:
    - Simple linear/nearest interpolation of amplitude and phase separately
    
For full Jones:
    - Need to handle unitary ambiguity (Yatawatta 2013)
    - Use quotient manifold methods or phase alignment
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Literal


def interpolate_jones_diagonal(
    jones_src: np.ndarray,
    time_src: np.ndarray,
    freq_src: np.ndarray,
    time_dst: np.ndarray,
    freq_dst: np.ndarray,
    time_method: str = 'linear',
    freq_method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate diagonal Jones matrices in time and frequency.
    
    Interpolates amplitude and phase separately to avoid phase wrapping issues.
    
    Parameters
    ----------
    jones_src : ndarray (n_time_src, n_freq_src, n_ant, 2, 2)
        Source Jones matrices
    time_src : ndarray (n_time_src,)
        Source times
    freq_src : ndarray (n_freq_src,)
        Source frequencies
    time_dst : ndarray (n_time_dst,)
        Destination times
    freq_dst : ndarray (n_freq_dst,)
        Destination frequencies
    time_method : str
        'linear', 'nearest', 'cubic'
    freq_method : str
        'linear', 'nearest', 'cubic'
    
    Returns
    -------
    jones_dst : ndarray (n_time_dst, n_freq_dst, n_ant, 2, 2)
        Interpolated Jones matrices
    """
    n_time_src, n_freq_src, n_ant = jones_src.shape[:3]
    n_time_dst = len(time_dst)
    n_freq_dst = len(freq_dst)
    
    # Extract amplitudes and phases for diagonal elements
    g_xx = jones_src[..., 0, 0]
    g_yy = jones_src[..., 1, 1]
    
    amp_xx = np.abs(g_xx)
    amp_yy = np.abs(g_yy)
    phase_xx = np.unwrap(np.angle(g_xx), axis=0)  # Unwrap along time
    phase_yy = np.unwrap(np.angle(g_yy), axis=0)
    
    # Interpolate in time first
    def interp_time(arr, method):
        if n_time_src == 1:
            # Broadcast
            return np.broadcast_to(arr, (n_time_dst,) + arr.shape[1:]).copy()
        
        # Shape: (n_time, n_freq, n_ant)
        result = np.zeros((n_time_dst, n_freq_src, n_ant))
        for f in range(n_freq_src):
            for a in range(n_ant):
                fn = interp1d(time_src, arr[:, f, a], kind=method, 
                             bounds_error=False, fill_value='extrapolate')
                result[:, f, a] = fn(time_dst)
        return result
    
    amp_xx_t = interp_time(amp_xx, time_method)
    amp_yy_t = interp_time(amp_yy, time_method)
    phase_xx_t = interp_time(phase_xx, time_method)
    phase_yy_t = interp_time(phase_yy, time_method)
    
    # Now interpolate in frequency
    def interp_freq(arr, method):
        if n_freq_src == 1:
            return np.broadcast_to(arr[:, 0:1, :], (n_time_dst, n_freq_dst, n_ant)).copy()
        
        # Shape: (n_time, n_freq, n_ant)
        result = np.zeros((n_time_dst, n_freq_dst, n_ant))
        for t in range(n_time_dst):
            for a in range(n_ant):
                fn = interp1d(freq_src, arr[t, :, a], kind=method,
                             bounds_error=False, fill_value='extrapolate')
                result[t, :, a] = fn(freq_dst)
        return result
    
    amp_xx_tf = interp_freq(amp_xx_t, freq_method)
    amp_yy_tf = interp_freq(amp_yy_t, freq_method)
    phase_xx_tf = interp_freq(phase_xx_t, freq_method)
    phase_yy_tf = interp_freq(phase_yy_t, freq_method)
    
    # Reconstruct complex gains
    g_xx_dst = amp_xx_tf * np.exp(1j * phase_xx_tf)
    g_yy_dst = amp_yy_tf * np.exp(1j * phase_yy_tf)
    
    # Build output Jones
    jones_dst = np.zeros((n_time_dst, n_freq_dst, n_ant, 2, 2), dtype=complex)
    jones_dst[..., 0, 0] = g_xx_dst
    jones_dst[..., 1, 1] = g_yy_dst
    
    return jones_dst


def align_jones_phase(
    jones: np.ndarray,
    ref_antenna: int = 0
) -> np.ndarray:
    """
    Align Jones matrices to remove arbitrary phase.
    
    For full Jones, align such that reference antenna has 
    identity-like phase structure.
    
    Parameters
    ----------
    jones : ndarray (..., n_ant, 2, 2)
        Jones matrices
    ref_antenna : int
        Reference antenna index
    
    Returns
    -------
    jones_aligned : ndarray
        Phase-aligned Jones matrices
    """
    # Get reference antenna Jones
    J_ref = jones[..., ref_antenna, :, :]
    
    # Compute inverse phase rotation
    # For diagonal: just divide by phase
    # For full Jones: more complex
    
    # Simple approach: align XX element phase
    phase_ref = np.angle(J_ref[..., 0, 0])
    
    # Apply phase rotation to all antennas
    rotation = np.exp(-1j * phase_ref)
    jones_aligned = jones.copy()
    
    # Broadcast rotation
    for i in range(jones.shape[-3]):  # n_ant
        jones_aligned[..., i, :, :] *= rotation[..., np.newaxis, np.newaxis]
    
    return jones_aligned


def interpolate_jones_full(
    jones_src: np.ndarray,
    time_src: np.ndarray,
    freq_src: np.ndarray,
    time_dst: np.ndarray,
    freq_dst: np.ndarray,
    time_method: str = 'linear',
    freq_method: str = 'linear',
    ref_antenna: int = 0
) -> np.ndarray:
    """
    Interpolate full 2x2 Jones matrices with unitary ambiguity handling.
    
    Uses phase alignment before interpolation to handle the unitary ambiguity
    that differs between solution intervals.
    
    Parameters
    ----------
    jones_src : ndarray (n_time_src, n_freq_src, n_ant, 2, 2)
        Source Jones matrices  
    time_src : ndarray (n_time_src,)
        Source times
    freq_src : ndarray (n_freq_src,)
        Source frequencies
    time_dst : ndarray (n_time_dst,)
        Destination times
    freq_dst : ndarray (n_freq_dst,)
        Destination frequencies
    time_method : str
        'linear', 'nearest'
    freq_method : str
        'linear', 'nearest'
    ref_antenna : int
        Reference antenna for phase alignment
    
    Returns
    -------
    jones_dst : ndarray (n_time_dst, n_freq_dst, n_ant, 2, 2)
        Interpolated Jones matrices
    """
    n_time_src, n_freq_src, n_ant = jones_src.shape[:3]
    n_time_dst = len(time_dst)
    n_freq_dst = len(freq_dst)
    
    # Align phases at each source time/freq
    jones_aligned = np.zeros_like(jones_src)
    for t in range(n_time_src):
        for f in range(n_freq_src):
            jones_aligned[t, f] = align_jones_phase(
                jones_src[t, f], ref_antenna
            )
    
    # Now interpolate each element
    jones_dst = np.zeros((n_time_dst, n_freq_dst, n_ant, 2, 2), dtype=complex)
    
    for i in range(2):
        for j in range(2):
            # Get element
            elem = jones_aligned[..., i, j]  # (n_time, n_freq, n_ant)
            
            # Interpolate as diagonal (amp/phase separate)
            amp = np.abs(elem)
            phase = np.unwrap(np.angle(elem), axis=0)
            
            # Time interpolation
            amp_t = _interp_1d(amp, time_src, time_dst, time_method, axis=0)
            phase_t = _interp_1d(phase, time_src, time_dst, time_method, axis=0)
            
            # Freq interpolation
            amp_tf = _interp_1d(amp_t, freq_src, freq_dst, freq_method, axis=1)
            phase_tf = _interp_1d(phase_t, freq_src, freq_dst, freq_method, axis=1)
            
            jones_dst[..., i, j] = amp_tf * np.exp(1j * phase_tf)
    
    return jones_dst


def _interp_1d(
    data: np.ndarray,
    x_src: np.ndarray,
    x_dst: np.ndarray,
    method: str,
    axis: int
) -> np.ndarray:
    """Helper for 1D interpolation along an axis."""
    if len(x_src) == 1:
        # Broadcast
        shape = list(data.shape)
        shape[axis] = len(x_dst)
        return np.broadcast_to(
            np.take(data, [0], axis=axis),
            shape
        ).copy()
    
    fn = interp1d(x_src, data, kind=method, axis=axis,
                  bounds_error=False, fill_value='extrapolate')
    return fn(x_dst)


def interpolate_multi_field(
    jones_dict: dict,
    time_dst: np.ndarray,
    freq_dst: np.ndarray,
    time_method: str = 'linear',
    freq_method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate Jones from multiple fields (for 3C147:3C286 syntax).
    
    Parameters
    ----------
    jones_dict : dict
        {field_name: (jones, time, freq)} for each field
    time_dst : ndarray
        Destination times
    freq_dst : ndarray
        Destination frequencies
    time_method : str
        Time interpolation method
    freq_method : str
        Frequency interpolation method
    
    Returns
    -------
    jones_dst : ndarray
        Interpolated Jones at destination times/freqs
    """
    # Collect all source times and their Jones
    all_times = []
    all_jones = []
    all_freqs = []
    
    for field, (jones, time, freq) in jones_dict.items():
        for t_idx, t in enumerate(time):
            all_times.append(t)
            all_jones.append(jones[t_idx])
            all_freqs.append(freq)
    
    # Sort by time
    sort_idx = np.argsort(all_times)
    all_times = np.array(all_times)[sort_idx]
    all_jones = np.array([all_jones[i] for i in sort_idx])
    
    # Use first frequency array (assume consistent)
    freq_src = all_freqs[0]
    
    # Now interpolate
    return interpolate_jones_diagonal(
        all_jones, all_times, freq_src,
        time_dst, freq_dst,
        time_method, freq_method
    )


__all__ = [
    'interpolate_jones_diagonal',
    'interpolate_jones_full',
    'interpolate_multi_field',
    'align_jones_phase'
]
