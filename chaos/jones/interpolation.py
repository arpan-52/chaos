"""
Jones Matrix Interpolation.

Interpolation of calibration solutions in time and frequency.
Handles amplitude and phase separately to avoid phase wrapping issues.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Literal


InterpolationMethod = Literal["nearest", "linear", "cubic"]


def interpolate_diagonal_jones(
    jones: np.ndarray,
    time_src: np.ndarray,
    freq_src: np.ndarray,
    time_dst: np.ndarray,
    freq_dst: np.ndarray,
    time_method: InterpolationMethod = "linear",
    freq_method: InterpolationMethod = "linear",
) -> np.ndarray:
    """
    Interpolate diagonal Jones matrices in time and frequency.
    
    Interpolates amplitude and phase separately to avoid phase wrapping.
    
    Parameters
    ----------
    jones : ndarray (n_time_src, n_freq_src, n_ant, 2, 2)
        Source Jones matrices (diagonal)
    time_src : ndarray (n_time_src,)
        Source timestamps
    freq_src : ndarray (n_freq_src,)
        Source frequencies
    time_dst : ndarray (n_time_dst,)
        Destination timestamps
    freq_dst : ndarray (n_freq_dst,)
        Destination frequencies
    time_method : str
        'nearest', 'linear', or 'cubic'
    freq_method : str
        'nearest', 'linear', or 'cubic'
    
    Returns
    -------
    jones_interp : ndarray (n_time_dst, n_freq_dst, n_ant, 2, 2)
    """
    n_time_src, n_freq_src, n_ant = jones.shape[:3]
    n_time_dst = len(time_dst)
    n_freq_dst = len(freq_dst)
    
    # Extract diagonal elements
    g_00 = jones[..., 0, 0]  # (n_time, n_freq, n_ant)
    g_11 = jones[..., 1, 1]
    
    # Interpolate each diagonal element
    g_00_interp = _interpolate_complex_2d(
        g_00, time_src, freq_src, time_dst, freq_dst, time_method, freq_method
    )
    g_11_interp = _interpolate_complex_2d(
        g_11, time_src, freq_src, time_dst, freq_dst, time_method, freq_method
    )
    
    # Build output Jones
    jones_interp = np.zeros((n_time_dst, n_freq_dst, n_ant, 2, 2), dtype=np.complex128)
    jones_interp[..., 0, 0] = g_00_interp
    jones_interp[..., 1, 1] = g_11_interp
    
    return jones_interp


def interpolate_full_jones(
    jones: np.ndarray,
    time_src: np.ndarray,
    freq_src: np.ndarray,
    time_dst: np.ndarray,
    freq_dst: np.ndarray,
    time_method: InterpolationMethod = "linear",
    freq_method: InterpolationMethod = "linear",
) -> np.ndarray:
    """
    Interpolate full 2x2 Jones matrices in time and frequency.
    
    Parameters
    ----------
    jones : ndarray (n_time_src, n_freq_src, n_ant, 2, 2)
        Source Jones matrices (full)
    time_src : ndarray (n_time_src,)
        Source timestamps
    freq_src : ndarray (n_freq_src,)
        Source frequencies
    time_dst : ndarray (n_time_dst,)
        Destination timestamps
    freq_dst : ndarray (n_freq_dst,)
        Destination frequencies
    time_method : str
        'nearest', 'linear', or 'cubic'
    freq_method : str
        'nearest', 'linear', or 'cubic'
    
    Returns
    -------
    jones_interp : ndarray (n_time_dst, n_freq_dst, n_ant, 2, 2)
    """
    n_time_src, n_freq_src, n_ant = jones.shape[:3]
    n_time_dst = len(time_dst)
    n_freq_dst = len(freq_dst)
    
    jones_interp = np.zeros((n_time_dst, n_freq_dst, n_ant, 2, 2), dtype=np.complex128)
    
    # Interpolate each element
    for i in range(2):
        for j in range(2):
            elem = jones[..., i, j]  # (n_time, n_freq, n_ant)
            elem_interp = _interpolate_complex_2d(
                elem, time_src, freq_src, time_dst, freq_dst, time_method, freq_method
            )
            jones_interp[..., i, j] = elem_interp
    
    return jones_interp


def _interpolate_complex_2d(
    data: np.ndarray,
    time_src: np.ndarray,
    freq_src: np.ndarray,
    time_dst: np.ndarray,
    freq_dst: np.ndarray,
    time_method: str,
    freq_method: str,
) -> np.ndarray:
    """
    Interpolate complex data in 2D (time and frequency).
    
    Interpolates amplitude and unwrapped phase separately.
    
    Parameters
    ----------
    data : ndarray (n_time_src, n_freq_src, n_ant)
        Complex data
    ...
    
    Returns
    -------
    data_interp : ndarray (n_time_dst, n_freq_dst, n_ant)
    """
    n_time_src, n_freq_src, n_ant = data.shape
    n_time_dst = len(time_dst)
    n_freq_dst = len(freq_dst)
    
    # Separate amplitude and phase
    amp = np.abs(data)
    phase = np.angle(data)
    
    # Unwrap phase along time axis to avoid discontinuities
    phase_unwrapped = np.unwrap(phase, axis=0)
    
    # Interpolate in time first
    amp_t = _interp_axis(amp, time_src, time_dst, time_method, axis=0)
    phase_t = _interp_axis(phase_unwrapped, time_src, time_dst, time_method, axis=0)
    
    # Then interpolate in frequency
    amp_tf = _interp_axis(amp_t, freq_src, freq_dst, freq_method, axis=1)
    phase_tf = _interp_axis(phase_t, freq_src, freq_dst, freq_method, axis=1)
    
    # Reconstruct complex
    return amp_tf * np.exp(1j * phase_tf)


def _interp_axis(
    data: np.ndarray,
    x_src: np.ndarray,
    x_dst: np.ndarray,
    method: str,
    axis: int,
) -> np.ndarray:
    """
    Interpolate along a single axis.
    
    Parameters
    ----------
    data : ndarray
        Data array
    x_src : ndarray
        Source coordinates
    x_dst : ndarray
        Destination coordinates
    method : str
        'nearest', 'linear', or 'cubic'
    axis : int
        Axis along which to interpolate
    
    Returns
    -------
    data_interp : ndarray
    """
    n_src = len(x_src)
    
    if n_src == 1:
        # Broadcast single value
        shape = list(data.shape)
        shape[axis] = len(x_dst)
        return np.broadcast_to(
            np.take(data, [0], axis=axis),
            shape
        ).copy()
    
    # scipy interp1d
    fn = interp1d(
        x_src, 
        data, 
        kind=method, 
        axis=axis,
        bounds_error=False, 
        fill_value="extrapolate"
    )
    return fn(x_dst)


def interpolate_time_only(
    jones: np.ndarray,
    time_src: np.ndarray,
    time_dst: np.ndarray,
    method: InterpolationMethod = "linear",
) -> np.ndarray:
    """
    Interpolate Jones matrices in time only (no frequency interpolation).
    
    Parameters
    ----------
    jones : ndarray (n_time_src, n_ant, 2, 2)
        Source Jones matrices
    time_src : ndarray (n_time_src,)
        Source timestamps
    time_dst : ndarray (n_time_dst,)
        Destination timestamps
    method : str
        'nearest', 'linear', or 'cubic'
    
    Returns
    -------
    jones_interp : ndarray (n_time_dst, n_ant, 2, 2)
    """
    n_time_src, n_ant = jones.shape[:2]
    n_time_dst = len(time_dst)
    
    jones_interp = np.zeros((n_time_dst, n_ant, 2, 2), dtype=np.complex128)
    
    for i in range(2):
        for j in range(2):
            elem = jones[..., i, j]  # (n_time, n_ant)
            
            amp = np.abs(elem)
            phase = np.unwrap(np.angle(elem), axis=0)
            
            amp_interp = _interp_axis(amp, time_src, time_dst, method, axis=0)
            phase_interp = _interp_axis(phase, time_src, time_dst, method, axis=0)
            
            jones_interp[..., i, j] = amp_interp * np.exp(1j * phase_interp)
    
    return jones_interp


def interpolate_freq_only(
    jones: np.ndarray,
    freq_src: np.ndarray,
    freq_dst: np.ndarray,
    method: InterpolationMethod = "linear",
) -> np.ndarray:
    """
    Interpolate Jones matrices in frequency only.
    
    Parameters
    ----------
    jones : ndarray (n_freq_src, n_ant, 2, 2)
        Source Jones matrices
    freq_src : ndarray (n_freq_src,)
        Source frequencies
    freq_dst : ndarray (n_freq_dst,)
        Destination frequencies
    method : str
        'nearest', 'linear', or 'cubic'
    
    Returns
    -------
    jones_interp : ndarray (n_freq_dst, n_ant, 2, 2)
    """
    n_freq_src, n_ant = jones.shape[:2]
    n_freq_dst = len(freq_dst)
    
    jones_interp = np.zeros((n_freq_dst, n_ant, 2, 2), dtype=np.complex128)
    
    for i in range(2):
        for j in range(2):
            elem = jones[..., i, j]  # (n_freq, n_ant)
            
            amp = np.abs(elem)
            phase = np.unwrap(np.angle(elem), axis=0)
            
            amp_interp = _interp_axis(amp, freq_src, freq_dst, method, axis=0)
            phase_interp = _interp_axis(phase, freq_src, freq_dst, method, axis=0)
            
            jones_interp[..., i, j] = amp_interp * np.exp(1j * phase_interp)
    
    return jones_interp
