"""
Apply Calibration Solutions.

Apply Jones solutions to MeasurementSet, creating CORRECTED_DATA column.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

from chaos.io.table_io import load_jones_table, list_jones_terms
from chaos.jones.operations import unapply_jones, composite_jones
from chaos.jones.terms import I_jones
from chaos.jones.interpolation import interpolate_time_only, interpolate_freq_only


def applycal(
    ms_path: str,
    cal_tables: Union[str, List[str]],
    jones_types: Optional[List[str]] = None,
    data_column: str = "DATA",
    output_column: str = "CORRECTED_DATA",
    field_id: Optional[int] = None,
    spw: Optional[int] = None,
    time_interp: str = "linear",
    freq_interp: str = "linear",
    verbose: bool = True,
) -> None:
    """
    Apply calibration solutions to MS.
    
    Corrects visibilities: V_corr = J_i^{-1} @ V_obs @ (J_j^{-1})^H
    
    Parameters
    ----------
    ms_path : str
        Path to MeasurementSet
    cal_tables : str or list of str
        Path(s) to HDF5 calibration table(s)
    jones_types : list of str, optional
        Which Jones types to apply. If None, apply all.
    data_column : str
        Input data column
    output_column : str
        Output corrected data column
    field_id : int, optional
        Apply only to specific field
    spw : int, optional
        Apply only to specific SPW
    time_interp : str
        Time interpolation method: 'nearest', 'linear', 'cubic'
    freq_interp : str
        Frequency interpolation method: 'nearest', 'linear', 'cubic'
    verbose : bool
        Print progress
    """
    from casacore.tables import table
    
    if isinstance(cal_tables, str):
        cal_tables = [cal_tables]
    
    if verbose:
        print(f"[CHAOS] Applying calibration to {ms_path}")
        print(f"[CHAOS] Calibration tables: {cal_tables}")
    
    # Load all Jones solutions
    all_jones = {}
    for cal_table in cal_tables:
        terms = list_jones_terms(cal_table)
        
        for term in terms:
            if jones_types is not None and term not in jones_types:
                continue
            
            data = load_jones_table(cal_table, term)
            all_jones[term] = data
            
            if verbose:
                print(f"[CHAOS] Loaded {term}: shape {data['jones'].shape}")
    
    if not all_jones:
        print("[CHAOS] Warning: No Jones terms to apply")
        return
    
    # Open MS
    with table(ms_path, readonly=False, ack=False) as tb:
        # Get data info
        antenna1 = tb.getcol("ANTENNA1")
        antenna2 = tb.getcol("ANTENNA2")
        time = tb.getcol("TIME")
        n_row = len(antenna1)
        
        # Determine n_ant
        n_ant = max(antenna1.max(), antenna2.max()) + 1
        
        # Get data
        vis_obs = tb.getcol(data_column)  # (n_row, n_chan, n_corr)
        n_chan = vis_obs.shape[1]
        n_corr = vis_obs.shape[2]
        
        # Get frequencies
        with table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as spw_tb:
            spw_idx = spw if spw is not None else 0
            freq = spw_tb.getcol("CHAN_FREQ")[spw_idx]
        
        if verbose:
            print(f"[CHAOS] Data: {n_row} rows, {n_chan} channels, {n_ant} antennas")
        
        # Build composite Jones per antenna
        # For simplicity, compute at MS time/freq grid
        unique_times = np.unique(time)
        
        if verbose:
            print(f"[CHAOS] Building composite Jones matrices...")
        
        # Build J_total for each antenna at each time
        # Shape: (n_time, n_ant, 2, 2)
        jones_composite = _build_composite_jones(
            all_jones, unique_times, freq, n_ant,
            time_interp, freq_interp, verbose
        )
        
        # Create time index mapping
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        # Apply corrections
        if verbose:
            print(f"[CHAOS] Applying corrections...")
        
        vis_corr = np.zeros_like(vis_obs)
        
        for row in range(n_row):
            if row % 10000 == 0 and verbose:
                print(f"[CHAOS] Processing row {row}/{n_row}")
            
            a1 = antenna1[row]
            a2 = antenna2[row]
            t_idx = time_to_idx[time[row]]
            
            V_obs = _corr_to_jones_row(vis_obs[row])  # (n_chan, 2, 2)
            
            # Get Jones for this baseline
            # Average over frequency if jones is per-channel
            if jones_composite.ndim == 4:
                # (n_time, n_ant, 2, 2)
                J_i = jones_composite[t_idx, a1]  # (2, 2)
                J_j = jones_composite[t_idx, a2]
                
                # Apply to all channels
                for ch in range(n_chan):
                    V_corr = unapply_jones(V_obs[ch], J_i, J_j)
                    vis_corr[row, ch] = _jones_to_corr(V_corr, n_corr)
            else:
                # (n_time, n_freq, n_ant, 2, 2)
                for ch in range(n_chan):
                    J_i = jones_composite[t_idx, ch, a1]
                    J_j = jones_composite[t_idx, ch, a2]
                    
                    V_corr = unapply_jones(V_obs[ch], J_i, J_j)
                    vis_corr[row, ch] = _jones_to_corr(V_corr, n_corr)
        
        # Write to output column
        if output_column not in tb.colnames():
            desc = tb.getcoldesc(data_column)
            tb.addcols({output_column: desc})
            if verbose:
                print(f"[CHAOS] Created column {output_column}")
        
        tb.putcol(output_column, vis_corr)
        
        if verbose:
            print(f"[CHAOS] Written corrected data to {output_column}")


def _build_composite_jones(
    all_jones: Dict[str, Dict],
    times: np.ndarray,
    freqs: np.ndarray,
    n_ant: int,
    time_interp: str,
    freq_interp: str,
    verbose: bool,
) -> np.ndarray:
    """
    Build composite Jones matrix from all terms.
    
    J_total = J_N @ J_{N-1} @ ... @ J_1
    
    Returns shape (n_time, n_ant, 2, 2) or (n_time, n_freq, n_ant, 2, 2)
    """
    n_time = len(times)
    n_freq = len(freqs)
    
    # Determine if we need per-channel solutions
    need_freq_axis = any(
        d["jones"].shape[1] > 1 for d in all_jones.values()
    )
    
    if need_freq_axis:
        J_total = I_jones((n_time, n_freq, n_ant))
    else:
        J_total = I_jones((n_time, n_ant))
    
    # Apply each Jones term (multiply in order)
    for jones_type, data in all_jones.items():
        jones = data["jones"]  # (n_t_src, n_f_src, n_ant, 2, 2)
        time_src = data["time"]
        freq_src = data["freq"]
        
        if verbose:
            print(f"[CHAOS] Applying {jones_type}...")
        
        # Interpolate to target grid
        if jones.shape[0] == 1 and jones.shape[1] == 1:
            # Single solution - broadcast
            jones_interp = np.broadcast_to(
                jones[0, 0],
                (n_time, n_ant, 2, 2) if not need_freq_axis else (n_time, n_freq, n_ant, 2, 2)
            ).copy()
        elif jones.shape[1] == 1:
            # Time-only solutions
            jones_t = jones[:, 0]  # (n_t_src, n_ant, 2, 2)
            jones_interp = interpolate_time_only(
                jones_t, time_src, times, method=time_interp
            )
            if need_freq_axis:
                jones_interp = np.broadcast_to(
                    jones_interp[:, np.newaxis],
                    (n_time, n_freq, n_ant, 2, 2)
                ).copy()
        elif jones.shape[0] == 1:
            # Freq-only solutions  
            jones_f = jones[0]  # (n_f_src, n_ant, 2, 2)
            jones_interp = interpolate_freq_only(
                jones_f, freq_src, freqs, method=freq_interp
            )
            # Broadcast to time
            jones_interp = np.broadcast_to(
                jones_interp[np.newaxis],
                (n_time, n_freq, n_ant, 2, 2)
            ).copy()
        else:
            # Full time-freq grid - need 2D interpolation
            # For now, use nearest in time, linear in freq
            from chaos.jones.interpolation import interpolate_diagonal_jones
            jones_interp = interpolate_diagonal_jones(
                jones, time_src, freq_src, times, freqs,
                time_method=time_interp, freq_method=freq_interp
            )
        
        # Multiply: J_total = jones_interp @ J_total
        if need_freq_axis:
            for t in range(n_time):
                for f in range(n_freq):
                    for ant in range(n_ant):
                        J_total[t, f, ant] = jones_interp[t, f, ant] @ J_total[t, f, ant]
        else:
            for t in range(n_time):
                for ant in range(n_ant):
                    if jones_interp.ndim == 3:
                        J_total[t, ant] = jones_interp[t, ant] @ J_total[t, ant]
                    else:
                        J_total[t, ant] = jones_interp[t, 0, ant] @ J_total[t, ant]
    
    return J_total


def _corr_to_jones_row(data: np.ndarray) -> np.ndarray:
    """
    Convert single row from MS format to Jones format.
    
    Input: (n_chan, n_corr)
    Output: (n_chan, 2, 2)
    """
    n_chan, n_corr = data.shape
    jones = np.zeros((n_chan, 2, 2), dtype=data.dtype)
    
    if n_corr == 4:
        jones[..., 0, 0] = data[..., 0]
        jones[..., 0, 1] = data[..., 1]
        jones[..., 1, 0] = data[..., 2]
        jones[..., 1, 1] = data[..., 3]
    elif n_corr == 2:
        jones[..., 0, 0] = data[..., 0]
        jones[..., 1, 1] = data[..., 1]
    elif n_corr == 1:
        jones[..., 0, 0] = data[..., 0]
        jones[..., 1, 1] = data[..., 0]
    
    return jones


def _jones_to_corr(jones: np.ndarray, n_corr: int) -> np.ndarray:
    """
    Convert Jones format back to MS correlation format.
    
    Input: (2, 2)
    Output: (n_corr,)
    """
    if n_corr == 4:
        return np.array([
            jones[0, 0],
            jones[0, 1],
            jones[1, 0],
            jones[1, 1],
        ])
    elif n_corr == 2:
        return np.array([jones[0, 0], jones[1, 1]])
    else:
        return np.array([jones[0, 0]])
