"""
HDF5 Calibration Table I/O.

Store and retrieve Jones solutions in HDF5 format.

File structure:
    cal_table.h5/
        {jones_type}/           # e.g., "K", "B", "G", "D"
            jones               # (n_time, n_freq, n_ant, 2, 2) complex
            time                # (n_time,) float64 - MJD seconds
            freq                # (n_freq,) float64 - Hz
            antenna             # (n_ant,) - antenna indices
            attrs:
                field           # string
                ref_antenna     # int
                mode            # 'phase_only', 'diagonal', 'full'
                solver          # 'chaos'
                created         # ISO timestamp
                metadata        # JSON string
"""

import numpy as np
import h5py
import json
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path


def save_jones_table(
    filepath: str,
    jones_type: str,
    jones: np.ndarray,
    time: np.ndarray,
    freq: np.ndarray,
    antenna: Optional[np.ndarray] = None,
    field: str = "",
    ref_antenna: int = 0,
    mode: str = "diagonal",
    metadata: Optional[Dict] = None,
    overwrite: bool = False,
) -> None:
    """
    Save Jones solutions to HDF5 table.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    jones_type : str
        Jones type: 'K', 'B', 'G', 'G[p]', 'D', 'X', 'KCROSS'
    jones : ndarray
        Jones matrices. Shape can be:
            (n_ant, 2, 2)                   - single time/freq
            (n_freq, n_ant, 2, 2)           - per channel
            (n_time, n_ant, 2, 2)           - per time
            (n_time, n_freq, n_ant, 2, 2)   - full grid
    time : ndarray
        Timestamps (MJD seconds)
    freq : ndarray
        Frequencies (Hz)
    antenna : ndarray, optional
        Antenna indices (default: 0, 1, 2, ...)
    field : str
        Field name
    ref_antenna : int
        Reference antenna
    mode : str
        'phase_only', 'diagonal', or 'full'
    metadata : dict, optional
        Additional metadata
    overwrite : bool
        If True, overwrite existing jones_type group
    """
    # Normalize jones shape to (n_time, n_freq, n_ant, 2, 2)
    jones = np.asarray(jones, dtype=np.complex128)
    time = np.atleast_1d(np.asarray(time, dtype=np.float64))
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    
    if jones.ndim == 3:
        # (n_ant, 2, 2) -> (1, 1, n_ant, 2, 2)
        jones = jones[np.newaxis, np.newaxis, ...]
    elif jones.ndim == 4:
        # Could be (n_freq, n_ant, 2, 2) or (n_time, n_ant, 2, 2)
        if len(freq) > 1 and jones.shape[0] == len(freq):
            # (n_freq, n_ant, 2, 2) -> (1, n_freq, n_ant, 2, 2)
            jones = jones[np.newaxis, ...]
        else:
            # (n_time, n_ant, 2, 2) -> (n_time, 1, n_ant, 2, 2)
            jones = jones[:, np.newaxis, ...]
    
    n_time, n_freq, n_ant = jones.shape[:3]
    
    if antenna is None:
        antenna = np.arange(n_ant, dtype=np.int32)
    
    # Ensure time/freq arrays match
    if len(time) == 1 and n_time > 1:
        time = np.full(n_time, time[0])
    if len(freq) == 1 and n_freq > 1:
        freq = np.full(n_freq, freq[0])
    
    # Open file
    file_mode = "a"  # Append mode
    
    with h5py.File(filepath, file_mode) as f:
        # Check if group exists
        if jones_type in f:
            if overwrite:
                del f[jones_type]
            else:
                raise ValueError(
                    f"Jones type '{jones_type}' already exists. "
                    f"Use overwrite=True to replace."
                )
        
        # Create group
        grp = f.create_group(jones_type)
        
        # Store data
        grp.create_dataset("jones", data=jones, compression="gzip")
        grp.create_dataset("time", data=time)
        grp.create_dataset("freq", data=freq)
        grp.create_dataset("antenna", data=antenna)
        
        # Attributes
        grp.attrs["field"] = field
        grp.attrs["ref_antenna"] = ref_antenna
        grp.attrs["mode"] = mode
        grp.attrs["solver"] = "chaos"
        grp.attrs["created"] = datetime.now().isoformat()
        
        if metadata:
            grp.attrs["metadata"] = json.dumps(metadata)


def load_jones_table(
    filepath: str,
    jones_type: str,
) -> Dict[str, Any]:
    """
    Load Jones solutions from HDF5 table.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    jones_type : str
        Jones type to load
    
    Returns
    -------
    data : dict
        Dictionary with keys:
            jones : ndarray (n_time, n_freq, n_ant, 2, 2)
            time : ndarray (n_time,)
            freq : ndarray (n_freq,)
            antenna : ndarray (n_ant,)
            field : str
            ref_antenna : int
            mode : str
            metadata : dict
    """
    with h5py.File(filepath, "r") as f:
        if jones_type not in f:
            raise KeyError(f"Jones type '{jones_type}' not found in {filepath}")
        
        grp = f[jones_type]
        
        data = {
            "jones": grp["jones"][...],
            "time": grp["time"][...],
            "freq": grp["freq"][...],
            "antenna": grp["antenna"][...],
            "field": grp.attrs.get("field", ""),
            "ref_antenna": grp.attrs.get("ref_antenna", 0),
            "mode": grp.attrs.get("mode", "diagonal"),
            "solver": grp.attrs.get("solver", "unknown"),
            "created": grp.attrs.get("created", ""),
        }
        
        # Parse metadata JSON
        if "metadata" in grp.attrs:
            data["metadata"] = json.loads(grp.attrs["metadata"])
        else:
            data["metadata"] = {}
    
    return data


def list_jones_terms(filepath: str) -> List[str]:
    """
    List Jones types in a calibration table.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    
    Returns
    -------
    terms : list of str
        List of Jones type names
    """
    if not Path(filepath).exists():
        return []
    
    with h5py.File(filepath, "r") as f:
        return list(f.keys())


def get_table_info(filepath: str) -> Dict[str, Dict]:
    """
    Get info about all Jones terms in a table.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file
    
    Returns
    -------
    info : dict
        Dict mapping jones_type to info dict
    """
    info = {}
    
    with h5py.File(filepath, "r") as f:
        for name in f.keys():
            grp = f[name]
            
            jones_shape = grp["jones"].shape
            
            info[name] = {
                "shape": jones_shape,
                "n_time": jones_shape[0],
                "n_freq": jones_shape[1],
                "n_ant": jones_shape[2],
                "field": grp.attrs.get("field", ""),
                "ref_antenna": grp.attrs.get("ref_antenna", 0),
                "mode": grp.attrs.get("mode", "diagonal"),
                "created": grp.attrs.get("created", ""),
            }
    
    return info


def merge_jones_files(
    input_files: List[str],
    output_file: str,
    overwrite: bool = False,
) -> None:
    """
    Merge multiple calibration tables into one.
    
    Parameters
    ----------
    input_files : list of str
        Input HDF5 files
    output_file : str
        Output merged file
    overwrite : bool
        Overwrite existing terms in output
    """
    for input_path in input_files:
        terms = list_jones_terms(input_path)
        
        for term in terms:
            data = load_jones_table(input_path, term)
            
            save_jones_table(
                output_file,
                term,
                data["jones"],
                data["time"],
                data["freq"],
                data["antenna"],
                data["field"],
                data["ref_antenna"],
                data["mode"],
                data["metadata"],
                overwrite=overwrite,
            )
