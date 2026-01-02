"""
HDF5 Calibration Table I/O.

Stores Jones matrices and metadata in HDF5 format.

Structure:
    cal_table.h5
    ├── K/
    │   ├── jones          (n_time, n_ant, 2, 2) complex128
    │   ├── time           (n_time,) float64 (MJD seconds)
    │   ├── freq           (n_freq,) float64 (Hz)
    │   ├── antenna        (n_ant,) int32
    │   ├── field          string
    │   └── attrs: {ref_ant, mode, solver, ...}
    ├── B/
    │   └── ...
    └── G/
        └── ...
"""

import numpy as np
import h5py
from datetime import datetime
from typing import Dict, Optional, Any, List


def save_jones_term(
    filename: str,
    term_name: str,
    jones: np.ndarray,
    time: np.ndarray,
    freq: np.ndarray,
    antenna: np.ndarray,
    field: str,
    ref_antenna: int = 0,
    mode: str = 'diagonal',
    solver: str = 'chaos',
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
):
    """
    Save a Jones term to HDF5 file.
    
    Parameters
    ----------
    filename : str
        Output HDF5 file path
    term_name : str
        Jones term name (K, B, G, P, D, X)
    jones : ndarray
        Jones matrices, shape (n_time, n_freq, n_ant, 2, 2) or (n_ant, 2, 2)
    time : ndarray
        Time stamps (MJD seconds)
    freq : ndarray
        Frequencies (Hz)
    antenna : ndarray
        Antenna indices
    field : str
        Field name(s)
    ref_antenna : int
        Reference antenna
    mode : str
        Calibration mode
    solver : str
        Solver name
    metadata : dict, optional
        Additional metadata
    overwrite : bool
        Overwrite existing term
    """
    # Ensure jones has correct shape
    jones = np.asarray(jones, dtype=np.complex128)
    time = np.asarray(time, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    antenna = np.asarray(antenna, dtype=np.int32)
    
    # Open file
    mode_flag = 'a'  # append
    with h5py.File(filename, mode_flag) as f:
        # Create or overwrite group
        if term_name in f:
            if overwrite:
                del f[term_name]
            else:
                raise ValueError(f"Term {term_name} already exists. Use overwrite=True.")
        
        grp = f.create_group(term_name)
        
        # Store data
        grp.create_dataset('jones', data=jones, compression='gzip')
        grp.create_dataset('time', data=time)
        grp.create_dataset('freq', data=freq)
        grp.create_dataset('antenna', data=antenna)
        grp.attrs['field'] = field
        grp.attrs['ref_antenna'] = ref_antenna
        grp.attrs['mode'] = mode
        grp.attrs['solver'] = solver
        grp.attrs['created'] = datetime.now().isoformat()
        
        if metadata:
            for key, value in metadata.items():
                try:
                    grp.attrs[key] = value
                except TypeError:
                    grp.attrs[key] = str(value)


def load_jones_term(
    filename: str,
    term_name: str
) -> Dict[str, Any]:
    """
    Load a Jones term from HDF5 file.
    
    Parameters
    ----------
    filename : str
        HDF5 file path
    term_name : str
        Jones term name
    
    Returns
    -------
    data : dict
        Dictionary with 'jones', 'time', 'freq', 'antenna', 'field', 
        'ref_antenna', 'mode', 'solver', and any additional metadata
    """
    with h5py.File(filename, 'r') as f:
        if term_name not in f:
            raise KeyError(f"Term {term_name} not found in {filename}")
        
        grp = f[term_name]
        
        data = {
            'jones': grp['jones'][...],
            'time': grp['time'][...],
            'freq': grp['freq'][...],
            'antenna': grp['antenna'][...],
            'field': grp.attrs['field'],
            'ref_antenna': grp.attrs['ref_antenna'],
            'mode': grp.attrs['mode'],
            'solver': grp.attrs['solver'],
        }
        
        # Load additional metadata
        for key in grp.attrs.keys():
            if key not in data:
                data[key] = grp.attrs[key]
        
        return data


def list_jones_terms(filename: str) -> List[str]:
    """
    List all Jones terms in an HDF5 file.
    
    Parameters
    ----------
    filename : str
        HDF5 file path
    
    Returns
    -------
    terms : list of str
        List of term names
    """
    with h5py.File(filename, 'r') as f:
        return list(f.keys())


def get_jones_info(filename: str) -> Dict[str, Dict[str, Any]]:
    """
    Get info about all Jones terms in file.
    
    Parameters
    ----------
    filename : str
        HDF5 file path
    
    Returns
    -------
    info : dict
        Dict of term name -> {shape, field, ref_antenna, mode, ...}
    """
    info = {}
    
    with h5py.File(filename, 'r') as f:
        for term in f.keys():
            grp = f[term]
            info[term] = {
                'shape': grp['jones'].shape,
                'n_time': len(grp['time']),
                'n_freq': len(grp['freq']),
                'n_ant': len(grp['antenna']),
                'field': grp.attrs['field'],
                'ref_antenna': grp.attrs['ref_antenna'],
                'mode': grp.attrs['mode'],
                'solver': grp.attrs['solver'],
            }
    
    return info


def merge_jones_files(
    output_file: str,
    input_files: List[str],
    overwrite: bool = False
):
    """
    Merge multiple HDF5 calibration files.
    
    Parameters
    ----------
    output_file : str
        Output file path
    input_files : list of str
        Input file paths
    overwrite : bool
        Overwrite existing terms
    """
    for input_file in input_files:
        terms = list_jones_terms(input_file)
        for term in terms:
            data = load_jones_term(input_file, term)
            save_jones_term(
                output_file,
                term,
                data['jones'],
                data['time'],
                data['freq'],
                data['antenna'],
                data['field'],
                data['ref_antenna'],
                data['mode'],
                data['solver'],
                metadata={k: v for k, v in data.items() 
                          if k not in ['jones', 'time', 'freq', 'antenna', 
                                       'field', 'ref_antenna', 'mode', 'solver']},
                overwrite=overwrite
            )


__all__ = [
    'save_jones_term',
    'load_jones_term', 
    'list_jones_terms',
    'get_jones_info',
    'merge_jones_files'
]
