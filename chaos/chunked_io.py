"""
Chunked Measurement Set I/O.

Memory-efficient reading and writing of MS data in chunks.
Supports solution intervals (solint) in time and frequency.
"""

import numpy as np
from typing import Iterator, Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class DataChunk:
    """A chunk of visibility data."""
    vis_obs: np.ndarray          # (n_bl, n_chan, 2, 2) complex
    vis_model: np.ndarray        # (n_bl, n_chan, 2, 2) complex
    flags: np.ndarray            # (n_bl, n_chan, 2, 2) bool
    weights: np.ndarray          # (n_bl, n_chan, 2, 2) float
    antenna1: np.ndarray         # (n_bl,) int
    antenna2: np.ndarray         # (n_bl,) int
    time: np.ndarray             # (n_bl,) float (MJD seconds)
    freq: np.ndarray             # (n_chan,) float (Hz)
    row_indices: np.ndarray      # Original row indices in MS
    time_bin: int                # Time bin index
    freq_bin: int                # Frequency bin index
    unique_time: float           # Representative time for this chunk
    
    @property
    def n_bl(self):
        return len(self.antenna1)
    
    @property
    def n_chan(self):
        return len(self.freq)
    
    @property
    def n_ant(self):
        return max(self.antenna1.max(), self.antenna2.max()) + 1


class ChunkedMSReader:
    """
    Memory-efficient MS reader with chunking.
    
    Yields data in time-frequency chunks suitable for calibration
    with solution intervals.
    """
    
    def __init__(
        self,
        ms_path: str,
        field_id: int = 0,
        spw: Optional[List[int]] = None,
        scans: Optional[List[int]] = None,
        data_column: str = 'DATA',
        model_column: str = 'MODEL_DATA',
        time_chunk_size: Optional[float] = None,  # seconds, None = all
        freq_chunk_size: Optional[int] = None     # channels, None = all
    ):
        """
        Initialize chunked reader.
        
        Parameters
        ----------
        ms_path : str
            Path to MeasurementSet
        field_id : int
            Field ID to select
        spw : list of int, optional
            SPWs to select (None = all)
        scans : list of int, optional
            Scans to select (None = all)
        data_column : str
            Column for observed data
        model_column : str
            Column for model data
        time_chunk_size : float, optional
            Time chunk size in seconds
        freq_chunk_size : int, optional
            Frequency chunk size in channels
        """
        self.ms_path = ms_path
        self.field_id = field_id
        self.spw = spw
        self.scans = scans
        self.data_column = data_column
        self.model_column = model_column
        self.time_chunk_size = time_chunk_size
        self.freq_chunk_size = freq_chunk_size
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load MS metadata."""
        from casacore.tables import table, taql
        
        # Get frequency info
        with table(f"{self.ms_path}/SPECTRAL_WINDOW", ack=False) as tb:
            self.all_freqs = tb.getcol('CHAN_FREQ')  # (n_spw, n_chan)
            self.chan_widths = tb.getcol('CHAN_WIDTH')
        
        # Get antenna info
        with table(f"{self.ms_path}/ANTENNA", ack=False) as tb:
            self.n_ant = tb.nrows()
            self.antenna_names = tb.getcol('NAME')
        
        # Get data description (SPW mapping)
        with table(f"{self.ms_path}/DATA_DESCRIPTION", ack=False) as tb:
            self.spw_ids = tb.getcol('SPECTRAL_WINDOW_ID')
        
        # Build selection query
        query_parts = [f"FIELD_ID == {self.field_id}"]
        if self.scans is not None:
            scan_str = ",".join(str(s) for s in self.scans)
            query_parts.append(f"SCAN_NUMBER IN [{scan_str}]")
        if self.spw is not None:
            spw_str = ",".join(str(s) for s in self.spw)
            # Need to map SPW to DATA_DESC_ID
            ddids = [i for i, s in enumerate(self.spw_ids) if s in self.spw]
            ddid_str = ",".join(str(d) for d in ddids)
            query_parts.append(f"DATA_DESC_ID IN [{ddid_str}]")
        
        self.query = " AND ".join(query_parts)
        
        # Get unique times for chunking
        with table(self.ms_path, ack=False) as tb:
            if self.query:
                tb_sel = taql(f"SELECT * FROM $tb WHERE {self.query}")
                self.unique_times = np.unique(tb_sel.getcol('TIME'))
                tb_sel.close()
            else:
                self.unique_times = np.unique(tb.getcol('TIME'))
        
        # Determine chunk boundaries
        self._compute_chunks()
    
    def _compute_chunks(self):
        """Compute time and frequency chunk boundaries."""
        # Time chunks
        if self.time_chunk_size is None:
            self.time_bins = [(0, len(self.unique_times))]
        else:
            self.time_bins = []
            t_start = 0
            t0 = self.unique_times[0]
            
            for i, t in enumerate(self.unique_times):
                if t - t0 > self.time_chunk_size:
                    self.time_bins.append((t_start, i))
                    t_start = i
                    t0 = t
            
            self.time_bins.append((t_start, len(self.unique_times)))
        
        # Frequency chunks (per SPW)
        self.freq_bins = {}
        for spw_id in (self.spw if self.spw else range(len(self.all_freqs))):
            n_chan = len(self.all_freqs[spw_id])
            if self.freq_chunk_size is None or self.freq_chunk_size >= n_chan:
                self.freq_bins[spw_id] = [(0, n_chan)]
            else:
                bins = []
                for start in range(0, n_chan, self.freq_chunk_size):
                    end = min(start + self.freq_chunk_size, n_chan)
                    bins.append((start, end))
                self.freq_bins[spw_id] = bins
    
    def __iter__(self) -> Iterator[DataChunk]:
        """Iterate over chunks."""
        from casacore.tables import table, taql
        
        with table(self.ms_path, ack=False) as tb:
            # Apply selection
            if self.query:
                tb_sel = taql(f"SELECT * FROM $tb WHERE {self.query}")
            else:
                tb_sel = tb
            
            # Get all times and sort
            all_times = tb_sel.getcol('TIME')
            sort_idx = np.argsort(all_times)
            
            for t_bin, (t_start, t_end) in enumerate(self.time_bins):
                # Time range for this bin
                time_min = self.unique_times[t_start]
                time_max = self.unique_times[min(t_end, len(self.unique_times)-1)]
                
                # Find rows in this time range
                time_mask = (all_times >= time_min) & (all_times <= time_max)
                row_idx = np.where(time_mask)[0]
                
                if len(row_idx) == 0:
                    continue
                
                # Read data for these rows
                vis_obs = tb_sel.getcol(self.data_column, row_idx[0], len(row_idx))
                vis_model = tb_sel.getcol(self.model_column, row_idx[0], len(row_idx))
                flags = tb_sel.getcol('FLAG', row_idx[0], len(row_idx))
                
                # Try to get weights
                try:
                    weights = tb_sel.getcol('WEIGHT_SPECTRUM', row_idx[0], len(row_idx))
                except:
                    weights = np.ones_like(vis_obs, dtype=float)
                
                antenna1 = tb_sel.getcol('ANTENNA1', row_idx[0], len(row_idx))
                antenna2 = tb_sel.getcol('ANTENNA2', row_idx[0], len(row_idx))
                times = tb_sel.getcol('TIME', row_idx[0], len(row_idx))
                ddids = tb_sel.getcol('DATA_DESC_ID', row_idx[0], len(row_idx))
                
                # For each SPW and freq chunk
                for spw_id, f_bins in self.freq_bins.items():
                    # Find rows for this SPW
                    spw_mask = np.array([self.spw_ids[d] == spw_id for d in ddids])
                    if not spw_mask.any():
                        continue
                    
                    spw_idx = np.where(spw_mask)[0]
                    freqs = self.all_freqs[spw_id]
                    
                    for f_bin, (f_start, f_end) in enumerate(f_bins):
                        # Extract chunk
                        chunk_vis_obs = vis_obs[spw_idx, f_start:f_end]
                        chunk_vis_model = vis_model[spw_idx, f_start:f_end]
                        chunk_flags = flags[spw_idx, f_start:f_end]
                        chunk_weights = weights[spw_idx, f_start:f_end]
                        chunk_freqs = freqs[f_start:f_end]
                        
                        # Reshape to (n_bl, n_chan, 2, 2)
                        # MS format is typically (n_row, n_chan, n_corr)
                        # Convert to 2x2 Jones format
                        chunk_vis_obs = self._to_jones_format(chunk_vis_obs)
                        chunk_vis_model = self._to_jones_format(chunk_vis_model)
                        chunk_flags = self._to_jones_format(chunk_flags)
                        chunk_weights = self._to_jones_format(chunk_weights)
                        
                        yield DataChunk(
                            vis_obs=chunk_vis_obs,
                            vis_model=chunk_vis_model,
                            flags=chunk_flags,
                            weights=chunk_weights,
                            antenna1=antenna1[spw_idx],
                            antenna2=antenna2[spw_idx],
                            time=times[spw_idx],
                            freq=chunk_freqs,
                            row_indices=row_idx[spw_idx],
                            time_bin=t_bin,
                            freq_bin=f_bin,
                            unique_time=np.median(times[spw_idx])
                        )
            
            if self.query:
                tb_sel.close()
    
    def _to_jones_format(self, data):
        """Convert from MS correlation format to 2x2 Jones."""
        # Assume 4 correlations: XX, XY, YX, YY (or RR, RL, LR, LL)
        if data.ndim == 2:
            # (n_row, n_corr) -> (n_row, 2, 2)
            n_row, n_corr = data.shape
            if n_corr == 4:
                out = np.zeros((n_row, 2, 2), dtype=data.dtype)
                out[:, 0, 0] = data[:, 0]
                out[:, 0, 1] = data[:, 1]
                out[:, 1, 0] = data[:, 2]
                out[:, 1, 1] = data[:, 3]
                return out
            elif n_corr == 2:
                out = np.zeros((n_row, 2, 2), dtype=data.dtype)
                out[:, 0, 0] = data[:, 0]
                out[:, 1, 1] = data[:, 1]
                return out
        elif data.ndim == 3:
            # (n_row, n_chan, n_corr) -> (n_row, n_chan, 2, 2)
            n_row, n_chan, n_corr = data.shape
            if n_corr == 4:
                out = np.zeros((n_row, n_chan, 2, 2), dtype=data.dtype)
                out[:, :, 0, 0] = data[:, :, 0]
                out[:, :, 0, 1] = data[:, :, 1]
                out[:, :, 1, 0] = data[:, :, 2]
                out[:, :, 1, 1] = data[:, :, 3]
                return out
            elif n_corr == 2:
                out = np.zeros((n_row, n_chan, 2, 2), dtype=data.dtype)
                out[:, :, 0, 0] = data[:, :, 0]
                out[:, :, 1, 1] = data[:, :, 1]
                return out
        
        return data


class ChunkedMSWriter:
    """
    Memory-efficient MS writer for corrected data.
    """
    
    def __init__(
        self,
        ms_path: str,
        output_column: str = 'CORRECTED_DATA'
    ):
        self.ms_path = ms_path
        self.output_column = output_column
        self._ensure_column_exists()
    
    def _ensure_column_exists(self):
        """Create output column if it doesn't exist."""
        from casacore.tables import table
        
        with table(self.ms_path, readonly=False, ack=False) as tb:
            if self.output_column not in tb.colnames():
                # Create column with same structure as DATA
                desc = tb.getcoldesc('DATA')
                tb.addcols({self.output_column: desc})
    
    def write_chunk(self, chunk: DataChunk, corrected_data: np.ndarray):
        """
        Write corrected data for a chunk.
        
        Parameters
        ----------
        chunk : DataChunk
            Original chunk with row_indices
        corrected_data : ndarray (n_bl, n_chan, 2, 2)
            Corrected visibilities
        """
        from casacore.tables import table
        
        # Convert from 2x2 to MS format
        data_ms = self._from_jones_format(corrected_data)
        
        with table(self.ms_path, readonly=False, ack=False) as tb:
            # Write row by row (handles non-contiguous rows)
            for i, row_idx in enumerate(chunk.row_indices):
                tb.putcell(self.output_column, row_idx, data_ms[i])
    
    def _from_jones_format(self, data):
        """Convert from 2x2 Jones to MS correlation format."""
        # (n_row, n_chan, 2, 2) -> (n_row, n_chan, 4)
        n_row, n_chan = data.shape[:2]
        out = np.zeros((n_row, n_chan, 4), dtype=data.dtype)
        out[:, :, 0] = data[:, :, 0, 0]
        out[:, :, 1] = data[:, :, 0, 1]
        out[:, :, 2] = data[:, :, 1, 0]
        out[:, :, 3] = data[:, :, 1, 1]
        return out


__all__ = [
    'DataChunk',
    'ChunkedMSReader',
    'ChunkedMSWriter'
]
