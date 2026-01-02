"""
MeasurementSet Reader.

Read visibility data, model, flags, and metadata from CASA MeasurementSets.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VisibilityData:
    """Container for visibility data from MS."""
    vis_obs: np.ndarray      # (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
    vis_model: np.ndarray    # (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
    flags: np.ndarray        # (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
    weights: np.ndarray      # (n_bl, 2, 2) or (n_bl, n_chan, 2, 2)
    antenna1: np.ndarray     # (n_bl,)
    antenna2: np.ndarray     # (n_bl,)
    time: np.ndarray         # (n_bl,)
    freq: np.ndarray         # (n_chan,)
    n_ant: int
    feed_type: str           # 'linear' or 'circular'


class MSReader:
    """
    Read data from MeasurementSet.
    
    Handles conversion from MS correlation format to 2x2 Jones format.
    """
    
    def __init__(self, ms_path: str):
        """
        Initialize MS reader.
        
        Parameters
        ----------
        ms_path : str
            Path to MeasurementSet
        """
        self.ms_path = ms_path
        self._validate_ms()
    
    def _validate_ms(self):
        """Check MS exists and has required columns."""
        from casacore.tables import table
        
        with table(self.ms_path, ack=False) as tb:
            cols = tb.colnames()
            if "DATA" not in cols:
                raise ValueError(f"MS missing DATA column: {self.ms_path}")
    
    def get_antenna_info(self) -> Dict:
        """Get antenna information."""
        from casacore.tables import table
        
        with table(f"{self.ms_path}/ANTENNA", ack=False) as tb:
            names = list(tb.getcol("NAME"))
            positions = tb.getcol("POSITION")
            n_ant = len(names)
        
        return {
            "names": names,
            "positions": positions,
            "n_ant": n_ant,
        }
    
    def get_field_info(self) -> Dict:
        """Get field (source) information."""
        from casacore.tables import table
        
        with table(f"{self.ms_path}/FIELD", ack=False) as tb:
            names = list(tb.getcol("NAME"))
            phase_dirs = tb.getcol("PHASE_DIR")  # (n_field, n_poly, 2)
        
        return {
            "names": names,
            "phase_dirs": phase_dirs,
            "n_field": len(names),
        }
    
    def get_spw_info(self) -> Dict:
        """Get spectral window information."""
        from casacore.tables import table
        
        with table(f"{self.ms_path}/SPECTRAL_WINDOW", ack=False) as tb:
            chan_freqs = tb.getcol("CHAN_FREQ")  # (n_spw, n_chan)
            chan_widths = tb.getcol("CHAN_WIDTH")
            n_spw = chan_freqs.shape[0]
        
        return {
            "chan_freqs": chan_freqs,
            "chan_widths": chan_widths,
            "n_spw": n_spw,
        }
    
    def get_feed_type(self) -> str:
        """Detect feed type from POLARIZATION table."""
        from casacore.tables import table
        from chaos.jones.feed_type import detect_feed_type
        
        with table(f"{self.ms_path}/POLARIZATION", ack=False) as tb:
            corr_types = tb.getcol("CORR_TYPE")[0]
        
        return detect_feed_type(corr_types)
    
    def read_data(
        self,
        field_id: Optional[int] = None,
        spw: Optional[int] = None,
        scan: Optional[int] = None,
        data_column: str = "DATA",
        model_column: str = "MODEL_DATA",
        average_channels: bool = False,
        channel_slice: Optional[Tuple[int, int]] = None,
    ) -> VisibilityData:
        """
        Read visibility data from MS.
        
        Parameters
        ----------
        field_id : int, optional
            Select specific field
        spw : int, optional
            Select specific spectral window
        scan : int, optional
            Select specific scan
        data_column : str
            Column for observed data
        model_column : str
            Column for model data
        average_channels : bool
            If True, average all channels
        channel_slice : tuple (start, end), optional
            Channel range to select
        
        Returns
        -------
        data : VisibilityData
            Visibility data container
        """
        from casacore.tables import table, taql
        
        # Build selection query
        query_parts = []
        if field_id is not None:
            query_parts.append(f"FIELD_ID == {field_id}")
        if spw is not None:
            query_parts.append(f"DATA_DESC_ID == {spw}")
        if scan is not None:
            query_parts.append(f"SCAN_NUMBER == {scan}")
        
        if query_parts:
            query = f"SELECT * FROM {self.ms_path} WHERE " + " AND ".join(query_parts)
            tb = taql(query)
        else:
            tb = table(self.ms_path, ack=False)
        
        try:
            # Read columns
            antenna1 = tb.getcol("ANTENNA1")
            antenna2 = tb.getcol("ANTENNA2")
            time = tb.getcol("TIME")
            
            data_raw = tb.getcol(data_column)  # (n_row, n_chan, n_corr)
            
            # Check if model exists
            if model_column in tb.colnames():
                model_raw = tb.getcol(model_column)
            else:
                # Use unit model (point source at phase center)
                model_raw = np.ones_like(data_raw)
                model_raw[..., 1] = 0  # XY = 0
                model_raw[..., 2] = 0  # YX = 0
            
            # Flags
            if "FLAG" in tb.colnames():
                flags_raw = tb.getcol("FLAG")
            else:
                flags_raw = np.zeros_like(data_raw, dtype=bool)
            
            # Weights
            if "WEIGHT_SPECTRUM" in tb.colnames():
                weights_raw = tb.getcol("WEIGHT_SPECTRUM")
            elif "WEIGHT" in tb.colnames():
                weights = tb.getcol("WEIGHT")  # (n_row, n_corr)
                # Broadcast to (n_row, n_chan, n_corr)
                n_chan = data_raw.shape[1]
                weights_raw = np.broadcast_to(
                    weights[:, np.newaxis, :],
                    data_raw.shape
                ).copy()
            else:
                weights_raw = np.ones_like(data_raw, dtype=np.float64)
            
        finally:
            tb.close()
        
        # Get frequency info
        spw_info = self.get_spw_info()
        spw_idx = spw if spw is not None else 0
        freq = spw_info["chan_freqs"][spw_idx]
        
        # Apply channel selection
        if channel_slice is not None:
            start, end = channel_slice
            data_raw = data_raw[:, start:end]
            model_raw = model_raw[:, start:end]
            flags_raw = flags_raw[:, start:end]
            weights_raw = weights_raw[:, start:end]
            freq = freq[start:end]
        
        # Get number of antennas
        n_ant = self.get_antenna_info()["n_ant"]
        
        # Get feed type
        feed_type = self.get_feed_type()
        
        # Convert from MS format to 2x2 Jones format
        vis_obs = self._corr_to_jones(data_raw)
        vis_model = self._corr_to_jones(model_raw)
        flags = self._corr_to_jones_bool(flags_raw)
        weights = self._corr_to_jones_real(weights_raw)
        
        # Average channels if requested
        if average_channels:
            vis_obs = self._average_channels(vis_obs, flags, weights)
            vis_model = self._average_channels(vis_model, flags, weights)
            flags = flags.any(axis=1, keepdims=False)
            weights = weights.mean(axis=1)
            freq = np.array([freq.mean()])
        
        return VisibilityData(
            vis_obs=vis_obs,
            vis_model=vis_model,
            flags=flags,
            weights=weights,
            antenna1=antenna1,
            antenna2=antenna2,
            time=time,
            freq=freq,
            n_ant=n_ant,
            feed_type=feed_type,
        )
    
    def _corr_to_jones(self, data: np.ndarray) -> np.ndarray:
        """
        Convert MS correlation format to 2x2 Jones format.
        
        MS format: (n_row, n_chan, n_corr) where n_corr = 1, 2, or 4
        Jones format: (n_row, n_chan, 2, 2) or (n_row, 2, 2)
        
        Correlation order:
            n_corr = 4: [XX, XY, YX, YY] or [RR, RL, LR, LL]
            n_corr = 2: [XX, YY] or [RR, LL]
            n_corr = 1: [I]
        """
        n_row = data.shape[0]
        n_chan = data.shape[1]
        n_corr = data.shape[2]
        
        jones = np.zeros((n_row, n_chan, 2, 2), dtype=data.dtype)
        
        if n_corr == 4:
            jones[..., 0, 0] = data[..., 0]  # XX or RR
            jones[..., 0, 1] = data[..., 1]  # XY or RL
            jones[..., 1, 0] = data[..., 2]  # YX or LR
            jones[..., 1, 1] = data[..., 3]  # YY or LL
        elif n_corr == 2:
            jones[..., 0, 0] = data[..., 0]  # XX or RR
            jones[..., 1, 1] = data[..., 1]  # YY or LL
        elif n_corr == 1:
            jones[..., 0, 0] = data[..., 0]
            jones[..., 1, 1] = data[..., 0]
        else:
            raise ValueError(f"Unsupported n_corr: {n_corr}")
        
        return jones
    
    def _corr_to_jones_bool(self, flags: np.ndarray) -> np.ndarray:
        """Convert flags from MS format to Jones format."""
        return self._corr_to_jones(flags.astype(np.float64)).astype(bool)
    
    def _corr_to_jones_real(self, weights: np.ndarray) -> np.ndarray:
        """Convert weights from MS format to Jones format."""
        return np.abs(self._corr_to_jones(weights.astype(np.complex128)))
    
    def _average_channels(
        self,
        vis: np.ndarray,
        flags: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Average visibilities over channels with flagging/weighting.
        
        Parameters
        ----------
        vis : ndarray (n_row, n_chan, 2, 2)
        flags : ndarray (n_row, n_chan, 2, 2)
        weights : ndarray (n_row, n_chan, 2, 2)
        
        Returns
        -------
        vis_avg : ndarray (n_row, 2, 2)
        """
        # Mask flagged data
        vis_masked = np.where(flags, 0, vis)
        weights_masked = np.where(flags, 0, weights)
        
        # Weighted average
        sum_wv = (vis_masked * weights_masked).sum(axis=1)
        sum_w = weights_masked.sum(axis=1)
        
        # Avoid division by zero
        sum_w = np.where(sum_w == 0, 1, sum_w)
        
        return sum_wv / sum_w


class ChunkedMSReader:
    """
    Memory-efficient chunked MS reader.
    
    Reads data in time chunks for processing large datasets.
    """
    
    def __init__(
        self,
        ms_path: str,
        time_chunk: float = 300.0,  # seconds
        freq_chunk: Optional[int] = None,  # channels
    ):
        """
        Initialize chunked reader.
        
        Parameters
        ----------
        ms_path : str
            Path to MS
        time_chunk : float
            Time chunk size in seconds
        freq_chunk : int, optional
            Frequency chunk size in channels
        """
        self.ms_path = ms_path
        self.time_chunk = time_chunk
        self.freq_chunk = freq_chunk
        self.reader = MSReader(ms_path)
    
    def iter_chunks(
        self,
        field_id: Optional[int] = None,
        spw: Optional[int] = None,
        data_column: str = "DATA",
        model_column: str = "MODEL_DATA",
    ):
        """
        Iterate over time chunks.
        
        Yields
        ------
        chunk : VisibilityData
            Data for each time chunk
        """
        from casacore.tables import table
        
        # Get unique times
        with table(self.ms_path, ack=False) as tb:
            times = tb.getcol("TIME")
        
        unique_times = np.unique(times)
        
        # Group into chunks
        t_start = unique_times[0]
        chunk_times = []
        
        for t in unique_times:
            if t - t_start > self.time_chunk and chunk_times:
                # Process this chunk
                yield self._read_time_chunk(
                    chunk_times, field_id, spw, data_column, model_column
                )
                chunk_times = []
                t_start = t
            
            chunk_times.append(t)
        
        # Last chunk
        if chunk_times:
            yield self._read_time_chunk(
                chunk_times, field_id, spw, data_column, model_column
            )
    
    def _read_time_chunk(
        self,
        times: List[float],
        field_id: Optional[int],
        spw: Optional[int],
        data_column: str,
        model_column: str,
    ) -> VisibilityData:
        """Read a specific time chunk."""
        from casacore.tables import taql
        
        t_min, t_max = min(times), max(times)
        
        query_parts = [f"TIME >= {t_min}", f"TIME <= {t_max}"]
        if field_id is not None:
            query_parts.append(f"FIELD_ID == {field_id}")
        if spw is not None:
            query_parts.append(f"DATA_DESC_ID == {spw}")
        
        query = f"SELECT * FROM {self.ms_path} WHERE " + " AND ".join(query_parts)
        
        # Read using base reader with manual query
        return self.reader.read_data(
            field_id=field_id,
            spw=spw,
            data_column=data_column,
            model_column=model_column,
        )
