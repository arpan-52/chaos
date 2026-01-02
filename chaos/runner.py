"""
CHAOS Calibration Runner.

Orchestrates the solve and apply pipeline based on configuration.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config_parser import (
    CalConfig, SolveEntry, ApplyEntry,
    load_config, parse_scan_range, parse_spw_range,
    parse_time_interval, parse_freq_interval
)
from .table_io import save_jones_term, load_jones_term, list_jones_terms
from .jones_terms import (
    P_jones_linear, P_jones_circular,
    composite_jones, apply_jones, unapply_jones,
    detect_feed_type, I_jones
)
from .parallactic import compute_parallactic_angles_from_ms, get_mount_type
from .interpolator import interpolate_jones_diagonal, interpolate_multi_field
from .ms_loader import MSLoader
from .quality import compute_quality_matrix
from .weighted_combiner import multi_ref_solve_and_combine
from .polish import polish_jones


class CalibrationRunner:
    """
    Runs calibration pipeline from configuration.
    """
    
    def __init__(
        self,
        config: CalConfig,
        ref_antenna: int = 0,
        solver: str = 'single_chain',
        single_ao: bool = False,
        polish: bool = True,
        polish_tol: float = 1e-10,
        verbose: bool = True
    ):
        """
        Initialize calibration runner.
        
        Parameters
        ----------
        config : CalConfig
            Parsed configuration
        ref_antenna : int
            Reference antenna
        solver : str
            'single_chain' or 'ratio_chain'
        single_ao : bool
            Single reference mode
        polish : bool
            Enable least squares polish
        polish_tol : float
            Polish tolerance
        verbose : bool
            Print progress
        """
        self.config = config
        self.ref_antenna = ref_antenna
        self.solver = solver
        self.single_ao = single_ao
        self.polish = polish
        self.polish_tol = polish_tol
        self.verbose = verbose
    
    def _print(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[CHAOS] {msg}")
    
    def run_solve(self):
        """Run all solve entries."""
        for entry in self.config.solve_entries:
            self._solve_entry(entry)
    
    def run_apply(self):
        """Run all apply entries."""
        for entry in self.config.apply_entries:
            self._apply_entry(entry)
    
    def run_all(self):
        """Run solve then apply."""
        self.run_solve()
        self.run_apply()
    
    def _solve_entry(self, entry: SolveEntry):
        """
        Process a single solve_jones entry.
        
        For entry like:
            jones_types: [K, B, G]
            fields: [3C147, 3C147, 3C147:3C286]
        
        We solve K first, then B (with K pre-applied), then G (with K,B pre-applied).
        """
        self._print(f"\n{'='*60}")
        self._print(f"SOLVE: {','.join(entry.jones_types)}")
        self._print(f"Fields: {','.join(entry.fields)}")
        self._print(f"Output: {entry.output_table}")
        self._print(f"{'='*60}")
        
        # Determine MS from first field
        ms_path = self._find_ms_for_field(entry.fields[0].split(':')[0])
        
        # Get feed type
        feed_type = self._get_feed_type(ms_path)
        self._print(f"Feed type: {feed_type}")
        
        # Pre-load any pre-apply tables
        pre_jones = {}
        if entry.pre_apply_jones and entry.pre_apply_tables:
            for term in entry.pre_apply_jones:
                for table in entry.pre_apply_tables:
                    if Path(table).exists() and term in list_jones_terms(table):
                        data = load_jones_term(table, term)
                        pre_jones[term] = data
                        self._print(f"Loaded pre-apply {term} from {table}")
                        break
        
        # Solve each Jones term in sequence
        solved_in_this_entry = {}
        
        for i, (jones_type, field_spec) in enumerate(zip(entry.jones_types, entry.fields)):
            self._print(f"\n--- Solving {jones_type} on {field_spec} ---")
            
            # Combine pre-apply terms
            pre_terms = {**pre_jones, **solved_in_this_entry}
            
            # Parse field (handle 3C147:3C286 syntax)
            fields = field_spec.split(':')
            
            # Solve
            jones, time, freq, info = self._solve_jones_term(
                ms_path=ms_path,
                jones_type=jones_type,
                fields=fields,
                scans=entry.scans,
                spw=entry.spw,
                freq_interval=entry.freq_interval,
                time_interval=entry.time_interval,
                model_column=entry.model_column,
                parallactic=entry.parallactic,
                feed_type=feed_type,
                pre_apply=pre_terms
            )
            
            # Save to output table
            save_jones_term(
                entry.output_table,
                jones_type,
                jones,
                time,
                freq,
                np.arange(jones.shape[-3]),  # antenna indices
                field_spec,
                ref_antenna=self.ref_antenna,
                mode='diagonal',
                solver=self.solver,
                metadata=info,
                overwrite=True
            )
            
            # Add to solved for this entry (for subsequent terms)
            solved_in_this_entry[jones_type] = {
                'jones': jones,
                'time': time,
                'freq': freq,
            }
            
            self._print(f"Saved {jones_type} to {entry.output_table}")
    
    def _solve_jones_term(
        self,
        ms_path: str,
        jones_type: str,
        fields: List[str],
        scans: str,
        spw: str,
        freq_interval: str,
        time_interval: str,
        model_column: str,
        parallactic: bool,
        feed_type: str,
        pre_apply: Dict[str, Any]
    ) -> tuple:
        """
        Solve for a single Jones term using CHAOS.
        
        Returns
        -------
        jones : ndarray
        time : ndarray
        freq : ndarray
        info : dict
        """
        # Parse intervals
        scan_list = parse_scan_range(scans)
        spw_list = parse_spw_range(spw)
        time_int = parse_time_interval(time_interval)
        freq_int = parse_freq_interval(freq_interval)
        
        # Load data
        loader = MSLoader(ms_path)
        
        # For each field, solve and combine
        all_jones = []
        all_times = []
        all_freqs = []
        
        for field_name in fields:
            self._print(f"  Processing field: {field_name}")
            
            # Get field ID from name
            field_id = self._get_field_id(ms_path, field_name)
            
            data = loader.load_data(
                field_id=field_id,
                spw=spw_list[0] if spw_list else 0,
                model_column=model_column
            )
            
            vis_obs = data['vis_obs']
            vis_model = data['vis_model']
            flags = data['flags']
            antenna1 = data['antenna1']
            antenna2 = data['antenna2']
            n_ant = data['n_ant']
            
            # Apply pre-solved terms
            if pre_apply:
                self._print(f"  Applying pre-solved: {list(pre_apply.keys())}")
                vis_obs = self._apply_pre_jones(
                    vis_obs, antenna1, antenna2, pre_apply, n_ant
                )
            
            # Apply parallactic angle correction if needed
            if parallactic:
                self._print(f"  Removing parallactic angle")
                times_pa, psi = compute_parallactic_angles_from_ms(ms_path, field_id)
                # Use mean parallactic angle for now
                psi_mean = psi.mean(axis=0)  # (n_ant,)
                vis_obs = self._remove_parallactic(
                    vis_obs, psi_mean, antenna1, antenna2, feed_type
                )
            
            # Compute quality matrix
            quality_matrix = compute_quality_matrix(
                vis_obs, flags, antenna1, antenna2, n_ant
            )
            
            # Detect bad antennas
            bad_antennas = self._detect_bad_antennas(
                vis_obs, antenna1, antenna2, flags
            )
            if bad_antennas:
                self._print(f"  Bad antennas: {sorted(bad_antennas)}")
            
            # Solve using CHAOS
            if self.single_ao:
                # Single reference mode
                from .chain_builder import build_chain
                from .single_chain_solver import SingleChainSolver
                from .ratio_chain_solver import RatioChainSolver
                
                chain_path, chain_quality = build_chain(
                    self.ref_antenna, quality_matrix, bad_antennas
                )
                
                if self.solver == 'single_chain':
                    slv = SingleChainSolver(
                        ref_antenna=self.ref_antenna, mode='diagonal'
                    )
                    jones, info = slv.solve(
                        vis_obs, vis_model, antenna1, antenna2,
                        chain_path, max_iter=100
                    )
                else:
                    slv = RatioChainSolver(
                        ref_antenna=self.ref_antenna, mode='diagonal'
                    )
                    jones, info = slv.solve(
                        vis_obs, vis_model, antenna1, antenna2,
                        quality_matrix, max_iter=100
                    )
            else:
                # Multi-reference weighted mode
                jones, diag = multi_ref_solve_and_combine(
                    vis_obs, vis_model, antenna1, antenna2,
                    quality_matrix, bad_antennas, self.ref_antenna,
                    mode='diagonal', solver=self.solver, max_iter=100
                )
            
            # Polish
            if self.polish:
                self._print(f"  Polishing with tol={self.polish_tol}")
                jones, polish_info = polish_jones(
                    jones, vis_obs, vis_model, antenna1, antenna2,
                    self.ref_antenna, mode='diagonal',
                    max_iter=100, tol=self.polish_tol
                )
            
            all_jones.append(jones)
            all_times.append(data.get('time', np.array([0.0])))
            all_freqs.append(data.get('freq', np.array([1e9])))
        
        # Combine if multiple fields
        if len(all_jones) == 1:
            jones = all_jones[0]
            time = all_times[0]
            freq = all_freqs[0]
        else:
            # Stack along time axis for interpolation later
            jones = np.stack(all_jones, axis=0)
            time = np.array([t.mean() if hasattr(t, 'mean') else t for t in all_times])
            freq = all_freqs[0]
        
        info = {
            'fields': fields,
            'jones_type': jones_type,
            'polished': self.polish,
            'solver': self.solver,
            'single_ao': self.single_ao
        }
        
        return jones, time, freq, info
    
    def _apply_entry(self, entry: ApplyEntry):
        """
        Process a single apply_jones entry.
        """
        self._print(f"\n{'='*60}")
        self._print(f"APPLY: {','.join(entry.jones_terms)} -> {entry.ms_file}")
        self._print(f"Output: {entry.output_column}")
        self._print(f"{'='*60}")
        
        ms_path = entry.ms_file
        feed_type = self._get_feed_type(ms_path)
        
        # Load all Jones terms
        jones_data = {}
        for i, term in enumerate(entry.jones_terms):
            # Find which table has this term
            for table in entry.cal_tables:
                if Path(table).exists() and term in list_jones_terms(table):
                    data = load_jones_term(table, term)
                    jones_data[term] = data
                    self._print(f"Loaded {term} from {table}")
                    break
        
        # Load MS data
        loader = MSLoader(ms_path)
        
        # Get all data (for now, single chunk)
        # TODO: use chunked I/O for large MS
        data = loader.load_data(field_id=0, spw=0)
        
        vis_obs = data['vis_obs']
        antenna1 = data['antenna1']
        antenna2 = data['antenna2']
        n_ant = data['n_ant']
        
        # Build composite Jones
        J_composite = self._build_composite_jones_simple(
            jones_data, entry.jones_terms, n_ant
        )
        
        # Apply parallactic angle if needed
        if entry.parallactic:
            self._print("Adding parallactic angle correction")
            times_pa, psi = compute_parallactic_angles_from_ms(ms_path, 0)
            psi_mean = psi.mean(axis=0)
            P_func = P_jones_linear if feed_type == 'linear' else P_jones_circular
            
            for a in range(n_ant):
                P_a = P_func(psi_mean[a])
                J_composite[a] = J_composite[a] @ P_a
        
        # Correct visibilities
        self._print("Applying corrections...")
        vis_corrected = np.zeros_like(vis_obs)
        
        for bl_idx in range(len(antenna1)):
            a1 = antenna1[bl_idx]
            a2 = antenna2[bl_idx]
            vis_corrected[bl_idx] = unapply_jones(
                vis_obs[bl_idx], J_composite[a1], J_composite[a2]
            )
        
        # Write to MS
        self._write_corrected_data(ms_path, vis_corrected, entry.output_column)
        self._print(f"Written to {entry.output_column}")
    
    def _build_composite_jones_simple(
        self,
        jones_data: Dict[str, Any],
        terms: List[str],
        n_ant: int
    ) -> np.ndarray:
        """
        Build composite Jones matrix J = J_N @ ... @ J_1.
        
        Simple version: assumes all Jones have same shape.
        
        Returns shape (n_ant, 2, 2)
        """
        # Start with identity
        J = I_jones((n_ant,))
        
        # Multiply terms in order
        for term in terms:
            if term not in jones_data:
                self._print(f"Warning: {term} not found, skipping")
                continue
            
            data = jones_data[term]
            jones = data['jones']
            
            # Handle different shapes
            if jones.ndim == 3:
                # (n_ant, 2, 2)
                jones_ant = jones
            elif jones.ndim == 4:
                # (n_time, n_ant, 2, 2) - use first/mean
                jones_ant = jones[0] if jones.shape[0] == 1 else jones.mean(axis=0)
            elif jones.ndim == 5:
                # (n_time, n_freq, n_ant, 2, 2) - use first
                jones_ant = jones[0, 0]
            else:
                self._print(f"Warning: unexpected shape {jones.shape} for {term}")
                continue
            
            # Multiply: J = jones_ant @ J
            for a in range(n_ant):
                J[a] = jones_ant[a] @ J[a]
        
        return J
    
    def _apply_pre_jones(
        self,
        vis_obs: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        pre_jones: Dict[str, Any],
        n_ant: int
    ) -> np.ndarray:
        """Apply pre-solved Jones terms to data."""
        vis_corrected = vis_obs.copy()
        
        # Build composite from pre-jones
        J = I_jones((n_ant,))
        
        for term, data in pre_jones.items():
            jones = data['jones']
            
            if jones.ndim == 3:
                jones_ant = jones
            elif jones.ndim >= 4:
                jones_ant = jones.reshape(-1, n_ant, 2, 2)[0]
            else:
                continue
            
            for a in range(n_ant):
                J[a] = jones_ant[a] @ J[a]
        
        # Apply correction
        for bl_idx in range(len(antenna1)):
            a1 = antenna1[bl_idx]
            a2 = antenna2[bl_idx]
            vis_corrected[bl_idx] = unapply_jones(vis_corrected[bl_idx], J[a1], J[a2])
        
        return vis_corrected
    
    def _remove_parallactic(
        self,
        vis: np.ndarray,
        psi: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        feed_type: str
    ) -> np.ndarray:
        """Remove parallactic angle from visibilities."""
        vis_corrected = vis.copy()
        
        P_func = P_jones_linear if feed_type == 'linear' else P_jones_circular
        
        for bl_idx in range(len(antenna1)):
            a1 = antenna1[bl_idx]
            a2 = antenna2[bl_idx]
            
            P_i = P_func(psi[a1])
            P_j = P_func(psi[a2])
            
            vis_corrected[bl_idx] = unapply_jones(vis_corrected[bl_idx], P_i, P_j)
        
        return vis_corrected
    
    def _detect_bad_antennas(
        self,
        vis_obs: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        flags: np.ndarray,
        threshold: float = 0.05
    ) -> set:
        """Detect bad antennas based on amplitude."""
        n_ant = max(antenna1.max(), antenna2.max()) + 1
        antenna_amps = {i: [] for i in range(n_ant)}
        
        for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
            if flags[idx].any():
                continue
            amp = np.abs(vis_obs[idx]).mean()
            antenna_amps[a1].append(amp)
            antenna_amps[a2].append(amp)
        
        antenna_means = {}
        for ant, amps in antenna_amps.items():
            if len(amps) > 0:
                antenna_means[ant] = np.mean(amps)
            else:
                antenna_means[ant] = 0.0
        
        all_means = [v for v in antenna_means.values() if v > 0]
        if len(all_means) == 0:
            return set()
        
        ref_amp = np.percentile(all_means, 75)
        cutoff = threshold * ref_amp
        
        bad = set()
        for ant, mean_amp in antenna_means.items():
            if 0 < mean_amp < cutoff:
                bad.add(ant)
        
        return bad
    
    def _find_ms_for_field(self, field_name: str) -> str:
        """Find MS containing a field."""
        for ms, fields in self.config.info.items():
            if field_name in fields:
                return ms
        raise ValueError(f"Field {field_name} not found in any MS")
    
    def _get_field_id(self, ms_path: str, field_name: str) -> int:
        """Get field ID from name."""
        from casacore.tables import table
        
        with table(f"{ms_path}/FIELD", ack=False) as tb:
            names = tb.getcol('NAME')
            for i, name in enumerate(names):
                if name == field_name:
                    return i
        
        # Try as integer
        try:
            return int(field_name)
        except ValueError:
            raise ValueError(f"Field {field_name} not found in {ms_path}")
    
    def _get_feed_type(self, ms_path: str) -> str:
        """Detect feed type from MS."""
        from casacore.tables import table
        
        try:
            with table(f"{ms_path}/POLARIZATION", ack=False) as tb:
                corr_types = tb.getcol('CORR_TYPE')[0]
                return detect_feed_type(corr_types)
        except:
            return 'linear'  # Default
    
    def _write_corrected_data(
        self,
        ms_path: str,
        vis_corrected: np.ndarray,
        column: str
    ):
        """Write corrected data to MS."""
        from casacore.tables import table
        
        with table(ms_path, readonly=False, ack=False) as tb:
            # Ensure column exists
            if column not in tb.colnames():
                desc = tb.getcoldesc('DATA')
                tb.addcols({column: desc})
            
            # Convert from (n_bl, 2, 2) to MS format (n_row, n_corr)
            n_row = vis_corrected.shape[0]
            n_corr = 4
            
            data_ms = np.zeros((n_row, n_corr), dtype=vis_corrected.dtype)
            data_ms[:, 0] = vis_corrected[:, 0, 0]
            data_ms[:, 1] = vis_corrected[:, 0, 1]
            data_ms[:, 2] = vis_corrected[:, 1, 0]
            data_ms[:, 3] = vis_corrected[:, 1, 1]
            
            tb.putcol(column, data_ms)


def run_from_config(
    config_path: str,
    ref_antenna: int = 0,
    solver: str = 'single_chain',
    single_ao: bool = False,
    polish: bool = True,
    polish_tol: float = 1e-10,
    solve_only: bool = False,
    apply_only: bool = False,
    verbose: bool = True
):
    """
    Run calibration from config file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file
    ref_antenna : int
        Reference antenna
    solver : str
        'single_chain' or 'ratio_chain'
    single_ao : bool
        Single reference mode
    polish : bool
        Enable least squares polish
    polish_tol : float
        Polish tolerance
    solve_only : bool
        Only run solve, skip apply
    apply_only : bool
        Only run apply, skip solve
    verbose : bool
        Print progress
    """
    config = load_config(config_path)
    
    runner = CalibrationRunner(
        config=config,
        ref_antenna=ref_antenna,
        solver=solver,
        single_ao=single_ao,
        polish=polish,
        polish_tol=polish_tol,
        verbose=verbose
    )
    
    if apply_only:
        runner.run_apply()
    elif solve_only:
        runner.run_solve()
    else:
        runner.run_all()


__all__ = ['CalibrationRunner', 'run_from_config']
