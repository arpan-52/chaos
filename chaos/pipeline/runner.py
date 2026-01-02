"""
Calibration Pipeline Runner.

Executes calibration pipeline from configuration.
"""

import numpy as np
from typing import Dict, Optional, Set, List
from pathlib import Path

from chaos.pipeline.config_parser import (
    CalConfig, SolveEntry, ApplyEntry, load_config,
    parse_scan_range, parse_spw_range, parse_time_interval, parse_freq_interval,
)
from chaos.core.solver import solve_jones, solve_jones_per_channel
from chaos.io.ms_reader import MSReader
from chaos.io.table_io import save_jones_table, load_jones_table, list_jones_terms
from chaos.io.applycal import applycal
from chaos.jones.terms import I_jones
from chaos.jones.operations import unapply_jones
from chaos.jones.feed_type import get_feed_type_from_ms, FEED_LINEAR


class CalibrationRunner:
    """
    Run calibration pipeline from configuration.
    """
    
    def __init__(
        self,
        config: CalConfig,
        ref_antenna: int = 0,
        flag_threshold: float = 0.8,
        polish_tol: float = 1e-10,
        max_iter: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize pipeline runner.
        
        Parameters
        ----------
        config : CalConfig
            Parsed configuration
        ref_antenna : int
            Reference antenna
        flag_threshold : float
            Fraction of flagged data to consider antenna bad (default 0.8 = 80%)
        polish_tol : float
            Tolerance for polish convergence
        max_iter : int
            Max iterations for polish
        verbose : bool
            Print progress
        """
        self.config = config
        self.ref_antenna = ref_antenna
        self.flag_threshold = flag_threshold
        self.polish_tol = polish_tol
        self.max_iter = max_iter
        self.verbose = verbose
    
    def _print(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[CHAOS] {msg}")
    
    def run_all(self):
        """Run all solve and apply entries."""
        self.run_solve()
        self.run_apply()
    
    def run_solve(self):
        """Run all solve entries."""
        for entry in self.config.solve_entries:
            self._solve_entry(entry)
    
    def run_apply(self):
        """Run all apply entries."""
        for entry in self.config.apply_entries:
            self._apply_entry(entry)
    
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
        
        # Find MS for first field
        ms_path = self._find_ms_for_field(entry.fields[0].split(":")[0])
        
        # Get feed type
        feed_type = get_feed_type_from_ms(ms_path)
        self._print(f"Feed type: {feed_type}")
        
        # Pre-load any pre-apply tables
        pre_jones = {}
        for term in entry.pre_apply_jones:
            for table in entry.pre_apply_tables:
                if Path(table).exists() and term in list_jones_terms(table):
                    data = load_jones_table(table, term)
                    pre_jones[term] = data
                    self._print(f"Loaded pre-apply {term} from {table}")
                    break
        
        # Solve each Jones term in sequence
        solved_this_entry = {}
        
        for i, (jones_type, field_spec) in enumerate(zip(entry.jones_types, entry.fields)):
            self._print(f"\n--- Solving {jones_type} on {field_spec} ---")
            
            # Combine pre-apply terms
            all_pre = {**pre_jones, **solved_this_entry}
            
            # Parse field (handle 3C147:3C286 syntax)
            fields = field_spec.split(":")
            
            # Solve
            jones, time, freq = self._solve_jones_term(
                ms_path=ms_path,
                jones_type=jones_type,
                fields=fields,
                entry=entry,
                feed_type=feed_type,
                pre_apply=all_pre,
            )
            
            # Save to output table
            save_jones_table(
                entry.output_table,
                jones_type,
                jones,
                time,
                freq,
                field=field_spec,
                ref_antenna=self.ref_antenna,
                mode=self._get_mode(jones_type),
                metadata={"fields": fields},
                overwrite=True,
            )
            
            # Add to solved for this entry
            solved_this_entry[jones_type] = {
                "jones": jones,
                "time": time,
                "freq": freq,
            }
            
            self._print(f"Saved {jones_type} to {entry.output_table}")
    
    def _solve_jones_term(
        self,
        ms_path: str,
        jones_type: str,
        fields: List[str],
        entry: SolveEntry,
        feed_type: str,
        pre_apply: Dict,
    ) -> tuple:
        """
        Solve for a single Jones term.
        
        Returns
        -------
        jones : ndarray
        time : ndarray
        freq : ndarray
        """
        reader = MSReader(ms_path)
        
        # Get field ID
        field_info = reader.get_field_info()
        field_id = None
        for i, name in enumerate(field_info["names"]):
            if name in fields:
                field_id = i
                break
        
        if field_id is None:
            field_id = 0
            self._print(f"Warning: field {fields} not found, using field 0")
        
        # Determine if per-channel
        freq_int = parse_freq_interval(entry.freq_interval)
        per_channel = freq_int == "per_channel"
        
        # Read data
        data = reader.read_data(
            field_id=field_id,
            data_column="DATA",
            model_column=entry.model_column,
            average_channels=not per_channel,
        )
        
        vis_obs = data.vis_obs
        vis_model = data.vis_model
        flags = data.flags
        antenna1 = data.antenna1
        antenna2 = data.antenna2
        n_ant = data.n_ant
        time = data.time
        freq = data.freq
        
        # Average to single time/freq point for now
        # TODO: implement solint properly
        if vis_obs.ndim == 4:
            # (n_bl, n_chan, 2, 2) - average over channels if not per_channel
            if not per_channel:
                vis_obs = vis_obs.mean(axis=1)
                vis_model = vis_model.mean(axis=1)
                flags = flags.any(axis=1)
        
        # Apply pre-solved terms
        if pre_apply:
            self._print(f"Applying pre-solved: {list(pre_apply.keys())}")
            vis_obs = self._apply_pre_jones(vis_obs, pre_apply, antenna1, antenna2, n_ant)
        
        # Apply parallactic angle correction if needed
        if entry.parallactic:
            self._print("Removing parallactic angle")
            vis_obs = self._remove_parallactic(
                vis_obs, ms_path, field_id, antenna1, antenna2, feed_type
            )
        
        # Solve
        if per_channel:
            jones, diag = solve_jones_per_channel(
                vis_obs, vis_model, antenna1, antenna2, n_ant,
                ref_antenna=self.ref_antenna,
                jones_type=jones_type,
                flags=flags,
                max_iter=self.max_iter,
                polish_tol=self.polish_tol,
                verbose=self.verbose,
            )
        else:
            jones, diag = solve_jones(
                vis_obs, vis_model, antenna1, antenna2, n_ant,
                ref_antenna=self.ref_antenna,
                jones_type=jones_type,
                flag_threshold=self.flag_threshold,
                flags=flags,
                max_iter=self.max_iter,
                polish_tol=self.polish_tol,
                verbose=self.verbose,
            )
        
        # Use mean time for single solution
        time_out = np.array([np.mean(time)])
        freq_out = np.array([np.mean(freq)]) if not per_channel else freq
        
        return jones, time_out, freq_out
    
    def _apply_pre_jones(
        self,
        vis_obs: np.ndarray,
        pre_jones: Dict,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
    ) -> np.ndarray:
        """Apply pre-solved Jones to data (correct it out)."""
        vis_corrected = vis_obs.copy()
        
        # Build composite Jones
        J = I_jones((n_ant,))
        
        for term, data in pre_jones.items():
            jones = data["jones"]
            
            # Get per-antenna Jones (handle different shapes)
            if jones.ndim == 3:
                jones_ant = jones
            elif jones.ndim == 4:
                jones_ant = jones[0]  # First time
            elif jones.ndim == 5:
                jones_ant = jones[0, 0]  # First time, first freq
            else:
                continue
            
            for ant in range(n_ant):
                J[ant] = jones_ant[ant] @ J[ant]
        
        # Apply correction to each baseline
        for bl_idx in range(len(antenna1)):
            a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
            
            if vis_corrected.ndim == 3:
                vis_corrected[bl_idx] = unapply_jones(vis_corrected[bl_idx], J[a1], J[a2])
            else:
                for ch in range(vis_corrected.shape[1]):
                    vis_corrected[bl_idx, ch] = unapply_jones(
                        vis_corrected[bl_idx, ch], J[a1], J[a2]
                    )
        
        return vis_corrected
    
    def _remove_parallactic(
        self,
        vis_obs: np.ndarray,
        ms_path: str,
        field_id: int,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        feed_type: str,
    ) -> np.ndarray:
        """Remove parallactic angle rotation from data."""
        from chaos.jones.terms import P_jones_linear, P_jones_circular
        from chaos.utils.parallactic import compute_parallactic_angles_from_ms
        
        times, psi = compute_parallactic_angles_from_ms(ms_path, field_id)
        
        # Use mean parallactic angle
        psi_mean = psi.mean(axis=0)  # (n_ant,)
        n_ant = len(psi_mean)
        
        P_func = P_jones_linear if feed_type == FEED_LINEAR else P_jones_circular
        
        vis_corrected = vis_obs.copy()
        
        for bl_idx in range(len(antenna1)):
            a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
            
            P_i = P_func(psi_mean[a1])
            P_j = P_func(psi_mean[a2])
            
            if vis_corrected.ndim == 3:
                vis_corrected[bl_idx] = unapply_jones(vis_corrected[bl_idx], P_i, P_j)
            else:
                for ch in range(vis_corrected.shape[1]):
                    vis_corrected[bl_idx, ch] = unapply_jones(
                        vis_corrected[bl_idx, ch], P_i, P_j
                    )
        
        return vis_corrected
    
    def _apply_entry(self, entry: ApplyEntry):
        """Process a single apply_jones entry."""
        self._print(f"\n{'='*60}")
        self._print(f"APPLY: {','.join(entry.jones_types)} -> {entry.ms_file}")
        self._print(f"Output: {entry.output_column}")
        self._print(f"{'='*60}")
        
        applycal(
            ms_path=entry.ms_file,
            cal_tables=entry.cal_tables,
            jones_types=entry.jones_types,
            output_column=entry.output_column,
            time_interp=entry.time_interp,
            freq_interp=entry.freq_interp,
            verbose=self.verbose,
        )
    
    def _find_ms_for_field(self, field_name: str) -> str:
        """Find MS containing a field."""
        for ms, fields in self.config.info.items():
            if field_name in fields:
                return ms
        
        # Return first MS if not found
        if self.config.info:
            return list(self.config.info.keys())[0]
        
        raise ValueError(f"No MS found for field {field_name}")
    
    def _get_mode(self, jones_type: str) -> str:
        """Get solve mode for jones type."""
        if jones_type == "G[p]":
            return "phase_only"
        elif jones_type == "D":
            return "full"
        else:
            return "diagonal"


def run_pipeline(
    config_path: str,
    ref_antenna: int = 0,
    flag_threshold: float = 0.8,
    polish_tol: float = 1e-10,
    max_iter: int = 100,
    solve_only: bool = False,
    apply_only: bool = False,
    verbose: bool = True,
):
    """
    Run calibration pipeline from config file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file
    ref_antenna : int
        Reference antenna
    flag_threshold : float
        Fraction flagged to consider antenna bad
    polish_tol : float
        Polish convergence tolerance
    max_iter : int
        Max polish iterations
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
        flag_threshold=flag_threshold,
        polish_tol=polish_tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    
    if apply_only:
        runner.run_apply()
    elif solve_only:
        runner.run_solve()
    else:
        runner.run_all()
