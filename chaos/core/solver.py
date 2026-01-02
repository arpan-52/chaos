"""
Main CHAOS Solver.

Entry point for solving Jones matrices from visibilities.
Always uses: initial guess (chain) + polish (least squares).
"""

import numpy as np
from typing import Dict, Set, Optional, Literal, Tuple

from chaos.core.chain_solver import ChainSolver
from chaos.core.polish import polish_jones
from chaos.jones.terms import I_jones
from chaos.utils.flagging import flag_rfi_iterative, FlagStats


JonesType = Literal["K", "B", "G", "G[p]", "D"]
SolveMode = Literal["phase_only", "diagonal", "full"]


def solve_jones(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant: int,
    ref_antenna: int = 0,
    jones_type: JonesType = "G",
    bad_antennas: Optional[Set[int]] = None,
    flag_threshold: float = 0.8,
    flags: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    rfi_flag: bool = True,
    rfi_sigma: float = 5.0,
    max_iter: int = 100,
    polish_tol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Solve for Jones matrices using CHAOS algorithm.
    
    Algorithm:
        1. RFI flagging (robust MAD-based)
        2. Build optimal chain from reference antenna
        3. Algebraic solution along chain (initial guess)
        4. Least-squares polish over all baselines (always runs)
    
    Parameters
    ----------
    vis_obs : ndarray (n_bl, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_bl, 2, 2)
        Model visibilities
    antenna1 : ndarray (n_bl,)
        First antenna indices
    antenna2 : ndarray (n_bl,)
        Second antenna indices
    n_ant : int
        Number of antennas
    ref_antenna : int
        Reference antenna index
    jones_type : str
        'K' (delay), 'B' (bandpass), 'G' (gain), 'G[p]' (phase-only), 'D' (leakage)
    bad_antennas : set of int, optional
        Antennas to exclude from solving
    flag_threshold : float
        Fraction of flagged data above which antenna is considered bad
    flags : ndarray (n_bl, 2, 2), optional
        Boolean flags (True = flagged)
    weights : ndarray (n_bl, 2, 2), optional
        Visibility weights
    rfi_flag : bool
        Enable robust RFI flagging (default: True)
    rfi_sigma : float
        RFI flagging threshold in sigma (default: 5.0)
    max_iter : int
        Max iterations for polish
    polish_tol : float
        Convergence tolerance for polish
    verbose : bool
        Print progress
    
    Returns
    -------
    jones : ndarray (n_ant, 2, 2)
        Solved Jones matrices
    diagnostics : dict
        Diagnostic information
    """
    if bad_antennas is None:
        bad_antennas = set()
    
    # Initialize flags
    if flags is None:
        flags = np.zeros(vis_obs.shape, dtype=bool)
    else:
        flags = flags.copy()
    
    # Step 0: RFI flagging
    flag_stats = None
    if rfi_flag:
        if verbose:
            print(f"[CHAOS] Running robust RFI flagging (sigma={rfi_sigma})")
        
        flags, flag_stats = flag_rfi_iterative(
            vis_obs, sigma=rfi_sigma, max_iter=3,
            existing_flags=flags, verbose=verbose
        )
        
        if verbose:
            print(f"[CHAOS] Flagged {flag_stats.fraction_flagged*100:.1f}% of data")
    
    # Detect bad antennas from flagging
    detected_bad = _detect_bad_antennas_from_flags(
        flags, antenna1, antenna2, n_ant, flag_threshold
    )
    bad_antennas = bad_antennas | detected_bad
    if verbose and detected_bad:
        print(f"[CHAOS] Auto-detected bad antennas (>{flag_threshold*100:.0f}% flagged): {sorted(detected_bad)}")
    
    # Determine solve mode from jones_type
    if jones_type == "G[p]":
        mode = "phase_only"
    elif jones_type == "D":
        mode = "full"  # D-terms are off-diagonal
    else:
        mode = "diagonal"
    
    if verbose:
        print(f"[CHAOS] Solving {jones_type} ({mode}) with ref_ant={ref_antenna}")
        print(f"[CHAOS] Data: {len(antenna1)} baselines, {n_ant} antennas")
        if bad_antennas:
            print(f"[CHAOS] Bad antennas: {sorted(bad_antennas)}")
    
    # Step 1: Build chain and get initial guess
    chain_solver = ChainSolver(
        ref_antenna=ref_antenna,
        mode=mode,
        bad_antennas=bad_antennas,
    )
    
    jones_init, chain_info = chain_solver.solve(
        vis_obs, vis_model, antenna1, antenna2, n_ant, 
        verbose=verbose
    )
    
    if verbose:
        print(f"[CHAOS] Initial guess computed via chain")
    
    # Step 2: Polish with least squares (always)
    jones_polished, polish_info = polish_jones(
        jones_init=jones_init,
        vis_obs=vis_obs,
        vis_model=vis_model,
        antenna1=antenna1,
        antenna2=antenna2,
        ref_antenna=ref_antenna,
        mode=mode,
        jones_type=jones_type,
        max_iter=max_iter,
        tol=polish_tol,
        verbose=verbose,
    )
    
    # Set bad antennas to identity
    for ant in bad_antennas:
        jones_polished[ant] = I_jones()
    
    # Diagnostics
    diagnostics = {
        "ref_antenna": ref_antenna,
        "jones_type": jones_type,
        "mode": mode,
        "bad_antennas": bad_antennas,
        "chain_info": chain_info,
        "polish_info": polish_info,
        "flag_stats": flag_stats,
        "n_ant": n_ant,
        "n_bl": len(antenna1),
    }
    
    return jones_polished, diagnostics


def _detect_bad_antennas_from_flags(
    flags: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant: int,
    threshold: float,
) -> Set[int]:
    """
    Detect bad antennas based on fraction of flagged data.
    
    An antenna is considered bad if more than `threshold` fraction
    of its baselines are flagged.
    
    Parameters
    ----------
    flags : ndarray (n_bl, ...)
        Boolean flags
    antenna1, antenna2 : ndarray (n_bl,)
        Antenna indices
    n_ant : int
        Number of antennas
    threshold : float
        Fraction threshold (e.g., 0.8 = 80%)
    
    Returns
    -------
    bad_antennas : set of int
    """
    # Count total and flagged per antenna
    total_per_ant = np.zeros(n_ant, dtype=int)
    flagged_per_ant = np.zeros(n_ant, dtype=int)
    
    # Collapse flags to per-baseline
    if flags.ndim > 1:
        bl_flagged = flags.reshape(len(flags), -1).all(axis=1)
    else:
        bl_flagged = flags
    
    for bl_idx in range(len(antenna1)):
        a1, a2 = antenna1[bl_idx], antenna2[bl_idx]
        total_per_ant[a1] += 1
        total_per_ant[a2] += 1
        if bl_flagged[bl_idx]:
            flagged_per_ant[a1] += 1
            flagged_per_ant[a2] += 1
    
    bad = set()
    for ant in range(n_ant):
        if total_per_ant[ant] > 0:
            frac = flagged_per_ant[ant] / total_per_ant[ant]
            if frac > threshold:
                bad.add(ant)
    
    return bad


def solve_jones_per_channel(
    vis_obs: np.ndarray,
    vis_model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    n_ant: int,
    ref_antenna: int = 0,
    jones_type: JonesType = "B",
    bad_antennas: Optional[Set[int]] = None,
    flags: Optional[np.ndarray] = None,
    rfi_flag: bool = True,
    rfi_sigma: float = 5.0,
    max_iter: int = 100,
    polish_tol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Solve for Jones matrices per channel (for bandpass).
    
    Parameters
    ----------
    vis_obs : ndarray (n_bl, n_chan, 2, 2)
        Observed visibilities
    vis_model : ndarray (n_bl, n_chan, 2, 2)
        Model visibilities
    ...
    rfi_flag : bool
        Enable robust RFI flagging
    rfi_sigma : float
        RFI flagging threshold
    
    Returns
    -------
    jones : ndarray (n_chan, n_ant, 2, 2)
        Solved Jones matrices per channel
    diagnostics : dict
    """
    n_bl, n_chan = vis_obs.shape[:2]
    
    if bad_antennas is None:
        bad_antennas = set()
    
    # RFI flag across all channels first
    if rfi_flag and flags is None:
        from chaos.utils.flagging import flag_rfi_iterative
        flags, _ = flag_rfi_iterative(
            vis_obs, sigma=rfi_sigma, max_iter=3, verbose=verbose
        )
    elif flags is None:
        flags = np.zeros(vis_obs.shape, dtype=bool)
    
    jones_all = np.zeros((n_chan, n_ant, 2, 2), dtype=np.complex128)
    
    for ch in range(n_chan):
        if verbose and ch % 10 == 0:
            print(f"[CHAOS] Solving channel {ch}/{n_chan}")
        
        vis_obs_ch = vis_obs[:, ch]
        vis_model_ch = vis_model[:, ch]
        flags_ch = flags[:, ch] if flags is not None else None
        
        jones_ch, _ = solve_jones(
            vis_obs_ch, vis_model_ch, antenna1, antenna2, n_ant,
            ref_antenna=ref_antenna,
            jones_type=jones_type,
            bad_antennas=bad_antennas,
            flags=flags_ch,
            rfi_flag=False,  # Already flagged above
            max_iter=max_iter,
            polish_tol=polish_tol,
            verbose=False,
        )
        
        jones_all[ch] = jones_ch
    
    diagnostics = {
        "ref_antenna": ref_antenna,
        "jones_type": jones_type,
        "n_chan": n_chan,
        "bad_antennas": bad_antennas,
    }
    
    return jones_all, diagnostics
