"""
Weighted Combiner for CHAOS.

Combines solutions from multiple reference antennas using:
1. Phase alignment to user-specified reference
2. Weighted averaging using geometric mean of chain quality
"""

import numpy as np


def align_phases(solutions, chain_qualities, ref_ant, n_ant, mode='diagonal'):
    """
    Align phases of all solutions to user-specified reference antenna.

    For each solution J^(r) (solved with antenna r as reference):
    - Get phase of ref_ant in that solution
    - Subtract from all antennas to align

    Parameters
    ----------
    solutions : dict
        {r: jones_array (n_ant, 2, 2)} for each reference r
    chain_qualities : dict
        {r: {ant: quality}} for each reference r
    ref_ant : int
        User-specified reference antenna (phases will be 0 here)
    n_ant : int
    mode : str
        'phase_only', 'diagonal', or 'full'

    Returns
    -------
    aligned : dict
        {r: aligned_jones_array (n_ant, 2, 2)}
    """
    aligned = {}

    for r, jones in solutions.items():
        jones_aligned = jones.copy()

        # Get phases of ref_ant in this solution
        phase_XX = np.angle(jones[ref_ant, 0, 0])
        phase_YY = np.angle(jones[ref_ant, 1, 1])

        if mode in ['phase_only', 'diagonal']:
            # Diagonal: rotate diagonal elements only
            jones_aligned[:, 0, 0] = np.abs(jones[:, 0, 0]) * np.exp(1j * (np.angle(jones[:, 0, 0]) - phase_XX))
            jones_aligned[:, 1, 1] = np.abs(jones[:, 1, 1]) * np.exp(1j * (np.angle(jones[:, 1, 1]) - phase_YY))

        else:  # full
            # Full Jones: rotate each row by corresponding phase
            # Row 0 (X-feed) by phase_XX, Row 1 (Y-feed) by phase_YY
            jones_aligned[:, 0, 0] = np.abs(jones[:, 0, 0]) * np.exp(1j * (np.angle(jones[:, 0, 0]) - phase_XX))
            jones_aligned[:, 0, 1] = np.abs(jones[:, 0, 1]) * np.exp(1j * (np.angle(jones[:, 0, 1]) - phase_XX))
            jones_aligned[:, 1, 0] = np.abs(jones[:, 1, 0]) * np.exp(1j * (np.angle(jones[:, 1, 0]) - phase_YY))
            jones_aligned[:, 1, 1] = np.abs(jones[:, 1, 1]) * np.exp(1j * (np.angle(jones[:, 1, 1]) - phase_YY))

        aligned[r] = jones_aligned

    return aligned


def combine_weighted(aligned_solutions, chain_qualities, n_ant, mode='diagonal'):
    """
    Weighted average of aligned solutions.

    Weight for antenna i from reference r = chain_quality[r][i]
    Normalized so max weight = 1

    Parameters
    ----------
    aligned_solutions : dict
        {r: jones_array (n_ant, 2, 2)}
    chain_qualities : dict
        {r: {ant: quality}}
    n_ant : int
    mode : str

    Returns
    -------
    jones_final : ndarray (n_ant, 2, 2)
        Final weighted Jones matrices
    weights_used : ndarray (n_ant, n_ref)
        Weights used for each antenna from each reference
    """
    ref_list = sorted(aligned_solutions.keys())
    n_ref = len(ref_list)

    # Build weight matrix
    weights = np.zeros((n_ant, n_ref))
    for r_idx, r in enumerate(ref_list):
        for ant in range(n_ant):
            if ant in chain_qualities[r]:
                weights[ant, r_idx] = chain_qualities[r][ant]
            else:
                weights[ant, r_idx] = 0.0

    # Normalize per antenna (max = 1)
    for ant in range(n_ant):
        max_w = weights[ant, :].max()
        if max_w > 0:
            weights[ant, :] /= max_w

    # Weighted average
    jones_final = np.zeros((n_ant, 2, 2), dtype=complex)

    for ant in range(n_ant):
        w_sum = 0.0
        j_sum = np.zeros((2, 2), dtype=complex)

        for r_idx, r in enumerate(ref_list):
            w = weights[ant, r_idx]
            if w > 0:
                j_sum += w * aligned_solutions[r][ant]
                w_sum += w

        if w_sum > 0:
            jones_final[ant] = j_sum / w_sum
        else:
            jones_final[ant] = np.eye(2)

    return jones_final, weights


def multi_ref_solve_and_combine(
    vis_obs, vis_model, antenna1, antenna2,
    quality_matrix, bad_antennas, ref_ant, mode, solver_type, max_iter=100
):
    """
    Main entry point: solve from all antennas, align, and combine.

    Parameters
    ----------
    vis_obs : ndarray (n_bl, 2, 2)
    vis_model : ndarray (n_bl, 2, 2)
    antenna1, antenna2 : ndarray (n_bl,)
    quality_matrix : ndarray (n_ant, n_ant)
    bad_antennas : set
    ref_ant : int
        User-specified reference for final alignment
    mode : str
        'phase_only', 'diagonal', 'full'
    solver_type : str
        'single_chain' or 'ratio_chain'
    max_iter : int

    Returns
    -------
    jones_final : ndarray (n_ant, 2, 2)
    diagnostics : dict
    """
    from .chain_builder import build_chain
    from .single_chain_solver import SingleChainSolver
    from .ratio_chain_solver import RatioChainSolver

    n_ant = quality_matrix.shape[0]

    # Storage for all solutions
    solutions = {}
    chain_qualities = {}
    solver_infos = {}

    print(f"\n[CHAOS] Multi-reference solving ({solver_type}, {mode} mode)")
    print(f"[CHAOS] Reference antenna for final alignment: {ref_ant}")
    print(f"[CHAOS] Bad antennas: {sorted(bad_antennas)}")
    print(f"[CHAOS] Solving from {n_ant - len(bad_antennas)} antennas as reference...")

    # Solve from each antenna as reference
    for r in range(n_ant):
        if r in bad_antennas:
            continue

        # Build chain from this reference
        chain_path, chain_qual = build_chain(r, quality_matrix, bad_antennas)

        if solver_type == 'single_chain':
            solver = SingleChainSolver(ref_antenna=r, mode=mode)
            jones, info = solver.solve(
                vis_obs, vis_model, antenna1, antenna2,
                chain_path, max_iter=max_iter
            )
        else:  # ratio_chain
            solver = RatioChainSolver(ref_antenna=r, mode=mode)
            jones, info = solver.solve(
                vis_obs, vis_model, antenna1, antenna2,
                quality_matrix, max_iter=max_iter
            )

        solutions[r] = jones
        chain_qualities[r] = chain_qual
        solver_infos[r] = info

        # Print progress
        n_solved = len(chain_qual)
        print(f"[CHAOS]   ref={r}: solved {n_solved}/{n_ant} antennas, "
              f"J_ref[0,0]={info['J_ref'][0,0]:.4f}")

    # Align phases to user-specified reference
    print(f"\n[CHAOS] Aligning phases to ref_ant={ref_ant}")
    aligned = align_phases(solutions, chain_qualities, ref_ant, n_ant, mode)

    # Weighted combination
    print(f"[CHAOS] Computing weighted average...")
    jones_final, weights = combine_weighted(aligned, chain_qualities, n_ant, mode)

    # Print weight statistics
    for ant in range(min(5, n_ant)):  # Show first 5 antennas
        nonzero_weights = weights[ant, weights[ant, :] > 0]
        if len(nonzero_weights) > 0:
            print(f"[CHAOS]   ant={ant}: {len(nonzero_weights)} contributing refs, "
                  f"weight range [{nonzero_weights.min():.3f}, {nonzero_weights.max():.3f}]")

    diagnostics = {
        'solutions': solutions,
        'chain_qualities': chain_qualities,
        'solver_infos': solver_infos,
        'aligned_solutions': aligned,
        'weights': weights,
        'ref_ant': ref_ant,
        'mode': mode,
        'solver_type': solver_type
    }

    return jones_final, diagnostics


__all__ = ['align_phases', 'combine_weighted', 'multi_ref_solve_and_combine']
