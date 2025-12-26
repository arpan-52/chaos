"""
Chain Builder for CHAOS.

Builds optimal chains from any reference antenna using quality-guided selection.
Computes chain quality as geometric mean of baseline qualities along path.
"""

import numpy as np


def build_chain(ref_ant, quality_matrix, bad_antennas=None):
    """
    Build optimal chain from ref_ant to all other antennas.

    Uses greedy algorithm: at each step, pick the unsolved antenna
    with the best quality baseline to any solved antenna.

    Parameters
    ----------
    ref_ant : int
        Reference antenna index
    quality_matrix : ndarray, shape (n_ant, n_ant)
        Baseline quality Q[i,j]
    bad_antennas : set or None
        Set of bad antenna indices to skip

    Returns
    -------
    chain_path : list of tuples
        [(ant_known, ant_unknown), ...] baselines in chain order
    chain_quality : dict
        {ant: geometric_mean_quality} for each antenna
    """
    if bad_antennas is None:
        bad_antennas = set()

    n_ant = quality_matrix.shape[0]

    # Track which antennas are solved
    solved = {ref_ant}
    chain_path = []

    # Track quality to reach each antenna (product along path, for geomean)
    # quality_product[i] = product of baseline qualities from ref_ant to i
    # path_length[i] = number of baselines
    quality_product = {ref_ant: 1.0}
    path_length = {ref_ant: 0}

    while len(solved) < n_ant:
        best_ant = None
        best_quality = -1
        best_known = None

        # Find best unsolved antenna
        for ant_unknown in range(n_ant):
            if ant_unknown in solved or ant_unknown in bad_antennas:
                continue

            # Find best baseline to this antenna from solved set
            for ant_known in solved:
                q = quality_matrix[ant_known, ant_unknown]

                if q > best_quality:
                    best_quality = q
                    best_ant = ant_unknown
                    best_known = ant_known

        if best_ant is None or best_quality <= 0:
            break  # No more reachable antennas

        # Add to chain
        chain_path.append((best_known, best_ant))
        solved.add(best_ant)

        # Update quality tracking
        quality_product[best_ant] = quality_product[best_known] * best_quality
        path_length[best_ant] = path_length[best_known] + 1

    # Compute geometric mean for each antenna
    chain_quality = {}
    for ant in solved:
        if path_length[ant] > 0:
            chain_quality[ant] = quality_product[ant] ** (1.0 / path_length[ant])
        else:
            chain_quality[ant] = 1.0  # ref antenna

    return chain_path, chain_quality


def build_all_chains(quality_matrix, bad_antennas=None):
    """
    Build chains from every antenna as reference.

    Parameters
    ----------
    quality_matrix : ndarray, shape (n_ant, n_ant)
    bad_antennas : set or None

    Returns
    -------
    all_chains : dict
        {ref_ant: (chain_path, chain_quality)}
    """
    if bad_antennas is None:
        bad_antennas = set()

    n_ant = quality_matrix.shape[0]
    all_chains = {}

    for ref_ant in range(n_ant):
        if ref_ant in bad_antennas:
            continue

        chain_path, chain_quality = build_chain(ref_ant, quality_matrix, bad_antennas)
        all_chains[ref_ant] = (chain_path, chain_quality)

    return all_chains


def get_chain_baselines(chain_path):
    """
    Extract unique baselines from chain path.

    Parameters
    ----------
    chain_path : list of tuples

    Returns
    -------
    baselines : set of tuples
        Unique (a1, a2) pairs with a1 < a2
    """
    baselines = set()
    for a1, a2 in chain_path:
        if a1 < a2:
            baselines.add((a1, a2))
        else:
            baselines.add((a2, a1))
    return baselines


__all__ = ['build_chain', 'build_all_chains', 'get_chain_baselines']
