"""
Chain-based Algebraic Solver.

Computes initial Jones estimate by chaining from reference antenna.
Uses baseline visibility ratios to propagate solution along optimal path.
"""

import numpy as np
from typing import Dict, Set, Tuple, List, Optional
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

from chaos.jones.terms import I_jones


class ChainSolver:
    """
    Chain-based solver for initial Jones estimate.
    
    Algorithm:
        1. Build SNR-weighted graph of baselines
        2. Find optimal spanning tree rooted at reference antenna
        3. Solve algebraically along tree edges
    """
    
    def __init__(
        self,
        ref_antenna: int = 0,
        mode: str = "diagonal",
        bad_antennas: Optional[Set[int]] = None,
    ):
        """
        Initialize chain solver.
        
        Parameters
        ----------
        ref_antenna : int
            Reference antenna (solution anchor)
        mode : str
            'phase_only', 'diagonal', or 'full'
        bad_antennas : set of int
            Antennas to exclude
        """
        self.ref_antenna = ref_antenna
        self.mode = mode
        self.bad_antennas = bad_antennas if bad_antennas else set()
    
    def solve(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve for Jones matrices via chaining.
        
        Parameters
        ----------
        vis_obs : ndarray (n_bl, 2, 2)
            Observed visibilities
        vis_model : ndarray (n_bl, 2, 2)
            Model visibilities
        antenna1, antenna2 : ndarray (n_bl,)
            Antenna indices
        n_ant : int
            Number of antennas
        verbose : bool
            Print progress
        
        Returns
        -------
        jones : ndarray (n_ant, 2, 2)
            Jones matrices
        info : dict
            Chain information
        """
        # Build baseline lookup
        bl_lookup = self._build_baseline_lookup(antenna1, antenna2)
        
        # Compute baseline quality (SNR-like metric)
        quality_matrix = self._compute_quality_matrix(
            vis_obs, vis_model, antenna1, antenna2, n_ant
        )
        
        # Build spanning tree
        chain_path = self._build_chain(quality_matrix, n_ant)
        
        if verbose:
            print(f"[CHAOS] Chain: {len(chain_path)} edges from ref_ant={self.ref_antenna}")
        
        # Solve along chain
        jones = self._solve_along_chain(
            vis_obs, vis_model, antenna1, antenna2, n_ant,
            chain_path, bl_lookup
        )
        
        info = {
            "chain_path": chain_path,
            "quality_matrix": quality_matrix,
        }
        
        return jones, info
    
    def _build_baseline_lookup(
        self, 
        antenna1: np.ndarray, 
        antenna2: np.ndarray
    ) -> Dict[Tuple[int, int], int]:
        """Build dict mapping (a1, a2) -> baseline index."""
        lookup = {}
        for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
            lookup[(a1, a2)] = idx
            lookup[(a2, a1)] = idx  # Both directions
        return lookup
    
    def _compute_quality_matrix(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
    ) -> np.ndarray:
        """
        Compute baseline quality matrix.
        
        Quality is based on amplitude of visibility (proxy for SNR).
        
        Parameters
        ----------
        ...
        
        Returns
        -------
        quality : ndarray (n_ant, n_ant)
            Symmetric quality matrix
        """
        quality = np.zeros((n_ant, n_ant), dtype=np.float64)
        
        for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
            if a1 in self.bad_antennas or a2 in self.bad_antennas:
                continue
            
            # Use mean amplitude as quality proxy
            amp = np.abs(vis_obs[idx]).mean()
            if np.isfinite(amp) and amp > 0:
                quality[a1, a2] = amp
                quality[a2, a1] = amp
        
        return quality
    
    def _build_chain(
        self,
        quality_matrix: np.ndarray,
        n_ant: int,
    ) -> List[Tuple[int, int]]:
        """
        Build optimal chain (spanning tree) from reference antenna.
        
        Uses maximum spanning tree (inverted to minimum for scipy).
        
        Parameters
        ----------
        quality_matrix : ndarray (n_ant, n_ant)
            Baseline qualities
        n_ant : int
            Number of antennas
        
        Returns
        -------
        chain_path : list of (from_ant, to_ant)
            Edges in order to traverse
        """
        # Convert to cost matrix (invert quality for minimum spanning tree)
        max_quality = quality_matrix.max()
        if max_quality == 0:
            max_quality = 1.0
        
        cost = np.zeros_like(quality_matrix)
        nonzero = quality_matrix > 0
        cost[nonzero] = max_quality / quality_matrix[nonzero]
        
        # Set bad antennas to inf cost
        for ant in self.bad_antennas:
            cost[ant, :] = np.inf
            cost[:, ant] = np.inf
        
        # Build sparse matrix and compute MST
        cost_sparse = csr_matrix(cost)
        mst = minimum_spanning_tree(cost_sparse)
        mst_array = mst.toarray()
        
        # BFS from reference antenna to get edge order
        chain_path = []
        visited = {self.ref_antenna}
        queue = [self.ref_antenna]
        
        while queue:
            current = queue.pop(0)
            
            # Find neighbors in MST
            for neighbor in range(n_ant):
                if neighbor in visited:
                    continue
                if neighbor in self.bad_antennas:
                    continue
                
                # Check if edge exists (either direction)
                if mst_array[current, neighbor] > 0 or mst_array[neighbor, current] > 0:
                    chain_path.append((current, neighbor))
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return chain_path
    
    def _solve_along_chain(
        self,
        vis_obs: np.ndarray,
        vis_model: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        n_ant: int,
        chain_path: List[Tuple[int, int]],
        bl_lookup: Dict[Tuple[int, int], int],
    ) -> np.ndarray:
        """
        Solve for Jones matrices along chain.
        
        For each edge (known -> unknown):
            V_obs = J_known @ M @ J_unknown^H
            => J_unknown = (V_obs @ J_known @ M^{-1})^H  (approximately)
        
        Parameters
        ----------
        ...
        
        Returns
        -------
        jones : ndarray (n_ant, 2, 2)
        """
        # Initialize with identity
        jones = I_jones((n_ant,))
        
        # Reference antenna constraint based on mode
        # For G, B, G[p]: phase = 0 for both pols
        # This is already satisfied by identity initialization
        
        solved = {self.ref_antenna}
        
        for (known_ant, unknown_ant) in chain_path:
            if unknown_ant in solved:
                continue
            
            # Get baseline visibility
            bl_idx = bl_lookup.get((known_ant, unknown_ant))
            if bl_idx is None:
                bl_idx = bl_lookup.get((unknown_ant, known_ant))
                if bl_idx is None:
                    # No baseline, skip
                    continue
            
            V_obs = vis_obs[bl_idx]
            M = vis_model[bl_idx]
            
            # Determine direction
            a1 = antenna1[bl_idx]
            a2 = antenna2[bl_idx]
            
            if a1 == known_ant:
                # V = J_known @ M @ J_unknown^H
                # J_unknown = solve from this
                J_known = jones[known_ant]
                J_unknown = self._solve_for_unknown(V_obs, M, J_known, "right")
            else:
                # V = J_unknown @ M @ J_known^H
                J_known = jones[known_ant]
                J_unknown = self._solve_for_unknown(V_obs, M, J_known, "left")
            
            jones[unknown_ant] = J_unknown
            solved.add(unknown_ant)
        
        return jones
    
    def _solve_for_unknown(
        self,
        V_obs: np.ndarray,
        M: np.ndarray,
        J_known: np.ndarray,
        position: str,
    ) -> np.ndarray:
        """
        Solve for unknown Jones given known Jones.
        
        If position == "right": V = J_known @ M @ J_unknown^H
        If position == "left":  V = J_unknown @ M @ J_known^H
        
        For diagonal mode, this is straightforward ratio.
        """
        if self.mode == "phase_only":
            return self._solve_phase_only(V_obs, M, J_known, position)
        elif self.mode == "diagonal":
            return self._solve_diagonal(V_obs, M, J_known, position)
        else:  # full
            return self._solve_full(V_obs, M, J_known, position)
    
    def _solve_diagonal(
        self,
        V_obs: np.ndarray,
        M: np.ndarray,
        J_known: np.ndarray,
        position: str,
    ) -> np.ndarray:
        """
        Solve for diagonal Jones.
        
        For V = J_i @ M @ J_j^H with diagonal J:
            V_XX = g_i_X * M_XX * g_j_X^*
            V_YY = g_i_Y * M_YY * g_j_Y^*
        """
        J = np.zeros((2, 2), dtype=np.complex128)
        
        for p in range(2):
            V_pp = V_obs[p, p]
            M_pp = M[p, p]
            g_known = J_known[p, p]
            
            if np.abs(M_pp) < 1e-10 or np.abs(g_known) < 1e-10:
                J[p, p] = 1.0
                continue
            
            if position == "right":
                # V = g_known * M * g_unknown^*
                # g_unknown^* = V / (g_known * M)
                # g_unknown = (V / (g_known * M))^*
                g_unknown = np.conj(V_pp / (g_known * M_pp))
            else:
                # V = g_unknown * M * g_known^*
                # g_unknown = V / (M * g_known^*)
                g_unknown = V_pp / (M_pp * np.conj(g_known))
            
            J[p, p] = g_unknown
        
        return J
    
    def _solve_phase_only(
        self,
        V_obs: np.ndarray,
        M: np.ndarray,
        J_known: np.ndarray,
        position: str,
    ) -> np.ndarray:
        """
        Solve for phase-only Jones (amplitude = 1).
        """
        J = self._solve_diagonal(V_obs, M, J_known, position)
        
        # Normalize to unit amplitude
        for p in range(2):
            amp = np.abs(J[p, p])
            if amp > 1e-10:
                J[p, p] = J[p, p] / amp
        
        return J
    
    def _solve_full(
        self,
        V_obs: np.ndarray,
        M: np.ndarray,
        J_known: np.ndarray,
        position: str,
    ) -> np.ndarray:
        """
        Solve for full 2x2 Jones.
        
        This is more complex - use iterative approach or approximation.
        For now, use diagonal as approximation.
        """
        # For D-terms, start with diagonal approximation
        # The polish step will refine this
        return self._solve_diagonal(V_obs, M, J_known, position)
