"""
Single Chain Solver for CHAOS.

Solves Jones matrices using direct algebraic chaining from reference antenna.

Modes:
- phase_only: J_ref = I (identity), only phases solved via chaining
- diagonal: J_ref = diag(g_X, g_Y) with g_X, g_Y real amplitudes (phases=0)
- full: J_ref = 2x2 complex (8 real params)
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from .array_ops import xp, to_cpu, to_gpu


class SingleChainSolver:
    """
    Jones calibration via single-chain direct solutions.
    """

    def __init__(self, ref_antenna, mode='diagonal'):
        """
        Parameters
        ----------
        ref_antenna : int
            Reference antenna index
        mode : str
            'phase_only', 'diagonal', or 'full'
        """
        self.ref_antenna = ref_antenna
        self.mode = mode

    def solve(self, vis_obs, vis_model, antenna1, antenna2, chain_path, max_iter=100):
        """
        Solve for Jones matrices.

        Parameters
        ----------
        vis_obs : ndarray, shape (n_bl, 2, 2)
            Observed visibilities
        vis_model : ndarray, shape (n_bl, 2, 2)
            Model visibilities
        antenna1, antenna2 : ndarray, shape (n_bl,)
            Antenna indices
        chain_path : list of tuples
            Chain path from build_chain()
        max_iter : int
            Max optimization iterations

        Returns
        -------
        jones : ndarray, shape (n_ant, 2, 2)
            Solved Jones matrices
        info : dict
            Diagnostics
        """
        n_ant = max(antenna1.max(), antenna2.max()) + 1

        # Move to GPU if available
        vis_obs_gpu = to_gpu(vis_obs)
        vis_model_gpu = to_gpu(vis_model)

        # Build baseline lookup
        bl_map = self._build_baseline_map(vis_obs_gpu, vis_model_gpu, antenna1, antenna2)

        if self.mode == 'phase_only':
            jones, info = self._solve_phase_only(bl_map, n_ant, chain_path)
        elif self.mode == 'diagonal':
            jones, info = self._solve_diagonal(bl_map, n_ant, chain_path, max_iter)
        else:  # full
            jones, info = self._solve_full(bl_map, n_ant, chain_path, max_iter)

        return to_cpu(jones), info

    def _build_baseline_map(self, vis_obs, vis_model, antenna1, antenna2):
        """Build baseline lookup dictionary."""
        bl_map = {}
        for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
            bl_map[(a1, a2)] = {
                'V_obs': vis_obs[idx],
                'V_model': vis_model[idx],
                'idx': idx
            }
            # Conjugate transpose for reverse direction
            bl_map[(a2, a1)] = {
                'V_obs': vis_obs[idx].conj().T,
                'V_model': vis_model[idx].conj().T,
                'idx': idx
            }
        return bl_map

    def _chain_solutions(self, J_ref, bl_map, n_ant, chain_path):
        """
        Chain through array using direct matrix solutions.

        Parameters
        ----------
        J_ref : ndarray (2, 2)
            Reference antenna Jones matrix
        bl_map : dict
            Baseline lookup
        n_ant : int
        chain_path : list of tuples

        Returns
        -------
        jones : ndarray (n_ant, 2, 2)
        """
        jones = xp.zeros((n_ant, 2, 2), dtype=complex)
        jones[self.ref_antenna] = J_ref

        is_diagonal = (self.mode == 'phase_only' or self.mode == 'diagonal')

        for (ant_known, ant_unknown) in chain_path:
            if (ant_known, ant_unknown) not in bl_map:
                jones[ant_unknown] = xp.eye(2)
                continue

            bl = bl_map[(ant_known, ant_unknown)]
            V = bl['V_obs']
            M = bl['V_model']
            J_known = jones[ant_known]

            try:
                if is_diagonal:
                    # Diagonal: solve XX and YY independently
                    # V[p,p] = g_known[p,p] * M[p,p] * g_unknown[p,p]^*
                    # => g_unknown[p,p] = (V[p,p] / (g_known[p,p] * M[p,p]))^*
                    J_unknown = xp.zeros((2, 2), dtype=complex)
                    for p in [0, 1]:
                        if J_known[p, p] != 0 and M[p, p] != 0:
                            g_unknown_conj = V[p, p] / (J_known[p, p] * M[p, p])
                            J_unknown[p, p] = g_unknown_conj.conj()
                        else:
                            J_unknown[p, p] = 1.0
                else:
                    # Full: matrix inversion
                    # V = J_known @ M @ J_unknown†
                    # => J_unknown = (M^(-1) @ J_known^(-1) @ V)^†
                    J_unknown_dag = xp.linalg.inv(M) @ xp.linalg.inv(J_known) @ V
                    J_unknown = J_unknown_dag.conj().T
            except:
                J_unknown = xp.eye(2)

            jones[ant_unknown] = J_unknown

        return jones

    def _solve_phase_only(self, bl_map, n_ant, chain_path):
        """Phase-only: J_ref = I (fixed)."""
        J_ref = xp.eye(2, dtype=complex)
        jones = self._chain_solutions(J_ref, bl_map, n_ant, chain_path)

        return jones, {
            'mode': 'phase_only',
            'J_ref': to_cpu(J_ref),
            'iterations': 0,
            'cost': None
        }

    def _solve_diagonal(self, bl_map, n_ant, chain_path, max_iter):
        """
        Diagonal: optimize |g_X|, |g_Y| of J_ref.

        CRITICAL: Optimize XX and YY INDEPENDENTLY (no coupling)
        """

        # Optimize g_X (XX only)
        def objective_X(g_X_val):
            J_ref = xp.diag(xp.array([g_X_val + 0j, 1.0 + 0j], dtype=complex))
            jones = self._chain_solutions(J_ref, bl_map, n_ant, chain_path)

            cost = 0.0
            for (a1, a2), bl_data in bl_map.items():
                if a1 < a2:
                    V_obs = bl_data['V_obs']
                    M = bl_data['V_model']
                    V_pred_XX = jones[a1, 0, 0] * M[0, 0] * jones[a2, 0, 0].conj()
                    cost += float(to_cpu(xp.abs(V_obs[0, 0] - V_pred_XX)**2))
            return cost

        result_X = minimize_scalar(
            objective_X,
            bounds=(0.1, 5.0),
            method='bounded',
            options={'maxiter': max_iter}
        )
        g_X_opt = result_X.x

        # Optimize g_Y (YY only)
        def objective_Y(g_Y_val):
            J_ref = xp.diag(xp.array([g_X_opt + 0j, g_Y_val + 0j], dtype=complex))
            jones = self._chain_solutions(J_ref, bl_map, n_ant, chain_path)

            cost = 0.0
            for (a1, a2), bl_data in bl_map.items():
                if a1 < a2:
                    V_obs = bl_data['V_obs']
                    M = bl_data['V_model']
                    V_pred_YY = jones[a1, 1, 1] * M[1, 1] * jones[a2, 1, 1].conj()
                    cost += float(to_cpu(xp.abs(V_obs[1, 1] - V_pred_YY)**2))
            return cost

        result_Y = minimize_scalar(
            objective_Y,
            bounds=(0.1, 5.0),
            method='bounded',
            options={'maxiter': max_iter}
        )
        g_Y_opt = result_Y.x

        # Final solution
        J_ref_opt = xp.diag(xp.array([g_X_opt + 0j, g_Y_opt + 0j], dtype=complex))
        jones = self._chain_solutions(J_ref_opt, bl_map, n_ant, chain_path)

        return jones, {
            'mode': 'diagonal',
            'J_ref': to_cpu(J_ref_opt),
            'g_X': g_X_opt,
            'g_Y': g_Y_opt,
            'iterations': result_X.nit + result_Y.nit,
            'cost': result_X.fun + result_Y.fun
        }

    def _solve_full(self, bl_map, n_ant, chain_path, max_iter):
        """Full Jones: optimize 8 real parameters."""

        def objective(x):
            # x = [Re(J00), Im(J00), Re(J01), Im(J01), Re(J10), Im(J10), Re(J11), Im(J11)]
            J_ref = xp.array([[x[0] + 1j*x[1], x[2] + 1j*x[3]],
                              [x[4] + 1j*x[5], x[6] + 1j*x[7]]], dtype=complex)
            jones = self._chain_solutions(J_ref, bl_map, n_ant, chain_path)

            cost = 0.0
            for (a1, a2), bl_data in bl_map.items():
                if a1 < a2:
                    V_obs = bl_data['V_obs']
                    M = bl_data['V_model']
                    V_pred = jones[a1] @ M @ jones[a2].conj().T
                    cost += float(to_cpu(xp.sum(xp.abs(V_obs - V_pred)**2)))
            return cost

        # Initial guess: identity
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        bounds = [(-5, 5)] * 8

        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )

        J_ref_opt = xp.array([[result.x[0] + 1j*result.x[1], result.x[2] + 1j*result.x[3]],
                              [result.x[4] + 1j*result.x[5], result.x[6] + 1j*result.x[7]]], dtype=complex)
        jones = self._chain_solutions(J_ref_opt, bl_map, n_ant, chain_path)

        return jones, {
            'mode': 'full',
            'J_ref': to_cpu(J_ref_opt),
            'iterations': result.nit,
            'cost': result.fun,
            'success': result.success
        }


__all__ = ['SingleChainSolver']
