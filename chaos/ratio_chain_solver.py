"""
Ratio Chain Solver for CHAOS.

Uses 2-baseline ratios to cancel intermediate Jones matrices.

Method:
- To solve J_u from known J_0, find best pivot antenna p
- Use baselines: (0,p) and (u,p)
- Form ratio: V_0p / V_up = J_0 M_0p M_up^-1 J_u^-1 (J_p cancels!)
- Solve algebraically for J_u
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from .array_ops import xp, to_cpu, to_gpu


class RatioChainSolver:
    """
    Jones calibration via ratio-chain method (2-baseline ratios).
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

    def solve(self, vis_obs, vis_model, antenna1, antenna2, quality_matrix, max_iter=100):
        """
        Solve for Jones matrices using ratio-chain method.

        Parameters
        ----------
        vis_obs : ndarray, shape (n_bl, 2, 2)
        vis_model : ndarray, shape (n_bl, 2, 2)
        antenna1, antenna2 : ndarray, shape (n_bl,)
        quality_matrix : ndarray, shape (n_ant, n_ant)
        max_iter : int

        Returns
        -------
        jones : ndarray, shape (n_ant, 2, 2)
        info : dict
        """
        n_ant = quality_matrix.shape[0]

        # Move to GPU
        vis_obs_gpu = to_gpu(vis_obs)
        vis_model_gpu = to_gpu(vis_model)

        # Build baseline lookup
        bl_map = self._build_baseline_map(vis_obs_gpu, vis_model_gpu, antenna1, antenna2)

        if self.mode == 'phase_only':
            jones, info = self._solve_phase_only(bl_map, quality_matrix, n_ant)
        elif self.mode == 'diagonal':
            jones, info = self._solve_diagonal(bl_map, quality_matrix, n_ant, max_iter)
        else:
            jones, info = self._solve_full(bl_map, quality_matrix, n_ant, max_iter)

        return to_cpu(jones), info

    def _build_baseline_map(self, vis_obs, vis_model, antenna1, antenna2):
        """Build baseline lookup."""
        bl_map = {}
        for idx, (a1, a2) in enumerate(zip(antenna1, antenna2)):
            bl_map[(a1, a2)] = {
                'V_obs': vis_obs[idx],
                'V_model': vis_model[idx],
                'idx': idx
            }
            bl_map[(a2, a1)] = {
                'V_obs': vis_obs[idx].conj().T,
                'V_model': vis_model[idx].conj().T,
                'idx': idx
            }
        return bl_map

    def _chain_from_J0(self, J_0, bl_map, quality_matrix, n_ant):
        """
        Chain using ratio method (2 baselines per antenna).

        For each unknown antenna u:
        1. Find best pivot p (maximizes combined quality of (ref,p) and (u,p))
        2. Compute ratio to cancel J_p
        3. Solve for J_u
        """
        jones = xp.zeros((n_ant, 2, 2), dtype=complex)
        jones[self.ref_antenna] = J_0

        solved = {self.ref_antenna}
        chain_info = []

        is_diagonal = (self.mode == 'phase_only' or self.mode == 'diagonal')

        while len(solved) < n_ant:
            best_ant = None
            best_pivot = None
            best_quality = -1

            # Find best (unknown, pivot) pair
            for ant_unknown in range(n_ant):
                if ant_unknown in solved:
                    continue

                for pivot in range(n_ant):
                    if pivot == ant_unknown:
                        continue

                    # Need both baselines
                    if (self.ref_antenna, pivot) not in bl_map:
                        continue
                    if (ant_unknown, pivot) not in bl_map:
                        continue

                    q_ref_pivot = quality_matrix[self.ref_antenna, pivot]
                    q_unknown_pivot = quality_matrix[ant_unknown, pivot]

                    combined_q = min(q_ref_pivot, q_unknown_pivot)

                    if combined_q > best_quality:
                        best_quality = combined_q
                        best_ant = ant_unknown
                        best_pivot = pivot

            if best_ant is None or best_quality <= 0:
                break

            # Extract Jones using ratio method
            bl_0p = bl_map[(self.ref_antenna, best_pivot)]
            bl_up = bl_map[(best_ant, best_pivot)]

            V_0p = bl_0p['V_obs']
            M_0p = bl_0p['V_model']
            V_up = bl_up['V_obs']
            M_up = bl_up['V_model']

            try:
                if is_diagonal:
                    # Diagonal: per polarization
                    J_u = xp.zeros((2, 2), dtype=complex)
                    for p in [0, 1]:
                        # V_0p[p,p] / V_up[p,p] = g_0[p] * M_0p[p,p] / (g_u[p] * M_up[p,p])
                        # g_u[p] = g_0[p] * M_0p[p,p] * V_up[p,p] / (M_up[p,p] * V_0p[p,p])
                        if V_0p[p, p] != 0 and M_up[p, p] != 0:
                            g_u = J_0[p, p] * M_0p[p, p] * V_up[p, p] / (M_up[p, p] * V_0p[p, p])
                            J_u[p, p] = g_u
                        else:
                            J_u[p, p] = 1.0
                else:
                    # Full matrix algebra
                    ratio = V_0p @ xp.linalg.inv(V_up)
                    J_u = xp.linalg.inv(ratio) @ J_0 @ M_0p @ xp.linalg.inv(M_up)
            except:
                J_u = xp.eye(2)

            jones[best_ant] = J_u
            solved.add(best_ant)
            chain_info.append({
                'unknown': best_ant,
                'pivot': best_pivot,
                'quality': best_quality
            })

        return jones, chain_info

    def _solve_phase_only(self, bl_map, quality_matrix, n_ant):
        """Phase-only: J_0 = I."""
        J_0 = xp.eye(2, dtype=complex)
        jones, chain_info = self._chain_from_J0(J_0, bl_map, quality_matrix, n_ant)

        return jones, {
            'mode': 'phase_only',
            'J_ref': to_cpu(J_0),
            'iterations': 0,
            'chain_info': chain_info
        }

    def _solve_diagonal(self, bl_map, quality_matrix, n_ant, max_iter):
        """Diagonal: optimize |g_X|, |g_Y|."""

        def objective_X(g_X_val):
            J_0 = xp.diag(xp.array([g_X_val + 0j, 1.0 + 0j], dtype=complex))
            jones, _ = self._chain_from_J0(J_0, bl_map, quality_matrix, n_ant)

            cost = 0.0
            for (a1, a2), bl_data in bl_map.items():
                if a1 < a2:
                    V_obs = bl_data['V_obs']
                    M = bl_data['V_model']
                    V_pred_XX = jones[a1, 0, 0] * M[0, 0] * jones[a2, 0, 0].conj()
                    cost += float(to_cpu(xp.abs(V_obs[0, 0] - V_pred_XX)**2))
            return cost

        result_X = minimize_scalar(objective_X, bounds=(0.1, 5.0), method='bounded',
                                   options={'maxiter': max_iter})
        g_X_opt = result_X.x

        def objective_Y(g_Y_val):
            J_0 = xp.diag(xp.array([g_X_opt + 0j, g_Y_val + 0j], dtype=complex))
            jones, _ = self._chain_from_J0(J_0, bl_map, quality_matrix, n_ant)

            cost = 0.0
            for (a1, a2), bl_data in bl_map.items():
                if a1 < a2:
                    V_obs = bl_data['V_obs']
                    M = bl_data['V_model']
                    V_pred_YY = jones[a1, 1, 1] * M[1, 1] * jones[a2, 1, 1].conj()
                    cost += float(to_cpu(xp.abs(V_obs[1, 1] - V_pred_YY)**2))
            return cost

        result_Y = minimize_scalar(objective_Y, bounds=(0.1, 5.0), method='bounded',
                                   options={'maxiter': max_iter})
        g_Y_opt = result_Y.x

        J_0_opt = xp.diag(xp.array([g_X_opt + 0j, g_Y_opt + 0j], dtype=complex))
        jones, chain_info = self._chain_from_J0(J_0_opt, bl_map, quality_matrix, n_ant)

        return jones, {
            'mode': 'diagonal',
            'J_ref': to_cpu(J_0_opt),
            'g_X': g_X_opt,
            'g_Y': g_Y_opt,
            'iterations': result_X.nit + result_Y.nit,
            'cost': result_X.fun + result_Y.fun,
            'chain_info': chain_info
        }

    def _solve_full(self, bl_map, quality_matrix, n_ant, max_iter):
        """Full Jones: 8 real parameters."""

        def objective(x):
            J_0 = xp.array([[x[0] + 1j*x[1], x[2] + 1j*x[3]],
                           [x[4] + 1j*x[5], x[6] + 1j*x[7]]], dtype=complex)
            jones, _ = self._chain_from_J0(J_0, bl_map, quality_matrix, n_ant)

            cost = 0.0
            for (a1, a2), bl_data in bl_map.items():
                if a1 < a2:
                    V_obs = bl_data['V_obs']
                    M = bl_data['V_model']
                    V_pred = jones[a1] @ M @ jones[a2].conj().T
                    cost += float(to_cpu(xp.sum(xp.abs(V_obs - V_pred)**2)))
            return cost

        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        bounds = [(-5, 5)] * 8

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-8})

        J_0_opt = xp.array([[result.x[0] + 1j*result.x[1], result.x[2] + 1j*result.x[3]],
                           [result.x[4] + 1j*result.x[5], result.x[6] + 1j*result.x[7]]], dtype=complex)
        jones, chain_info = self._chain_from_J0(J_0_opt, bl_map, quality_matrix, n_ant)

        return jones, {
            'mode': 'full',
            'J_ref': to_cpu(J_0_opt),
            'iterations': result.nit,
            'cost': result.fun,
            'success': result.success,
            'chain_info': chain_info
        }


__all__ = ['RatioChainSolver']
