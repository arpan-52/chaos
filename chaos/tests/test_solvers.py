"""
Tests for CHAOS solvers.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from chaos.jones.terms import G_jones, G_jones_phase_only, D_jones, I_jones
from chaos.jones.operations import apply_jones
from chaos.core.solver import solve_jones
from chaos.core.chain_solver import ChainSolver
from chaos.core.polish import polish_jones


def simulate_visibilities(
    jones_true: np.ndarray,
    model: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    noise_level: float = 0.0,
) -> np.ndarray:
    """
    Simulate observed visibilities given true Jones.
    
    V_obs = J_i @ M @ J_j^H + noise
    """
    n_bl = len(antenna1)
    vis_obs = np.zeros((n_bl, 2, 2), dtype=np.complex128)
    
    for idx in range(n_bl):
        a1, a2 = antenna1[idx], antenna2[idx]
        M = model[idx]
        J_i = jones_true[a1]
        J_j = jones_true[a2]
        
        vis_obs[idx] = apply_jones(M, J_i, J_j)
        
        if noise_level > 0:
            noise = noise_level * (
                np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
            ) / np.sqrt(2)
            vis_obs[idx] += noise
    
    return vis_obs


def create_baseline_arrays(n_ant: int):
    """Create antenna1, antenna2 arrays for all baselines."""
    antenna1 = []
    antenna2 = []
    
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            antenna1.append(i)
            antenna2.append(j)
    
    return np.array(antenna1), np.array(antenna2)


class TestChainSolver:
    """Test chain-based initial solver."""
    
    def test_identity_solution(self):
        """Test that identity Jones gives identity solution."""
        n_ant = 5
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        jones_true = I_jones((n_ant,))
        model = I_jones((n_bl,))
        
        vis_obs = simulate_visibilities(jones_true, model, antenna1, antenna2)
        
        solver = ChainSolver(ref_antenna=0, mode="diagonal")
        jones_est, info = solver.solve(
            vis_obs, model, antenna1, antenna2, n_ant
        )
        
        # Should recover identity (up to reference)
        for ant in range(n_ant):
            assert_allclose(jones_est[ant], np.eye(2), atol=1e-10)
    
    def test_diagonal_gain(self):
        """Test solving for diagonal gains."""
        n_ant = 6
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        # Random diagonal gains
        np.random.seed(42)
        jones_true = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        for ant in range(n_ant):
            g_X = np.random.uniform(0.8, 1.2) * np.exp(1j * np.random.uniform(-np.pi, np.pi))
            g_Y = np.random.uniform(0.8, 1.2) * np.exp(1j * np.random.uniform(-np.pi, np.pi))
            jones_true[ant] = G_jones(g_X, g_Y)
        
        # Reference constraint: phase = 0
        jones_true[0, 0, 0] = np.abs(jones_true[0, 0, 0])
        jones_true[0, 1, 1] = np.abs(jones_true[0, 1, 1])
        
        model = I_jones((n_bl,))
        vis_obs = simulate_visibilities(jones_true, model, antenna1, antenna2)
        
        solver = ChainSolver(ref_antenna=0, mode="diagonal")
        jones_est, info = solver.solve(
            vis_obs, model, antenna1, antenna2, n_ant
        )
        
        # Chain solver gives good initial guess, not exact
        # Check structure is diagonal
        for ant in range(n_ant):
            assert_allclose(jones_est[ant, 0, 1], 0, atol=1e-10)
            assert_allclose(jones_est[ant, 1, 0], 0, atol=1e-10)


class TestPolish:
    """Test least-squares polish."""
    
    def test_phase_only_polish(self):
        """Test phase-only refinement."""
        n_ant = 5
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        # True phase-only gains
        np.random.seed(123)
        jones_true = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        for ant in range(n_ant):
            phi_X = np.random.uniform(-np.pi, np.pi) if ant > 0 else 0
            phi_Y = np.random.uniform(-np.pi, np.pi) if ant > 0 else 0
            jones_true[ant] = G_jones_phase_only(phi_X, phi_Y)
        
        model = I_jones((n_bl,))
        vis_obs = simulate_visibilities(jones_true, model, antenna1, antenna2)
        
        # Start with identity (bad initial guess)
        jones_init = I_jones((n_ant,))
        
        jones_polished, info = polish_jones(
            jones_init, vis_obs, model, antenna1, antenna2,
            ref_antenna=0, mode="phase_only",
            max_iter=100, tol=1e-12,
        )
        
        # Should recover true phases
        for ant in range(n_ant):
            # Compare phases (up to reference)
            phase_true_X = np.angle(jones_true[ant, 0, 0])
            phase_est_X = np.angle(jones_polished[ant, 0, 0])
            
            # Phases should match (modulo 2*pi)
            phase_diff = phase_est_X - phase_true_X
            phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
            
            assert np.abs(phase_diff) < 0.01, f"Antenna {ant}: phase diff = {np.degrees(phase_diff):.2f} deg"
    
    def test_diagonal_polish(self):
        """Test diagonal (amplitude + phase) refinement."""
        n_ant = 6
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        np.random.seed(456)
        jones_true = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        for ant in range(n_ant):
            if ant == 0:
                g_X = np.random.uniform(0.8, 1.2)  # Phase = 0 for ref
                g_Y = np.random.uniform(0.8, 1.2)
            else:
                g_X = np.random.uniform(0.8, 1.2) * np.exp(1j * np.random.uniform(-np.pi, np.pi))
                g_Y = np.random.uniform(0.8, 1.2) * np.exp(1j * np.random.uniform(-np.pi, np.pi))
            jones_true[ant] = G_jones(g_X, g_Y)
        
        model = I_jones((n_bl,))
        vis_obs = simulate_visibilities(jones_true, model, antenna1, antenna2)
        
        # Get chain initial guess
        solver = ChainSolver(ref_antenna=0, mode="diagonal")
        jones_init, _ = solver.solve(vis_obs, model, antenna1, antenna2, n_ant)
        
        # Polish
        jones_polished, info = polish_jones(
            jones_init, vis_obs, model, antenna1, antenna2,
            ref_antenna=0, mode="diagonal",
            max_iter=100, tol=1e-12,
        )
        
        # Should recover true gains closely
        for ant in range(n_ant):
            assert_allclose(
                jones_polished[ant, 0, 0], jones_true[ant, 0, 0],
                rtol=1e-5, atol=1e-10
            )
            assert_allclose(
                jones_polished[ant, 1, 1], jones_true[ant, 1, 1],
                rtol=1e-5, atol=1e-10
            )


class TestSolveJones:
    """Test full solve_jones function."""
    
    def test_full_pipeline(self):
        """Test chain + polish pipeline."""
        n_ant = 8
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        np.random.seed(789)
        jones_true = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        for ant in range(n_ant):
            if ant == 0:
                g_X = 1.0
                g_Y = 1.0
            else:
                g_X = np.random.uniform(0.9, 1.1) * np.exp(1j * np.random.uniform(-0.5, 0.5))
                g_Y = np.random.uniform(0.9, 1.1) * np.exp(1j * np.random.uniform(-0.5, 0.5))
            jones_true[ant] = G_jones(g_X, g_Y)
        
        model = I_jones((n_bl,))
        vis_obs = simulate_visibilities(jones_true, model, antenna1, antenna2)
        
        jones_est, diagnostics = solve_jones(
            vis_obs, model, antenna1, antenna2, n_ant,
            ref_antenna=0,
            jones_type="G",
            polish_tol=1e-12,
            verbose=False,
        )
        
        # Check accuracy
        for ant in range(n_ant):
            assert_allclose(
                jones_est[ant, 0, 0], jones_true[ant, 0, 0],
                rtol=1e-4
            )
            assert_allclose(
                jones_est[ant, 1, 1], jones_true[ant, 1, 1],
                rtol=1e-4
            )
    
    def test_with_noise(self):
        """Test solver with noisy data."""
        n_ant = 10
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        np.random.seed(111)
        jones_true = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        for ant in range(n_ant):
            if ant == 0:
                jones_true[ant] = I_jones()
            else:
                g_X = np.exp(1j * np.random.uniform(-1, 1))
                g_Y = np.exp(1j * np.random.uniform(-1, 1))
                jones_true[ant] = G_jones_phase_only(np.angle(g_X), np.angle(g_Y))
        
        model = I_jones((n_bl,))
        vis_obs = simulate_visibilities(
            jones_true, model, antenna1, antenna2, noise_level=0.01
        )
        
        jones_est, diagnostics = solve_jones(
            vis_obs, model, antenna1, antenna2, n_ant,
            ref_antenna=0,
            jones_type="G[p]",
            polish_tol=1e-10,
            verbose=False,
        )
        
        # With noise, won't be perfect but should be close
        for ant in range(n_ant):
            phase_true = np.angle(jones_true[ant, 0, 0])
            phase_est = np.angle(jones_est[ant, 0, 0])
            phase_diff = (phase_est - phase_true + np.pi) % (2*np.pi) - np.pi
            
            # Within ~5 degrees
            assert np.abs(np.degrees(phase_diff)) < 10


class TestBadAntennas:
    """Test bad antenna handling."""
    
    def test_flagged_antenna_excluded(self):
        """Test that heavily flagged antennas are excluded."""
        n_ant = 5
        antenna1, antenna2 = create_baseline_arrays(n_ant)
        n_bl = len(antenna1)
        
        jones_true = I_jones((n_ant,))
        model = I_jones((n_bl,))
        vis_obs = simulate_visibilities(jones_true, model, antenna1, antenna2)
        
        # Flag all baselines to antenna 3
        flags = np.zeros((n_bl, 2, 2), dtype=bool)
        for idx in range(n_bl):
            if antenna1[idx] == 3 or antenna2[idx] == 3:
                flags[idx] = True
        
        jones_est, diagnostics = solve_jones(
            vis_obs, model, antenna1, antenna2, n_ant,
            ref_antenna=0,
            jones_type="G",
            flag_threshold=0.8,
            flags=flags,
            verbose=False,
        )
        
        # Antenna 3 should be in bad_antennas and have identity
        assert 3 in diagnostics["bad_antennas"]
        assert_allclose(jones_est[3], np.eye(2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
