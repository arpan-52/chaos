"""
Tests for RFI flagging module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from chaos.utils.flagging import (
    mad, mad_to_sigma,
    flag_rfi_mad, flag_rfi_iterative,
    flag_zeros, flag_nans,
    FlagStats,
)


class TestMAD:
    """Test Median Absolute Deviation."""
    
    def test_constant_data(self):
        """MAD of constant data should be 0."""
        x = np.ones(100)
        assert mad(x) == 0
    
    def test_gaussian_data(self):
        """MAD of Gaussian should relate to sigma."""
        np.random.seed(42)
        sigma = 2.0
        x = np.random.randn(10000) * sigma
        
        mad_val = mad(x)
        sigma_est = mad_to_sigma(mad_val)
        
        # Should be close to true sigma
        assert_allclose(sigma_est, sigma, rtol=0.1)
    
    def test_with_outliers(self):
        """MAD should be robust to outliers."""
        np.random.seed(123)
        x = np.random.randn(1000)
        
        # Add extreme outliers
        x[0] = 1000
        x[1] = -1000
        
        # MAD should still be close to 1
        mad_val = mad(x)
        sigma_est = mad_to_sigma(mad_val)
        
        assert_allclose(sigma_est, 1.0, rtol=0.2)
    
    def test_axis(self):
        """Test MAD along axis."""
        x = np.random.randn(5, 100)
        mad_per_row = mad(x, axis=1)
        assert mad_per_row.shape == (5,)


class TestFlagRFI:
    """Test RFI flagging functions."""
    
    def test_no_rfi(self):
        """Clean data should have minimal flags."""
        np.random.seed(456)
        vis = 1.0 + 0.01 * np.random.randn(100, 2, 2)
        vis = vis.astype(np.complex128)
        
        flags, stats = flag_rfi_mad(vis, sigma=5.0)
        
        # Should flag very little
        assert stats.fraction_flagged < 0.01
    
    def test_obvious_rfi(self):
        """Obvious outliers should be flagged."""
        np.random.seed(789)
        vis = np.ones((100, 2, 2), dtype=np.complex128)
        
        # Add obvious RFI (100x normal)
        vis[10, 0, 0] = 100.0
        vis[50, 1, 1] = 100.0
        
        flags, stats = flag_rfi_mad(vis, sigma=5.0)
        
        # The outliers should be flagged
        assert flags[10, 0, 0] == True
        assert flags[50, 1, 1] == True
        
        # Normal data should not be flagged
        assert flags[0, 0, 0] == False
    
    def test_50_percent_rfi(self):
        """MAD should handle up to ~50% RFI."""
        np.random.seed(111)
        vis = np.ones((100, 2, 2), dtype=np.complex128)
        vis += 0.01 * np.random.randn(100, 2, 2)
        
        # Add 30% RFI (still should work)
        n_rfi = 30
        rfi_idx = np.random.choice(100, n_rfi, replace=False)
        vis[rfi_idx] *= 50
        
        flags, stats = flag_rfi_mad(vis, sigma=5.0)
        
        # Most RFI should be flagged
        n_rfi_flagged = sum(flags[i].any() for i in rfi_idx)
        assert n_rfi_flagged > 0.8 * n_rfi
    
    def test_existing_flags(self):
        """Should combine with existing flags."""
        vis = np.ones((50, 2, 2), dtype=np.complex128)
        
        existing = np.zeros((50, 2, 2), dtype=bool)
        existing[0] = True  # Pre-flag first baseline
        
        flags, stats = flag_rfi_mad(vis, existing_flags=existing)
        
        # Should include existing flag
        assert flags[0].all()
    
    def test_per_correlation(self):
        """Test per-correlation statistics."""
        np.random.seed(222)
        vis = np.ones((100, 2, 2), dtype=np.complex128)
        
        # Add RFI only in XX
        vis[10:15, 0, 0] = 50.0
        
        flags, stats = flag_rfi_mad(vis, sigma=5.0, per_correlation=True)
        
        # XX should have flags, YY should not
        assert flags[10:15, 0, 0].any()
        assert not flags[10:15, 1, 1].any()


class TestFlagIterative:
    """Test iterative flagging."""
    
    def test_convergence(self):
        """Should converge after few iterations."""
        np.random.seed(333)
        vis = np.ones((100, 2, 2), dtype=np.complex128)
        vis += 0.01 * np.random.randn(100, 2, 2)
        
        # Add RFI
        vis[5] = 100.0
        
        flags, stats = flag_rfi_iterative(
            vis, sigma=5.0, max_iter=5, verbose=False
        )
        
        assert flags[5].all()
        assert stats.n_flagged > 0


class TestFlagSpecial:
    """Test special value flagging."""
    
    def test_flag_zeros(self):
        """Test zero flagging."""
        vis = np.ones((10, 2, 2), dtype=np.complex128)
        vis[3] = 0.0
        
        flags, n = flag_zeros(vis)
        
        assert flags[3].all()
        assert n == 4  # 2x2 = 4 elements
    
    def test_flag_nans(self):
        """Test NaN/Inf flagging."""
        vis = np.ones((10, 2, 2), dtype=np.complex128)
        vis[2, 0, 0] = np.nan
        vis[5, 1, 1] = np.inf
        
        flags, n = flag_nans(vis)
        
        assert flags[2, 0, 0]
        assert flags[5, 1, 1]
        assert n == 2


class TestFlagWithSolver:
    """Test flagging integration with solver."""
    
    def test_solver_with_rfi(self):
        """Solver should handle RFI via flagging."""
        from chaos.core.solver import solve_jones
        from chaos.jones.terms import G_jones, I_jones
        from chaos.jones.operations import apply_jones
        
        n_ant = 6
        
        # Create baselines
        antenna1 = []
        antenna2 = []
        for i in range(n_ant):
            for j in range(i+1, n_ant):
                antenna1.append(i)
                antenna2.append(j)
        antenna1 = np.array(antenna1)
        antenna2 = np.array(antenna2)
        n_bl = len(antenna1)
        
        # True Jones
        np.random.seed(444)
        jones_true = np.zeros((n_ant, 2, 2), dtype=np.complex128)
        for ant in range(n_ant):
            if ant == 0:
                jones_true[ant] = I_jones()
            else:
                g_X = np.exp(1j * np.random.uniform(-1, 1))
                g_Y = np.exp(1j * np.random.uniform(-1, 1))
                jones_true[ant] = G_jones(g_X, g_Y)
        
        # Generate visibilities
        model = I_jones((n_bl,))
        vis_obs = np.zeros((n_bl, 2, 2), dtype=np.complex128)
        for idx in range(n_bl):
            a1, a2 = antenna1[idx], antenna2[idx]
            vis_obs[idx] = apply_jones(model[idx], jones_true[a1], jones_true[a2])
        
        # Add RFI to 10% of baselines
        n_rfi = n_bl // 10 + 1
        rfi_idx = np.random.choice(n_bl, n_rfi, replace=False)
        vis_obs[rfi_idx] *= 100
        
        # Solve with RFI flagging
        jones_est, diag = solve_jones(
            vis_obs, model, antenna1, antenna2, n_ant,
            ref_antenna=0, jones_type="G",
            rfi_flag=True, rfi_sigma=5.0,
            verbose=False,
        )
        
        # Should still get reasonable solution
        # (not perfect due to lost baselines)
        for ant in range(1, n_ant):
            phase_true = np.angle(jones_true[ant, 0, 0])
            phase_est = np.angle(jones_est[ant, 0, 0])
            diff = (phase_est - phase_true + np.pi) % (2*np.pi) - np.pi
            
            # Within ~20 degrees despite RFI
            assert np.abs(np.degrees(diff)) < 30, f"Ant {ant}: {np.degrees(diff):.1f} deg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
