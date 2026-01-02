"""
Tests for Jones matrix definitions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from chaos.jones.terms import (
    I_jones, K_jones, K_jones_from_delay, KCROSS_jones,
    B_jones, G_jones, G_jones_phase_only,
    P_jones_linear, P_jones_circular, D_jones, X_jones,
)
from chaos.jones.operations import (
    apply_jones, unapply_jones, composite_jones,
    jones_determinant, jones_inverse, jones_hermitian,
)


class TestIdentity:
    """Test identity Jones matrix."""
    
    def test_shape_scalar(self):
        I = I_jones()
        assert I.shape == (2, 2)
        assert_allclose(I, np.eye(2))
    
    def test_shape_broadcast(self):
        I = I_jones((5,))
        assert I.shape == (5, 2, 2)
        for i in range(5):
            assert_allclose(I[i], np.eye(2))
    
    def test_shape_2d(self):
        I = I_jones((3, 4))
        assert I.shape == (3, 4, 2, 2)


class TestKJones:
    """Test delay Jones matrices."""
    
    def test_zero_phase(self):
        K = K_jones(0.0, 0.0)
        assert_allclose(K, np.eye(2))
    
    def test_diagonal(self):
        K = K_jones(np.pi/4, np.pi/2)
        assert K[0, 1] == 0
        assert K[1, 0] == 0
        assert_allclose(np.abs(K[0, 0]), 1.0)
        assert_allclose(np.abs(K[1, 1]), 1.0)
    
    def test_from_delay(self):
        tau = 1e-9  # 1 ns
        freq = 1e9  # 1 GHz
        K = K_jones_from_delay(tau, tau, freq)
        # Phase = 2*pi*freq*tau = 2*pi
        assert_allclose(K, np.eye(2), atol=1e-10)
    
    def test_broadcast(self):
        phases = np.linspace(0, np.pi, 10)
        K = K_jones(phases, phases)
        assert K.shape == (10, 2, 2)


class TestKCROSS:
    """Test cross-hand delay Jones."""
    
    def test_structure(self):
        KC = KCROSS_jones(1e-9, 1e9)
        assert KC[0, 0] == 1.0
        assert KC[0, 1] == 0
        assert KC[1, 0] == 0
        assert np.abs(KC[1, 1]) == pytest.approx(1.0)


class TestGJones:
    """Test gain Jones matrices."""
    
    def test_complex_gain(self):
        g_X = 1.5 * np.exp(1j * np.pi/6)
        g_Y = 1.2 * np.exp(1j * np.pi/4)
        G = G_jones(g_X, g_Y)
        
        assert G[0, 0] == g_X
        assert G[1, 1] == g_Y
        assert G[0, 1] == 0
        assert G[1, 0] == 0
    
    def test_phase_only(self):
        G = G_jones_phase_only(np.pi/4, np.pi/3)
        assert_allclose(np.abs(G[0, 0]), 1.0)
        assert_allclose(np.abs(G[1, 1]), 1.0)


class TestPJones:
    """Test parallactic angle Jones matrices."""
    
    def test_linear_identity(self):
        P = P_jones_linear(0.0)
        assert_allclose(P, np.eye(2))
    
    def test_linear_rotation(self):
        psi = np.pi/4
        P = P_jones_linear(psi)
        
        # Should be rotation matrix
        c = np.cos(psi)
        s = np.sin(psi)
        expected = np.array([[c, -s], [s, c]])
        assert_allclose(P, expected)
    
    def test_linear_orthogonal(self):
        psi = np.pi/6
        P = P_jones_linear(psi)
        
        # P @ P.T = I for rotation matrix
        assert_allclose(P @ P.T.conj(), np.eye(2), atol=1e-10)
    
    def test_circular_identity(self):
        P = P_jones_circular(0.0)
        assert_allclose(P, np.eye(2))
    
    def test_circular_diagonal(self):
        psi = np.pi/4
        P = P_jones_circular(psi)
        
        assert P[0, 1] == 0
        assert P[1, 0] == 0
        assert_allclose(P[0, 0], np.exp(-1j * psi))
        assert_allclose(P[1, 1], np.exp(1j * psi))


class TestDJones:
    """Test leakage Jones matrices."""
    
    def test_structure(self):
        d_X = 0.05 + 0.02j
        d_Y = 0.03 - 0.01j
        D = D_jones(d_X, d_Y)
        
        assert D[0, 0] == 1.0
        assert D[1, 1] == 1.0
        assert D[0, 1] == d_X
        assert D[1, 0] == d_Y
    
    def test_small_leakage(self):
        D = D_jones(0.01, 0.01)
        # Should be close to identity
        assert_allclose(D, np.eye(2), atol=0.02)


class TestXJones:
    """Test cross-hand phase Jones."""
    
    def test_structure(self):
        phi = np.pi/4
        X = X_jones(phi)
        
        assert X[0, 0] == 1.0
        assert X[0, 1] == 0
        assert X[1, 0] == 0
        assert_allclose(X[1, 1], np.exp(1j * phi))


class TestOperations:
    """Test Jones operations."""
    
    def test_apply_identity(self):
        V = np.array([[1+1j, 0.1], [0.1j, 1-0.5j]])
        I = I_jones()
        
        V_out = apply_jones(V, I, I)
        assert_allclose(V_out, V)
    
    def test_apply_unapply_inverse(self):
        V = np.array([[1+1j, 0.1], [0.1j, 1-0.5j]])
        J = G_jones(1.5*np.exp(1j*0.3), 1.2*np.exp(1j*0.5))
        
        V_corrupted = apply_jones(V, J, J)
        V_recovered = unapply_jones(V_corrupted, J, J)
        
        assert_allclose(V_recovered, V, rtol=1e-10)
    
    def test_composite_identity(self):
        I = I_jones()
        C = composite_jones([I, I, I])
        assert_allclose(C, I)
    
    def test_composite_order(self):
        G1 = G_jones(2.0, 1.0)
        G2 = G_jones(1.0, 2.0)
        
        # C = G2 @ G1 (signal order: G1 then G2)
        C = composite_jones([G1, G2])
        expected = G2 @ G1
        assert_allclose(C, expected)
    
    def test_jones_inverse(self):
        J = G_jones(1.5*np.exp(1j*0.3), 1.2*np.exp(1j*0.5))
        J_inv = jones_inverse(J)
        
        assert_allclose(J @ J_inv, np.eye(2), atol=1e-10)
    
    def test_jones_hermitian(self):
        J = G_jones(1.5*np.exp(1j*0.3), 1.2*np.exp(1j*0.5))
        J_H = jones_hermitian(J)
        
        assert_allclose(J_H, J.conj().T)


class TestReferenceConstraints:
    """Test reference antenna constraints."""
    
    def test_gain_ref_phase_zero(self):
        """Reference antenna should have phase = 0 for both pols."""
        # Simulate solving with ref_ant = 0
        # After solution, ref antenna phases should be 0
        G_ref = G_jones(1.0, 1.0)  # Amp=1, phase=0
        
        assert_allclose(np.angle(G_ref[0, 0]), 0)
        assert_allclose(np.angle(G_ref[1, 1]), 0)
    
    def test_d_term_ref_constraint(self):
        """For unpolarized cal, ref antenna d_X = 0."""
        D_ref = D_jones(0.0, 0.02)  # d_X = 0 for ref
        
        assert D_ref[0, 1] == 0  # d_X = 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
