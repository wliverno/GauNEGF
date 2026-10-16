"""setFock builds the density its Fock implies (base NEGF and NEGFE). Login-safe: synthetic shells, no Gaussian."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

import gauNEGF.scf as scf_mod
from gauNEGF.scf import NEGF

HAR = 27.211386


def _base_shell(F_eV):
    """Base NEGF with constant -0.1j self-energies on sites 0 and 3, Fermi pinned at 0.5 eV."""
    N = 4
    obj = NEGF.__new__(NEGF)
    obj.F = np.asarray(F_eV, dtype=float) / HAR
    obj.S = np.eye(N)
    obj.X = np.asarray(fractional_matrix_power(np.eye(N), -0.5))
    sig = np.zeros((N, N), dtype=complex); sig[0, 0] = -0.1j
    sig2 = np.zeros((N, N), dtype=complex); sig2[3, 3] = -0.1j
    obj.sigma12 = sig + sig2
    obj.Gam1 = 1j * (sig - sig.conj().T)
    obj.Gam2 = 1j * (sig2 - sig2.conj().T)
    obj.updFermi = False
    obj.fermi = obj.mu1 = obj.mu2 = 0.5
    obj.qV = 0.0
    obj.spin = 'r'
    obj.Eminf = -1e6
    obj.convLevel = 9999
    obj.bar = MagicMock()
    return obj


F0 = np.diag([-1.0, 0.0, 1.0, 2.0]) + 0.2 * (np.eye(4, k=1) + np.eye(4, k=-1))


def test_base_focktop_uses_self_F_not_gaussian():
    obj = _base_shell(F0)
    obj.FockToP()           # today: getFock(MagicMock bar) replaces self.F -> fails here
    assert np.allclose(obj.F, F0 / HAR)


def test_base_setfock_sets_density_and_stores_it(monkeypatch):
    stored = []
    monkeypatch.setattr(scf_mod, 'storeDen', lambda bar, P, spin: stored.append(np.array(P)))
    F1 = F0 - 0.3 * np.eye(4)
    obj = _base_shell(F0)
    obj.setFock(F1)
    twin = _base_shell(F1)
    twin.FockToP()
    assert np.allclose(obj.F, F1 / HAR)
    assert np.allclose(obj.P, twin.P)
    assert len(stored) == 1 and np.allclose(stored[0], obj.P)
    assert np.isclose(obj.nelec, 2 * np.trace(np.asarray(obj.P) @ obj.S).real)


def test_base_setfock_needs_voltage():
    obj = _base_shell(F0)
    del obj.mu1
    with pytest.raises(AssertionError, match='Voltage not set'):
        obj.setFock(F0)


import gauNEGF.scfE as scfE_mod
from test_fermi_freeze_guard import _build_focktop_shell


def _negfe_with_F(F_eV):
    obj = _build_focktop_shell()
    obj.F = np.asarray(F_eV, dtype=float) / HAR
    obj.g.setF(np.asarray(F_eV, dtype=float), obj.mu1, obj.mu2)
    return obj


def test_negfe_setfock_runs_search_on_new_fock_and_stores_density(monkeypatch):
    stored = []
    monkeypatch.setattr(scfE_mod, 'storeDen', lambda bar, P, spin: stored.append(np.array(P)))
    F1 = np.diag([-1.3, -0.3, 0.7, 1.7])
    obj = _build_focktop_shell()                  # starts on diag(-1, 0, 1, 2)
    obj.setFock(F1)
    twin = _negfe_with_F(F1)
    twin.FockToP()
    assert np.allclose(obj.F, F1 / HAR)
    assert np.allclose(np.asarray(obj.g.F), F1)   # contacts rebuilt from the new Fock
    assert np.isclose(obj.fermi, twin.fermi)      # search ran on the new Fock
    assert np.allclose(obj.P, twin.P)
    assert len(stored) == 1 and np.allclose(stored[0], obj.P)


def test_negfe_setfock_symmetrizes_contacts(monkeypatch):
    monkeypatch.setattr(scfE_mod, 'storeDen', lambda bar, P, spin: None)
    obj = _build_focktop_shell()
    obj._symmetrize_contacts = True
    obj.setFock(np.diag([-1.2, 0.0, 1.0, -0.8]))
    assert np.isclose(obj.F[0, 0], obj.F[3, 3]) and np.isclose(obj.F[0, 0] * HAR, -1.0)


def test_negfe_setfock_needs_setvoltage():
    obj = _build_focktop_shell()
    del obj.mu1
    with pytest.raises(RuntimeError, match='setVoltage'):
        obj.setFock(np.diag([-1.0, 0.0, 1.0, 2.0]))
