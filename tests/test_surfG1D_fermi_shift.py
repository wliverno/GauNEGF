"""
Regression test: surfG1D Fermi-level rigid-band shift for contactFromFock=False.

When the device Fermi moves during SCF, aList/bList must behave as if rigid-shifted
by dFermi = mu - mu0. Algebraically, that's identical to evaluating g at E - dFermi
with the original alpha/beta. This test verifies the algebraic identity:

    sigma_shifted(E0 + dE, i)  ==  sigma_unshifted(E0, i)

for a contact whose internal alpha/beta have non-trivial overlap (Salpha, Sbeta != I, 0)
but whose tau coupling is orthogonal (stau = 0). For orthogonal coupling, t/bar_t do
NOT depend on E, so the full sigma identity holds (not just the surface-g part).

Bug manifests as non-integer transmission for non-orthogonal contacts (e.g. CNT).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from gauNEGF.surfG1D import surfG


def _build_device_and_contacts(N=6, n=1, Sbeta_val=0.2, E0=0.7):
    """Small 1D chain device with non-orthogonal 1-orbital contacts.

    Contact's intrinsic alpha = E0*Salpha (so its band sits centered at E0),
    beta = -1.0, Sbeta = Sbeta_val (non-trivial -> band position depends on E).
    """
    F = np.zeros((N, N), dtype=complex)
    S = np.eye(N, dtype=complex)
    for i in range(N - 1):
        F[i, i + 1] = -1.0
        F[i + 1, i] = -1.0

    Salpha = np.eye(n, dtype=complex)
    alpha = E0 * Salpha
    beta = -1.0 * np.eye(n, dtype=complex)
    Sbeta = Sbeta_val * np.eye(n, dtype=complex)

    tau = -1.0 * np.eye(n, dtype=complex)
    stau = np.zeros((n, n), dtype=complex)

    indsList = [[0], [N - 1]]
    return F, S, indsList, tau, stau, alpha, Salpha, beta, Sbeta, E0


def test_sigma_shift_identity_nonorthogonal_alpha():
    """sigma(E0+dE) with dFermi=dE must equal sigma(E0) with dFermi=0.

    Uses orthogonal coupling (stau=0) so t = -tau is E-independent, and the
    algebraic identity holds across the full sigma (not just surface g).
    """
    F, S, indsList, tau, stau, alpha, Sa, beta, Sb, E0 = _build_device_and_contacts()
    dE = 0.5

    # Reference: no shift. setF with mu = E0 captures fermi0 and leaves dFermi=0.
    gRef = surfG(F, S, indsList,
                 taus=[tau, tau], staus=[stau, stau],
                 alphas=[alpha, alpha], aOverlaps=[Sa, Sa],
                 betas=[beta, beta], bOverlaps=[Sb, Sb])
    gRef.setF(F, mu1=E0, mu2=E0)
    sigRef = np.array(gRef.sigma(E0, 0))

    # Shifted: same construction, but mu shifts to E0 + dE after capture.
    gShift = surfG(F, S, indsList,
                   taus=[tau, tau], staus=[stau, stau],
                   alphas=[alpha, alpha], aOverlaps=[Sa, Sa],
                   betas=[beta, beta], bOverlaps=[Sb, Sb])
    gShift.setF(F, mu1=E0, mu2=E0)            # capture fermi0 = E0
    gShift.setF(F, mu1=E0 + dE, mu2=E0 + dE)  # now dFermi = dE
    sigShift = np.array(gShift.sigma(E0 + dE, 0))

    np.testing.assert_allclose(
        sigShift, sigRef, atol=1e-8,
        err_msg="sigma(E0+dE) with dFermi=dE must equal sigma(E0) with dFermi=0 "
                "for orthogonal coupling and non-trivial contact Salpha/Sbeta.")


def test_sigma_changes_with_fermi_shift():
    """Sanity: shifting Fermi without recomputing sigma at shifted E must change sigma.

    Confirms the shift mechanism is actually doing something -- if dFermi were
    ignored, sigma(E0+dE) before and after the second setF would be identical.
    """
    F, S, indsList, tau, stau, alpha, Sa, beta, Sb, E0 = _build_device_and_contacts()
    dE = 0.5

    g = surfG(F, S, indsList,
              taus=[tau, tau], staus=[stau, stau],
              alphas=[alpha, alpha], aOverlaps=[Sa, Sa],
              betas=[beta, beta], bOverlaps=[Sb, Sb])
    g.setF(F, mu1=E0, mu2=E0)
    sig_before = np.array(g.sigma(E0 + dE, 0))

    g.setF(F, mu1=E0 + dE, mu2=E0 + dE)
    sig_after = np.array(g.sigma(E0 + dE, 0))

    max_diff = np.max(np.abs(sig_before - sig_after))
    assert max_diff > 1e-6, (
        f"sigma at E0+dE must change once dFermi is set; max diff = {max_diff:.2e}. "
        "Either the shift mechanism is missing or dE is too small to expose it.")


def test_dFermiList_and_fermi0List_present():
    """The shift bookkeeping attributes must exist after init."""
    F, S, indsList, tau, stau, alpha, Sa, beta, Sb, E0 = _build_device_and_contacts()
    g = surfG(F, S, indsList,
              taus=[tau, tau], staus=[stau, stau],
              alphas=[alpha, alpha], aOverlaps=[Sa, Sa],
              betas=[beta, beta], bOverlaps=[Sb, Sb])
    assert hasattr(g, 'dFermiList'), "surfG must expose dFermiList for rigid-shift bookkeeping"
    assert hasattr(g, 'fermi0List'), "surfG must expose fermi0List for reference-Fermi capture"
    assert len(g.dFermiList) == len(indsList)
    assert len(g.fermi0List) == len(indsList)
    # Before any setF, dFermi must be zero and fermi0 must be None.
    assert all(d == 0.0 for d in g.dFermiList)
    assert all(f is None for f in g.fermi0List)


def test_reference_snapshots_present():
    """aList0/bList0/aSList0/bSList0 must be snapshotted after init."""
    F, S, indsList, tau, stau, alpha, Sa, beta, Sb, E0 = _build_device_and_contacts()
    g = surfG(F, S, indsList,
              taus=[tau, tau], staus=[stau, stau],
              alphas=[alpha, alpha], aOverlaps=[Sa, Sa],
              betas=[beta, beta], bOverlaps=[Sb, Sb])
    for name in ('aList0', 'bList0', 'aSList0', 'bSList0'):
        assert hasattr(g, name), f"surfG must snapshot {name} at init"
        snap = getattr(g, name)
        assert len(snap) == len(indsList)
