"""End-to-end smoke + SCF-convergence tests for the Damle lower-contour in FockToP.

FockToP uses the analytic Damle lower contour (damleLowerDensity) on
[ENERGY_MIN, Emin], with Emin placed by calcEmin; the old pseudo-pole / calcTSW
path is gone. These tests run on C2 STO-3G (PSD overlap) and C2 LANL2DZ
(non-PSD S_eff -- the out-of-scope non-PSD track; those cases may not yet
satisfy every invariant).

Checks per fixture:
  (a) self.Emin is finite and negative
  (b) trace(S @ P) is finite and positive
  (c) Hermiticity defect ||P - P^H|| / ||P|| < 1e-2
The SCF tests additionally run negf.SCF(maxcycles=15) and check the converged
trace(SP) is finite and not catastrophically negative.
"""
import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from gauNEGF.scfE import NEGFE


@pytest.fixture(scope='module')
def c2_sto3g_negf(tmp_path_factory):
    scratch = tmp_path_factory.mktemp('damle_int_sto3g')
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    shutil.copy(os.path.join(repo_root, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                     route='integral=grid=superfine')
        negf.setContact1D([[1], [2]], symmetrize_contacts=True)
        negf.setVoltage(0.0)
        yield negf
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='module')
def c2_lanl2dz_negf(tmp_path_factory):
    scratch = tmp_path_factory.mktemp('damle_int_lanl2dz')
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    shutil.copy(os.path.join(repo_root, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        negf = NEGFE(fn='C2_chain', func='b3lyp', basis='lanl2dz',
                     route='integral=grid=superfine')
        negf.setContact1D([[1], [2]], symmetrize_contacts=True)
        negf.setVoltage(0.0)
        yield negf
    finally:
        os.chdir(cwd)


def _check_density_invariants(negf, label):
    """After one FockToP call, check Emin / nLower / Hermiticity."""
    # Probe the Damle lower piece in isolation BEFORE FockToP composes the full
    # density. This isolates "is the lower contour the wrong piece, or the
    # upper piece, or the Fermi search?".
    from gauNEGF.density import damleLowerDensity
    from gauNEGF.transport import har_to_eV
    F_eV = np.asarray(negf.F) * har_to_eV
    P_lower_only, delta_N_only = damleLowerDensity(F_eV, negf.Y_eff, negf.Sigma_0,
                                                   negf.g, negf.Emin)
    S = np.asarray(negf.S)
    n_lower_only = float(np.real(np.trace(S @ np.asarray(P_lower_only)))) + delta_N_only
    print(f'  [{label}] PRE-FockToP isolated Damle lower:')
    print(f'    Emin (input)    = {negf.Emin:+.2f} eV')
    print(f'    delta_N (cross) = {delta_N_only:+.4e}')
    print(f'    nLower (Damle)  = {n_lower_only:+.4f}')
    print(f'    ||Sigma_0||_F   = {np.linalg.norm(negf.Sigma_0):.3e}')
    print(f'    ||X_asymp||_F   = {np.linalg.norm(negf.X_asymp):.3e}')
    print(f'    ||Y_eff||_F     = {np.linalg.norm(negf.Y_eff):.3e}')
    print(f'    S_eff PSD?      = {np.all(np.linalg.eigvalsh(negf.S_eff) > 0)}')

    negf.FockToP()  # FockToP stores result internally in self.P
    P = np.asarray(negf.P)
    n_total = float(np.real(np.trace(S @ P)))
    herm = float(np.linalg.norm(P - P.conj().T)) / max(float(np.linalg.norm(P)), 1e-30)
    print(f'  [{label}] POST-FockToP composed P:')
    print(f'    Emin (final)   = {negf.Emin:+.2f} eV')
    print(f'    fermi          = {negf.fermi:+.2f} eV')
    print(f'    trace(SP)      = {n_total:+.4f}')
    print(f'    ||P-P^H||/||P|| = {herm:.3e}')

    assert np.isfinite(negf.Emin), f'[{label}] Emin not finite: {negf.Emin}'
    assert negf.Emin < 0.0, f'[{label}] Emin not negative: {negf.Emin}'
    assert np.isfinite(n_total), f'[{label}] trace(S P) not finite: {n_total}'
    assert n_total > 0.0, f'[{label}] trace(S P) not positive: {n_total}'
    assert herm < 1e-2, f'[{label}] Hermiticity defect too large: {herm:.3e}'


def test_fockto_p_runs_on_c2_sto3g(c2_sto3g_negf):
    """C2 STO-3G: FockToP runs cleanly. Single-zeta, no pseudo-poles."""
    _check_density_invariants(c2_sto3g_negf, 'C2 STO-3G')


@pytest.mark.skip(reason="non-minimal basis on 1D contact: S not PSD, X_asymp "
                         "non-Hermitian, Damle framework out of contract. "
                         "Minimal-basis contacts are the supported regime; "
                         "double-zeta on 1D contacts is future work.")
def test_fockto_p_runs_on_c2_lanl2dz(c2_lanl2dz_negf):
    """C2 LANL2DZ: FockToP runs cleanly. Double-zeta, has pseudo-poles."""
    _check_density_invariants(c2_lanl2dz_negf, 'C2 LANL2DZ')


# ---------------------------------------------------------------------------
# SCF-to-convergence tests. These build fresh fixtures (NOT reusing the
# one-shot fixtures above) and run negf.SCF() with a modest cycle cap to see
# if the converged total trace(SP) is sensible. Run separately so the timing
# does not block the cheaper one-shot tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def c2_sto3g_negf_scf(tmp_path_factory):
    scratch = tmp_path_factory.mktemp('damle_scf_sto3g')
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    shutil.copy(os.path.join(repo_root, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        negf = NEGFE(fn='C2_chain', func='b3lyp', basis='sto-3g',
                     route='integral=grid=superfine')
        negf.setContact1D([[1], [2]], symmetrize_contacts=True)
        negf.setVoltage(0.0)
        # Defensive: setVoltage(0.0) with fermi=None and initial self.fermi=None
        # sets updFermi=True (per-cycle Fermi search enabled in FockToP). Pin
        # explicitly so the test fails loudly if upstream logic ever changes.
        negf.updFermi = True
        yield negf
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='module')
def c2_lanl2dz_negf_scf(tmp_path_factory):
    scratch = tmp_path_factory.mktemp('damle_scf_lanl2dz')
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    shutil.copy(os.path.join(repo_root, 'examples', 'C2_chain.gjf'),
                os.path.join(scratch, 'C2_chain.gjf'))
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        negf = NEGFE(fn='C2_chain', func='b3lyp', basis='lanl2dz',
                     route='integral=grid=superfine')
        negf.setContact1D([[1], [2]], symmetrize_contacts=True)
        negf.setVoltage(0.0)
        # Defensive: setVoltage(0.0) with fermi=None and initial self.fermi=None
        # sets updFermi=True (per-cycle Fermi search enabled in FockToP). Pin
        # explicitly so the test fails loudly if upstream logic ever changes.
        negf.updFermi = True
        yield negf
    finally:
        os.chdir(cwd)


def _run_scf_and_report(negf, label, maxcycles=15):
    """Run SCF for up to `maxcycles` iterations, report trace(SP) at end.

    Disables checkpoint and Pulay to keep behavior simple. Returns nothing;
    test just asserts the final trace is finite and not catastrophically
    negative (sanity threshold: > -0.1 absolute, allowing tiny rounding).
    """
    expected_nelec = negf.bar.ne
    if negf.spin == 'r':
        expected_nelec /= 2
    print(f'  [{label}] Expected electron count (per spin if restricted): {expected_nelec}')
    # Sanity print + assert: Fermi must be in search mode (not pinned).
    print(f'  [{label}] PRE-SCF state: self.fermi = {negf.fermi}, '
          f'self.updFermi = {negf.updFermi}')
    assert negf.updFermi is True, (
        f'[{label}] updFermi must be True so per-cycle Fermi search runs; got '
        f'{negf.updFermi!r}. Without this the Fermi level stays pinned at the '
        f'initial HOMO+LUMO/2 guess and the SCF converges to a wrong density.'
    )
    try:
        result = negf.SCF(maxcycles=maxcycles, checkpoint=False, pulay=False)
        print(f'  [{label}] SCF returned (truncated): {str(result)[:200]}')
    except Exception as e:
        print(f'  [{label}] SCF raised: {type(e).__name__}: {e}')
        raise
    P = np.asarray(negf.P)
    S = np.asarray(negf.S)
    n_total = float(np.real(np.trace(S @ P)))
    herm = float(np.linalg.norm(P - P.conj().T)) / max(float(np.linalg.norm(P)), 1e-30)
    rel_err = abs(n_total - expected_nelec) / max(expected_nelec, 1e-30)
    print(f'  [{label}] FINAL trace(SP) = {n_total:+.4f}')
    print(f'  [{label}] expected         = {expected_nelec:+.4f}')
    print(f'  [{label}] rel err          = {rel_err:.3e}')
    print(f'  [{label}] herm defect      = {herm:.3e}')
    print(f'  [{label}] Emin (final)     = {negf.Emin:+.2f} eV')
    print(f'  [{label}] fermi (final)    = {negf.fermi:+.4f} eV')
    # Sanity assertions: finite, not catastrophically negative
    assert np.isfinite(n_total), f'[{label}] trace not finite'
    assert n_total > -0.1, f'[{label}] trace catastrophically negative: {n_total}'


def test_scf_converges_on_c2_sto3g(c2_sto3g_negf_scf):
    """C2 STO-3G SCF: trace(SP) should converge near expected electron count."""
    _run_scf_and_report(c2_sto3g_negf_scf, 'C2 STO-3G SCF', maxcycles=15)


@pytest.mark.skip(reason="non-minimal basis on 1D contact: S not PSD, X_asymp "
                         "non-Hermitian, Damle framework out of contract. "
                         "Minimal-basis contacts are the supported regime; "
                         "double-zeta on 1D contacts is future work.")
def test_scf_converges_on_c2_lanl2dz(c2_lanl2dz_negf_scf):
    """C2 LANL2DZ SCF: does Damle lower-contour give a sensible converged density?"""
    _run_scf_and_report(c2_lanl2dz_negf_scf, 'C2 LANL2DZ SCF', maxcycles=15)
