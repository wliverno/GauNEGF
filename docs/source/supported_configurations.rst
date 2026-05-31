==========================
Supported Configurations
==========================

Honest status of what works, what is limited, and what is not ready for
production use in the current release. Three tiers:

- **Green** -- tested and fully working.
- **Yellow** -- works in limited cases; convergence may be finicky.
- **Red** -- needs testing or has a known limitation.

The corresponding root-level reference is ``CAPABILITIES.md``.

Headline feature: non-orthogonal contact handling
==================================================

The big change in this release is proper treatment of non-orthogonal
device-contact coupling in the NEGF Green's function integrals. The
derivation is in
``docs/superpowers/specs/2026-03-18-cross-term-corrections-derivation.md``
and represents work going back to March; it touches every contact
implementation and both the density and DOS integrals.

The problem
-----------

When the device-contact overlap ``stau`` is non-zero, the contact
self-energy

.. math::

   \Sigma(E) = (E \cdot \mathrm{stau} - \tau) \, g_\mathrm{surf}(E)
              \, (E \cdot \mathrm{stau} - \tau)^\dagger

carries terms linear and quadratic in :math:`E`. The previous code
dropped the cross-terms generated when this expression is plugged into
the device Green's function integrals, silently treating non-orthogonal
contacts as orthogonal.

What is correct now
-------------------

``crossTermQ`` and ``crossTermQTot`` methods on every surfG implementation
return the energy-dependent cross-term correction. The density and DOS
integrals (``GrInt`` -> ``GrIntCross``, ``densityComplex``,
``damleLowerDensity``) accept and propagate the correction through the
contour. Result: correct charge / DOS / transmission on any
non-orthogonal contact setup.

Supporting work shipped at the same time
-----------------------------------------

- **Damle analytic lower contour** -- one eigendecomposition plus an
  analytic cross-term :math:`\delta N` replaces the prior deep-tail
  ``densityComplex`` call. Roughly 30-450x faster than the adaptive
  contour; ``damle_dN_warn`` flags cases where the lower contour holds
  non-trivial weight.
- **2-probe asymptotic Sigma fit** -- places the deep
  :math:`\Sigma_0` probes relative to the integration window instead
  of hardcoded ``(-1e3, -1e4, -1e5)``.
- **eigh(X @ F @ X) correctness fix** -- ``eigh(inv(S) @ F)`` was
  silently using the upper triangle of a non-Hermitian matrix and
  returning wrong eigenvalues whenever ``S`` was non-trivial. Replaced
  everywhere.
- **Production-readiness pass 1** -- trimmed inaccurate and verbose
  docstrings and dead code across ``gauNEGF/``.

Green: tested and fully working
================================

Energy-independent NEGF
    :class:`gauNEGF.scf.NEGF` base class with
    :meth:`gauNEGF.scf.NEGF.setSigma` -- constant complex self-energy
    plus Pulay DIIS SCF. Base SCF I/O, voltage bias, all spin
    treatments (restricted, unrestricted, generalized).

Energy-dependent Bethe contacts
    :meth:`gauNEGF.scfE.NEGFE.setContactBethe` -- full Bethe lattice
    with the new non-orthogonal cross-term correction.

Energy-dependent 1D contacts on minimal basis
    :meth:`gauNEGF.scfE.NEGFE.setContact1D` with
    ``contactFromFock=True`` -- C2-STO3G and Au3-CRENBS converge
    end-to-end through the Damle lower contour, asymptotic Sigma fit,
    and cross-term path.

SOC for Bethe contacts
    Slater-Koster spin-orbit terms are supported and tested.

Transport
    :func:`gauNEGF.transport.calculate_transmission`,
    :func:`gauNEGF.transport.calculate_dos`,
    :func:`gauNEGF.transport.calculate_current`. DOS with cross-term Q
    correction. Transport checkpointing.

Atomic 3D Bethe contacts
    :class:`gauNEGF.surfG3D.surfGAt3D` -- per-atom 3D builder is fully
    tested.

Numerical infrastructure
    ``inv_sqrt_general``, ``fractional_matrix_power``, JIT-compiled
    integrals, adaptive complex contour, Fermi search
    (Muller / secant / poly / bisect / predict).

Yellow: works in limited cases
================================

setContact1D with external alpha/beta matrices on minimal basis
    Works, but convergence is finicky depending on system. Smaller
    test coverage than the ``contactFromFock=True`` path.

Red: needs testing or has known limitations
=============================================

surfG3 (full 3D-periodic contact wrapper)
    The math works independently -- DOS makes sense, band diagrams
    plot correctly -- but contact generation needs work and end-to-end
    testing. Do not assume converged transport results yet.

Non-minimal basis on 1D contacts
    LANL2DZ-class (double-zeta, diffuse, ECP-with-valence) breaks the
    S-PSD precondition of the asymptotic Sigma framework. The
    corresponding integration tests are deliberately skipped with this
    reason. Fix is k-resolved overlap regularization at the surfG
    construction step (planned for the next release).

TODO for the next release
==========================

- End-to-end validation of ``surfG3`` (3D-periodic contact transport
  on a real device, comparison to reference).
- K-resolved overlap regularization to lift the minimal-basis
  precondition on 1D contacts.
- Deprecation pass on legacy ``transport.py`` functions and
  ``DOSFermiSearch``.
