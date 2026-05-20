# TSW Convergence and Non-PSD Overlap Matrices

This note documents a subtle failure mode in the `calcTSW` lower-bound search
when the device overlap matrix `S` is not strictly positive semi-definite, the
analytical framing that explains it, and the design choices in the current
implementation that protect against it.

It is written for anyone reading `calcTSW` (or related Fermi-search code) and
wondering why the convergence test is two-sided, why there is a hard
`Emin_floor`, or why naive `Tr(P @ S)` can do something it has no business
doing.

---

## 1. What `calcTSW` is computing

`calcTSW` returns a lower integration bound `Eminf` such that the complex-
contour integral

```
Tr(P @ S),  with  P = density matrix from contour [Eminf, Emax] in UHP
```

is stable against further lowering of `Eminf`. The reference is the same
quantity evaluated at the configured `Emin_floor` (effectively the "infinite"
contour). The goal is to find the *largest* `Eminf` whose contour still
captures all the integrated spectral weight, because a smaller contour means
fewer quadrature points to converge the rest of the SCF.

The function takes successively wider contours by doubling `|Eminf|` each
iteration. When the new and reference TSW agree within tolerance, that
`Eminf` is good enough.

---

## 2. The monotonicity guarantee (when S is PSD)

For PSD `S` and a physically constructed retarded Green's function with PSD
broadening `Γ = i(Σ - Σ†)`, the spectral function
`A(E) = -Im G^R(E) / π = G^R Γ G^A / (2π)` is PSD at every real `E`, and the
contour integral reduces (by Cauchy's theorem and Plemelj) to

```
Tr(P @ S) = ∫_Eminf^Emax  Tr[A(E) S]  dE
```

`Tr[A(E) S] = Tr[S^{1/2} A(E) S^{1/2}] ≥ 0`, so the integrand is non-negative
pointwise and `Tr(P @ S)` is **monotonically non-increasing** in `Eminf`.
Lowering `Eminf` can only add weight, never remove it.

With this guarantee, the original one-sided convergence test
`(TSW_ref - TSW_new)/TSW_ref < tol` is sound: it only ever needs to detect
"TSW_new is too small," because TSW_new larger than TSW_ref is impossible.

---

## 3. How non-PSD S breaks the guarantee

In ill-conditioned basis sets (e.g. all-electron Gaussian bases with mild
linear dependence, basis sets that include diffuse functions on heavy atoms,
or any case where `S` has eigenvalues at or below floating-point noise), the
PSD assumption on `S` fails. The spectral function `A(E)` remains PSD on its
own, but `Tr[A(E) S]` can be **negative** at energies where `A` has overlap
with the negative-eigenvalue eigenvector of `S`.

The most common manifestation is a **deep pseudo-pole**: a generalized
eigenstate of `(F + Σ(E), S)` lands far below the physical band (often
thousands of eV deep, well outside any chemically meaningful spectrum), and
its projection onto the device basis produces a sharp DOS feature with
**negative integrated weight**. These pseudo-poles are not detectable from
the bare Fock eigenspectrum because they only appear once the energy-dependent
self-energy `Σ(E)` is included.

When such a pseudo-pole exists:

- A contour whose `[Eminf, Emax]` range *includes* the pseudo-pole picks up
  its negative contribution -> total TSW is reduced.
- A contour whose range *excludes* it (i.e. `Eminf > E_pseudo`) does not see
  it -> total TSW is larger by the magnitude of the missing negative
  contribution.

**Lower `Eminf` gives smaller `Tr(P @ S)`. Monotonicity is reversed.** A
one-sided test of the form `(TSW_ref - TSW_new)/TSW_ref < tol` treats this
as trivially converged, because `TSW_new > TSW_ref` makes the left side
negative -- and the function returns an Eminf that misses spectral weight by
hundreds of electrons without flagging anything.

---

## 4. The fix: two-sided test plus hard floor

`calcTSW` uses two protections in combination.

**Two-sided convergence test:**
```
if abs(TSW_ref - TSW_new) / abs(TSW_ref) < tol:
```
Symmetric in sign, so both undercounting (legitimate failure mode for PSD S)
and overcounting (pseudo-pole exclusion under non-PSD S) keep the loop
expanding `|Eminf|`.

**`Emin_floor` parameter (defaults to `ENERGY_MIN` from config):**
- Used as the contour bound when computing the reference TSW, so the
  reference is always the deepest contour we are willing to evaluate.
- Used as a hard limit on the doubling loop. If the next `2 * Eminf` would
  pass `Emin_floor`, the function warns and returns rather than silently
  expanding further. The warning explicitly names non-PSD S as the likely
  cause and suggests `eigvalsh(S)` as the next diagnostic.

For typical systems whose minimum Fock eigenvalue is `~-1000 eV` or higher,
doubling reaches the default `Emin_floor` of `-1e6 eV` in ~10 iterations, so
`FERMI_SEARCH_CYCLES` does not need to be large.

---

## 5. Diagnosing a suspected pseudo-pole

If `calcTSW` triggers the floor warning, or if the SCF is producing odd
electron counts, the standard checks are:

**1. DOS sweep at deep energies.** Compute
`-Im Tr[G^R(E + iη) @ S] / π` on a linear grid from
`Emin_floor` up to the band edge. The integration's broadening `η = g.eta`
must be used directly -- the `_compute_dos_at_energy` helper in `density.py`
uses the config-level `ETA`, which may differ. A pseudo-pole shows as a
narrow spike orders of magnitude above the surrounding DOS values.

**2. Eigenvalue sweep of `S`.** `eigvalsh(S)` gives the spectrum of the
overlap matrix. If the smallest eigenvalue is small (`< 1e-6`) or negative,
non-PSD-related artifacts are likely. The pseudo-pole's negative weight
scales with how negative that eigenvalue is and how much overlap the
pseudo-pole eigenvector has with it.

**3. Fixed-N vs adaptive contour comparison.** Running `densityComplexN`
with a large fixed `N` (e.g. 1000) at several `Eminf` values isolates
quadrature artifacts from analytical artifacts. If the contour-shape
dependence persists at fixed N, the issue is in the integrand, not in the
adaptive convergence detection.

---

## 6. Trade-off with the cross-term Q

An alternative integrated-DOS formulation for embedded systems in
non-orthogonal bases is

```
n(E) = -Im Tr[G^R(E) (S - Q(E))] / π
```

where `Q(E)` is the cross-term related to the energy-dependent embedding
(roughly `dΣ/dE`). It can be motivated by writing the integrated DOS as
`-1/π Im d/dE ln det G^R(E)` (a Lloyd-style derivation; see P. Lloyd,
*Proc. Phys. Soc.* **90**, 207 (1967) for the original scattering-theory
form, and the TranSIESTA / non-orthogonal NEGF references already cited
in `density.py` for the density-matrix machinery). Whether this exact
correction term has a standard published name in the NEGF-DFT context
is something to verify against the literature; the form used in gauNEGF
matches what falls out of the derivation regardless of naming.

The relevant property for this discussion is that the combination
`Tr[G^R (S - Q)]` is **contour-independent even when S is not PSD** --
the `Q` correction absorbs the basis-overlap pathologies that produce
negative-weight pseudo-poles in the naive `Tr[A S]`.

The trade-off in `calcTSW`:

- **Naive `Tr(P @ S)` (no Q):** cheaper (no Q evaluation), but requires the
  contour to bracket every pseudo-pole. The current `calcTSW`
  implementation uses this with the two-sided test + floor as protection.

- **`Tr(P @ S) + delta_N` (with Q):** contour-independent, works even with
  tight `|Eminf|` bounds, but requires `g.crossTermQTot(E)` to be available
  and evaluated alongside `sigmaTot(E)` at every quadrature point. Used
  elsewhere in `density.py` (see `calcEmin`).

For TSW convergence checks specifically, naive `Tr(P @ S)` is the right
choice when (a) the basis is well-conditioned, or (b) `Emin_floor` is set
deep enough that pseudo-poles are always inside the reference contour. The
floor warning is the indicator that condition (b) has failed and either the
floor needs to be lowered or the basis needs attention.

---

## 7. Configuration

The relevant config knobs in `gauNEGF/config.py`:

```python
ENERGY_MIN       # eV - default Emin_floor for calcTSW and reference bound
                 # MUST BE NEGATIVE
FERMI_SEARCH_CYCLES  # max doubling iterations in calcTSW
FERMI_CALCULATION_TOL # convergence tolerance for relative TSW change
FERMI_DEBUG      # if True, print per-iteration Eminf and dTSW
```

`Emin_floor` is also a kwarg on `calcTSW` if a per-call override is needed
without touching the config.
