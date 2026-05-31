# Lower-Contour Density: Logic, Math, and Probe-Energy Design

ASCII-only math: `Sigma`, `mu`, `eta`,
`pi` spelled out; `->` is "tends to"; `~` is "asymptotically/approximately";
`^dagger` is conjugate transpose; `Tr` is trace; `Im` imaginary part.

This note derives, from scratch, why the lower contour works the way it does:
the energy-dependent contact self-energy and its linear tail, the effective
overlap `S_eff = S - X`, the density-of-states behavior in the deep tail, the
analytic density matrix, the analytic cross-term `delta_N` (which needs no extra
eigendecomposition), and the two-probe relative scheme used in production for
extracting `X` and `Sigma_0` from the asymptotic.

--------------------------------------------------------------------------------

## 0. Setup and notation

Device Fock `F` (N x N, eV), overlap `S` (N x N). Contacts contribute a
retarded self-energy `Sigma(E)` (energy-dependent, complex, non-Hermitian: its
anti-Hermitian part is the broadening). The retarded Green's function is

    G^R(E) = inv( (E + i*eta) * S - F - Sigma(E) ).

Equilibrium density matrix (occupied states up to `mu`):

    P = (1 / (2*pi)) * integral_{-inf}^{mu} A(E) dE,    A(E) = i*(G^R - G^A) = G^R Gamma G^A,

with `Gamma = i*(Sigma - Sigma^dagger)`. Numerically we split the energy axis:

    [ENERGY_MIN, Emin]   lower contour  (deep, below the active band)
    [Emin, mu]           upper contour  (the band; always densityComplex)

The whole question of this project is how to treat the lower contour, and
whether it is even needed.

--------------------------------------------------------------------------------

## 1. Why `Sigma(E)` grows linearly: the `X` term and `S_eff = S - X`

The contact self-energy is built from the device-lead coupling and the lead
surface Green's function:

    Sigma(E) = tau(E) g_lead(E) tau(E)^dagger.

For a NON-orthogonal junction the coupling block of `(E*S - H)` carries an
overlap piece:

    tau(E) = E * S_dl - H_dl       (S_dl = device-lead overlap block).

Deep below the lead band the surface GF decays like a free resolvent:

    g_lead(E) ~ inv(E * S_lead - H_lead) ~ (1/E) * inv(S_lead),   E -> -inf.

Substituting:

    Sigma(E) ~ (E*S_dl) * (1/E) inv(S_lead) * (E*S_dl)^dagger
             = E * [ S_dl inv(S_lead) S_dl^dagger ]  +  (constant)  +  O(1/E).

So the leading deep-energy behavior is LINEAR in E:

    Sigma(E) = X * E  +  Sigma_0  +  C/E  +  O(1/E^2).            (1)

The crucial point: the linear coefficient `X = S_dl inv(S_lead) S_dl^dagger` is
built purely from OVERLAP blocks -- it is an artifact of non-orthogonality, not
real physics. It vanishes for an orthogonal junction (`S_dl = 0`).

Now look at the resolvent denominator with the linear part substituted:

    (E*S - F - Sigma(E)) ~ E*S - F - Sigma_0 - X*E
                         = E*(S - X) - (F + Sigma_0)
                         = E*S_eff - (F + Sigma_0),     S_eff = S - X.        (2)

The linear-in-E self-energy RENORMALIZES the overlap, `S -> S_eff`. Everything
downstream (the orthogonalizer `Y_eff = S_eff^{-1/2}`, the effective Fock
`Fbar = Y_eff (F + Sigma_0) Y_eff`) follows from removing this overlap artifact.
This is the entire reason `S_eff` exists.

--------------------------------------------------------------------------------

## 2. Toy model: one level, made fully explicit

Take N = 1, real on-site `eps`, scalar overlap `S = 1`, and the linear tail
`Sigma(E) = Sigma_0 + X*E` (drop `C/E` for now). Then

    G^R(E) = 1 / (E - eps - Sigma_0 - X*E)
           = 1 / ( (1 - X)*E - (eps + Sigma_0) )
           = (1/S_eff) * 1 / ( E - E_star ),

with

    S_eff = 1 - X,        E_star = (eps + Sigma_0) / S_eff.                   (3)

Two facts fall out immediately:

  - the physical pole moves from `eps` to `E_star = (eps + Sigma_0)/S_eff`;
  - the spectral WEIGHT is `1/S_eff`, not 1.

That weight `1/S_eff` is the toy-model shadow of the cross-term: in a
non-orthogonal basis the electron count attached to a state is not 1 but
`1/S_eff`, and the difference from 1 is exactly what `delta_N` accounts for at
the matrix level (Section 5). When `X = 0` (orthogonal), `S_eff = 1`, weight 1,
`delta_N = 0`.

--------------------------------------------------------------------------------

## 3. DOS in the tail: why the deep contour is negligible

The spectral function (density of states) for the toy model, with broadening
`eta`:

    A(E) = -(1/pi) Im G^R(E)
         = (1/pi) * (eta / S_eff) / ( (E - E_star)^2 + eta^2 ).               (4)

A Lorentzian centered at `E_star`, total weight `1/S_eff`. Now the tail. For
`E` far below `E_star` (`|E - E_star| >> eta`):

    A(E) ~ (1/pi) * (eta / S_eff) / (E - E_star)^2   ~   1 / E^2.             (5)

The DOS dies like `1/E^2`. The charge sitting below a cutoff `Emin` (with
`Emin < E_star`) is

    N(<Emin) = integral_{-inf}^{Emin} A(E) dE
             = (1/pi)(1/S_eff) * [ pi/2 + atan((Emin - E_star)/eta) ]
             ~ (1/pi)(1/S_eff) * (eta / |Emin - E_star|)   ->   0.            (6)

So: IF `Emin` is placed below every pole `E_star`, the deep contour
`[ENERGY_MIN, Emin]` contains negligible charge, falling off like
`1 / |Emin - E_star|`. The charge ratio `||tail|| / ||band||` drops to 1e-5
to 1e-7 when `Emin` is below all poles. CONCLUSION: when `Emin` is below all
poles, no separate lower-contour correction is needed -- a single
`densityComplex[Emin, mu]` suffices.

The lower contour only does real work in the contingency where `Emin` CANNOT be
pushed below a pole (a deep core, or a pseudo-pole). Then `[ENERGY_MIN, Emin]`
straddles real spectral weight and must be integrated -- and the right tool is
the analytic Damle method below (real-axis `densityReal` fails silently on the
sharp pole).

--------------------------------------------------------------------------------

## 4. The density matrix: one eigendecomposition

In the `Y_eff` basis, `Y_eff S_eff Y_eff = I`, and from (2)

    G^R(E) = Y_eff * inv(E*I - Fbar) * Y_eff,   Fbar = Y_eff (F + Sigma_0) Y_eff.

Diagonalize ONCE: `Fbar = V D V^{-1}`, set `Vc = inv(V^dagger)` (so
`Vc^dagger = V^{-1}`). Then

    inv(E*I - Fbar) = V * diag( 1/(E - D_i) ) * Vc^dagger.                    (7)

The equilibrium integral `integral G^R Gamma G^A dE` becomes a sum of scalar
integrals of the form `integral dE / ((E - D_i)(E - D_j^*))`, each elementary:

    integral_a^b dE / ((E - D_i)(E - D_j^*))
        = [ ln(E - D_i) - ln(E - D_j^*) ] / (D_i - D_j^*)  evaluated a..b.

That is precisely what `density()` (density.py:295) computes: the `1/(D_i - D_j^*)`
prefactor, the `log(1 - mu/D) - log(1 - Emin/D)` differences, and the broadening
`Gamma` rotated into the eigenbasis. Cost: ONE `eig(Fbar)`.

--------------------------------------------------------------------------------

## 5. The cross-term `delta_N`: analytic, NO second eig

The non-orthogonal electron count is `N = Tr(P S) + delta_N`, where (integrate.py:314)

    delta_N = -(1/pi) Im( integral Tr( G^R(E) Q(E) ) dE ),   Q(E) = crossTermQTot(E).

This is a SINGLE resolvent (not the double `G^R Gamma G^A`), so it is even
simpler. Reuse the same `V, D, Vc` from Section 4. Cycle the trace:

    Tr( G^R(E) Q(E) ) = Tr( Y_eff V diag(1/(E-D_i)) Vc^dagger Y_eff Q(E) )
                      = Tr( diag(1/(E-D_i)) * [ Vc^dagger Y_eff Q(E) Y_eff V ] )
                      = sum_i  b_i(E) / (E - D_i),                            (8)

    where   b(E) = Vc^dagger Y_eff Q(E) Y_eff V    (matmuls only, no eig).

Linearize `Q(E) ~ Q0 + Q1*E` from two anchor energies, so `b(E) = B0 + B1*E`
with `B0 = Ml Q0 Mr`, `B1 = Ml Q1 Mr`, `Ml = Vc^dagger Y_eff`, `Mr = Y_eff V`
(both precomputed once). Each scalar integral, splitting
`b0 + b1*E = b1*(E - D) + (b0 + b1*D)`:

    integral_lo^hi (b0 + b1*E)/(E - D) dE
        = b1*(hi - lo) + (b0 + b1*D) * [ log(1 - hi/D) - log(1 - lo/D) ].     (9)

    delta_N = -(1/pi) Im( sum_i { ... } ).                                   (10)

Pole bookkeeping (why this is automatically correct). Write `D_i = d_r + i*d_i`,
`d_i` small. The imaginary part of the log-difference is

    Im[ log(hi - D_i) - log(lo - D_i) ] = arg(hi - D_i) - arg(lo - D_i)
        = pi   if Re(D_i) in (lo, hi)       (the path crosses the pole)
        = 0    if Re(D_i) outside [lo, hi].                                   (11)

So `delta_N` collects contributions ONLY from states whose energy lies inside
the integration window -- exactly the physical content of "charge in this
window." Poles outside contribute a purely real log-difference and drop out.
For a tail placed below all poles (Section 3), every `Re(D_i) > hi`, so
`delta_N ~ 0`, consistent with `Tr(P S) ~ 0`.

The entire added cost beyond the density's single `eig`: two `crossTermQTot`
evaluations, four small matmuls, a diagonal, and a length-N analytic sum.
No second eigendecomposition.

--------------------------------------------------------------------------------

## 6. The asymptotic fit: why deep probes, never `Emin`

We must extract `X` and `Sigma_0` from (1) by sampling `Sigma(E)` at chosen
energies. Two rules follow directly from (1):

  (a) `Sigma(E) = X*E + Sigma_0 + C/E + ...` is LINEAR only asymptotically. The
      `C/E` (and higher) terms are the lead's band structure leaking in; they
      are large near the band edge and die deep below it. Sampling near `Emin`
      (which sits at the band edge by construction) means fitting a line through
      a point that still has curvature -- the slope `X` comes out wrong.

  (b) `X` is a GLOBAL object: `S_eff = S - X` and `Y_eff = S_eff^{-1/2}` feed
      every downstream quantity. A biased `X` corrupts the whole effective
      basis. So accuracy of `X` matters more than convenience of sampling.

Therefore the probes must sit DEEP (where `C/E` is negligible), and `Emin` is
specifically the wrong place to sample. The production scheme (Section 7) places
both probes relative to `ENERGY_MIN`, which is deep by construction and avoids
any hardcoded probe depths that could be wrong for unusual system scales.

--------------------------------------------------------------------------------

## 7. Probe design: relative to `ENERGY_MIN` and `Emin`; why two suffice

### 7.1 Error of a two-point fit

Fit the line from two energies `E1, E2` against the true `Sigma = X*E + Sigma_0
+ C/E`:

    X_fit   = ( Sigma(E2) - Sigma(E1) ) / (E2 - E1)
            = X + C*(1/E2 - 1/E1)/(E2 - E1)
            = X - C / (E1 * E2).                                             (12)

    Sigma0_fit = Sigma(E1) - X_fit * E1
               = Sigma_0 + C*(1/E1 + 1/E2).                                  (13)

Read off the key facts:

  - The slope error `-C/(E1 E2)` depends ONLY on the product `E1*E2`, NOT on the
    spacing `E2 - E1`. Two deep points give an excellent slope regardless of how
    close together they are.
  - The intercept error `C*(1/E1 + 1/E2)` also vanishes as the probes deepen.

So two probes at `ENERGY_MIN` scale drive both errors to `~ C / ENERGY_MIN^2`
and `~ C / ENERGY_MIN`, i.e. negligible.

### 7.2 Why two probes, not three?

The model has TWO parameters (`X`, `Sigma_0`). Two points determine them
exactly -- the fit is not improved by a third. A third point does exactly one
thing: it makes the system overdetermined, so the lstsq RESIDUAL becomes a
measurement of how non-linear `Sigma` still is at the probe depth. With only
two points the residual is identically zero by construction -- you lose the
self-check, not the fit.

Both probes sitting at `ENERGY_MIN` scale (deep, where linearity is essentially
guaranteed by the physics of (1)) means the linearity diagnostic adds little:
non-linearity there would signal a contact whose band extends anomalously deep,
detectable in other ways. The third probe is not computed in production; it
could be gated behind a debug flag if the self-check were ever needed.

### 7.3 Production probe scheme

    E1 = ENERGY_MIN
    E2 = (Emin + ENERGY_MIN) / 2          (the midpoint)

Because `|ENERGY_MIN| >> |Emin|` in every realistic case, `E2 ~ ENERGY_MIN/2`:
both probes are deep (good linearity), a factor ~2 apart (a clean baseline for
the slope). From (12)-(13): slope error `~ C/(0.5*ENERGY_MIN^2)`, intercept
error `~ 3C/ENERGY_MIN`. Both tiny. Fully scale-relative: no hardcoded probe
depths.

### 7.4 One caveat: cancellation in `Sigma_0`

`Sigma0_fit = Sigma(E1) - X_fit*E1`. The subtracted terms are both `~ X*E1`,
which at `E1 = ENERGY_MIN = -1e6` is `~ 1e6 * ||X||`. With `||X|| ~ 0.5`, that is
`~ 5e5`, while `Sigma_0 ~ O(10)`. The difference loses about
`log10(5e5 / 10) ~ 4.7` digits; in float64 (~16 digits) we keep ~11 -- fine.
But it degrades as `ENERGY_MIN` deepens, so do NOT push `ENERGY_MIN` to, say,
`-1e12` for this fit. At `-1e6` we are comfortable. (If `ENERGY_MIN` ever needs
to be far deeper for the integration limit, decouple the fit scale from it and
use a fixed deep multiple of `Emin` instead.)

--------------------------------------------------------------------------------

## 8. Summary

  1. Non-orthogonal contacts make `Sigma(E) ~ X*E + Sigma_0 + C/E`; the linear
     `X` is an overlap artifact, removed by `S_eff = S - X` (eqs 1-2, toy 3).
  2. DOS in the tail dies like `1/E^2`, so the deep contour holds negligible
     charge once `Emin` is below all poles -> no correction needed in the normal
     case (eqs 4-6).
  3. The density matrix is one `eig(Fbar)` (eq 7).
  4. The cross-term `delta_N` reuses that same eig, needs no second one, and
     automatically counts only in-window states (eqs 8-11).
  5. The asymptotic fit must use DEEP probes (never `Emin`), because `X` is
     global and the band-edge region is non-linear (Section 6).
  6. Two probes `{ENERGY_MIN, (Emin+ENERGY_MIN)/2}` fit the two-parameter line
     exactly; a third is only a linearity self-check, not required for the fit
     (eqs 12-13). Watch the `Sigma_0` cancellation but it is fine at `-1e6`.
