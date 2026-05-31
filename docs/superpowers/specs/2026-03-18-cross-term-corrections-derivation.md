# Cross-Term Corrections and surfG Interface Redesign for Non-Orthogonal NEGF

## Errata (2026-03-19): bar_tau on the complex contour (GHF-general)

The original derivation below uses $\tau^\dagger$ (conjugate transpose) as the right-side
coupling factor in $\Sigma$, $G_{LD}^R$, and $Q_{\mathrm{rev}}$. This is correct **on the
real energy axis** where $z = z^*$, but **incorrect on the complex contour** used by
`densityComplexN` / `densityComplex`.

**Root cause:** The equation of motion $(z\mathbf{S} - \mathbf{H})\mathbf{G}^R = \mathbf{I}$
gives the (L,D) coupling block as $z S_{LD} - H_{LD}$, using $z$ (not $z^*$). Since
$\tau(z) = z S_{DL} - H_{DL}$, the right-side factor is:

$$\bar{\tau}(z) \;=\; z\,S_{DL}^\dagger - H_{DL}^\dagger$$

For **real** $S$, $H$: this simplifies to $\tau(z)^T$ (plain transpose).
For **complex Hermitian** $S$, $H$ (GHF/SOC): $\bar{\tau}$ must be constructed explicitly
because $\tau^T$ does not conjugate the static matrix elements, while $\tau^\dagger$
conjugates $z$. Neither shortcut is correct in the general case.

The error from using $\tau^\dagger$ is $(z - z^*)\,S_{DL}^\dagger = 2i\,\mathrm{Im}(z)\,S_{DL}^\dagger$.

- **Real axis** ($\mathrm{Im}(z) = \eta \approx 0$): $\bar{\tau} \approx \tau^\dagger$, no visible error.
- **Complex contour** ($\mathrm{Im}(z) \sim$ several eV): error is large, proportional to overlap $S_{LD}$.

**Corrected formulas** (replace $\tau^\dagger$ with explicit $\bar{\tau}(z) = z S_{DL}^\dagger - H_{DL}^\dagger$):

| Original | Corrected |
|----------|-----------|
| $\Sigma = \tau\,g\,\tau^\dagger$ | $\Sigma(z) = \tau(z)\,g(z)\,\bar{\tau}(z)$ |
| $G_{LD}^R = -g\,\tau^\dagger\,G_{DD}^R$ | $G_{LD}^R = -g\,\bar{\tau}\,G_{DD}^R$ |
| $Q_{\mathrm{rev}} = S_{DL}\,g\,\tau^\dagger$ | $Q_{\mathrm{rev}} = S_{DL}\,g\,\bar{\tau}$ |
| Code: `B.conj().T` | Code: `z * S.conj().T - V.conj().T` (explicit) |

$Q_{\mathrm{fwd}} = \tau\,g\,S_{LD}$ is **unchanged** ($S_{LD} = S_{DL}^\dagger$ is the
overlap matrix, which is $z$-independent).

This affects every `.conj().T` on a **coupling matrix** (tau, B, t\_reg) across surfG1D,
surfGBAt, and surfGAt3D -- in the surface GF iteration, sigma, and crossTermQ. See
implementation plan `docs/superpowers/plans/2026-03-19-contour-transpose-fix.md`.

---

## Problem Statement

In the current implementation, the Fermi energy search computes the device electron count as:

$$N_D = \mathrm{Tr}(P_{DD} S_{DD})$$

where $P_{DD}$ is the device density matrix and $S_{DD}$ is the device overlap matrix. This is correct **only in an orthogonal basis** ($S_{DL} = S_{DR} = 0$).

In a non-orthogonal basis, the Mulliken electron count for device orbitals requires contributions from the off-diagonal blocks of the full system density matrix:

$$N_D = \mathrm{Tr}(P_{DD} S_{DD}) + \mathrm{Tr}(P_{DL} S_{LD}) + \mathrm{Tr}(P_{DR} S_{RD})$$

The current code omits the second and third terms. This document derives the missing cross-terms and their impact on the Fermi search and density matrix.

---

## 1. Full System Setup

### Block structure

The full system (Left lead + Device + Right lead) has:

$$
\mathbf{H} = \begin{pmatrix} H_{LL} & H_{LD} & 0 \\ H_{DL} & H_{DD} & H_{DR} \\ 0 & H_{RD} & H_{RR} \end{pmatrix}, \qquad
\mathbf{S} = \begin{pmatrix} S_{LL} & S_{LD} & 0 \\ S_{DL} & S_{DD} & S_{DR} \\ 0 & S_{RD} & S_{RR} \end{pmatrix}
$$

where we assume no direct left-right coupling ($H_{LR} = S_{LR} = 0$). All blocks satisfy Hermiticity: $H_{LD} = H_{DL}^\dagger$, $S_{LD} = S_{DL}^\dagger$.

### Definitions

| Symbol | Definition | Code equivalent |
|--------|-----------|-----------------|
| $S_{DL}$, $S_{LD} = S_{DL}^\dagger$ | Device-lead overlap blocks | `stauList[i]` (surfG1D), `Slist[a]` (surfG3D) |
| $\tau_\alpha = E S_{D\alpha} - H_{D\alpha}$ | Energy-dependent coupling (D to lead $\alpha$) | `E*stau - tau` in surfG1D |
| $g_\alpha^R = (E S_{\alpha\alpha} - H_{\alpha\alpha})^{-1}$ | Surface Green's function of isolated lead $\alpha$ | `g.g(E, i)` |
| $\Sigma_\alpha = \tau_\alpha\, g_\alpha^R\, \bar{\tau}_\alpha$ | Self-energy from lead $\alpha$ | `g.sigma(E, i)` |
| $G_{DD}^R = (E S_{DD} - H_{DD} - \Sigma_L - \Sigma_R)^{-1}$ | Device retarded Green's function | What the code currently computes |

---

## 2. Off-Diagonal Green's Function Blocks

Starting from the equation of motion $(E\mathbf{S} - \mathbf{H})\mathbf{G}^R = \mathbf{I}$, we derive all off-diagonal blocks of the retarded Green's function. Every algebraic step is shown.

### 2.1 Block notation and the nine equations

Define the diagonal and coupling blocks of the matrix $E\mathbf{S} - \mathbf{H}$:

| Block position | Expression | Shorthand |
|---|---|---|
| $(L,L)$ | $E S_{LL} - H_{LL}$ | $A_L$ (inverse: $g_L^R = A_L^{-1}$) |
| $(D,D)$ | $E S_{DD} - H_{DD}$ | $A_D$ |
| $(R,R)$ | $E S_{RR} - H_{RR}$ | $A_R$ (inverse: $g_R^R = A_R^{-1}$) |
| $(D,L)$ | $E S_{DL} - H_{DL}$ | $\tau_L$ |
| $(D,R)$ | $E S_{DR} - H_{DR}$ | $\tau_R$ |
| $(L,D)$ | $E S_{LD} - H_{LD} = E S_{DL}^\dagger - H_{DL}^\dagger$ | $\bar{\tau}_L$ |
| $(R,D)$ | $E S_{RD} - H_{RD} = E S_{DR}^\dagger - H_{DR}^\dagger$ | $\bar{\tau}_R$ |
| $(L,R)$, $(R,L)$ | $0$ (no direct left-right coupling) | $0$ |

**$\bar{\tau}$ vs $\tau^\dagger$:** On the real axis ($E$ real, $S$ and $H$ Hermitian), $\bar{\tau}_\alpha = \tau_\alpha^\dagger$ because $(E S_{D\alpha} - H_{D\alpha})^\dagger = E S_{D\alpha}^\dagger - H_{D\alpha}^\dagger = E S_{\alpha D} - H_{\alpha D}$. On the complex contour ($E = z$ with $\mathrm{Im}(z) \neq 0$), $\bar{\tau}_\alpha(z) \neq \tau_\alpha(z)^\dagger$. See errata at the top of this document.

The full block matrix equation $(E\mathbf{S} - \mathbf{H})\mathbf{G}^R = \mathbf{I}$ is:

$$
\begin{pmatrix} A_L & \bar{\tau}_L & 0 \\ \tau_L & A_D & \tau_R \\ 0 & \bar{\tau}_R & A_R \end{pmatrix}
\begin{pmatrix} G_{LL} & G_{LD} & G_{LR} \\ G_{DL} & G_{DD} & G_{DR} \\ G_{RL} & G_{RD} & G_{RR} \end{pmatrix}
= \begin{pmatrix} I & 0 & 0 \\ 0 & I & 0 \\ 0 & 0 & I \end{pmatrix}
$$

Multiplying row-by-row and equating column-by-column gives nine block equations. We label each by (row, column). All $G$ blocks carry superscript $R$ (retarded), dropped for readability.

**Left-lead row ($L$):**

$$
\begin{aligned}
(L,L): \quad & A_L\, G_{LL} + \bar{\tau}_L\, G_{DL} = I \\
(L,D): \quad & A_L\, G_{LD} + \bar{\tau}_L\, G_{DD} = 0 \\
(L,R): \quad & A_L\, G_{LR} + \bar{\tau}_L\, G_{DR} = 0
\end{aligned}
$$

**Device row ($D$):**

$$
\begin{aligned}
(D,L): \quad & \tau_L\, G_{LL} + A_D\, G_{DL} + \tau_R\, G_{RL} = 0 \\
(D,D): \quad & \tau_L\, G_{LD} + A_D\, G_{DD} + \tau_R\, G_{RD} = I \\
(D,R): \quad & \tau_L\, G_{LR} + A_D\, G_{DR} + \tau_R\, G_{RR} = 0
\end{aligned}
$$

**Right-lead row ($R$):**

$$
\begin{aligned}
(R,L): \quad & \bar{\tau}_R\, G_{DL} + A_R\, G_{RL} = 0 \\
(R,D): \quad & \bar{\tau}_R\, G_{DD} + A_R\, G_{RD} = 0 \\
(R,R): \quad & \bar{\tau}_R\, G_{DR} + A_R\, G_{RR} = I
\end{aligned}
$$

### 2.2 Derivation of $G_{LD}^R$ (one equation, one step)

Start from equation **(L,D)**:

$$A_L\, G_{LD} + \bar{\tau}_L\, G_{DD} = 0$$

This contains only $G_{LD}$ and $G_{DD}$. Since $G_{DD}$ is determined independently (see Section 2.4, Step 7), we solve for $G_{LD}$. Move the second term to the right:

$$A_L\, G_{LD} = -\bar{\tau}_L\, G_{DD}$$

Left-multiply both sides by $A_L^{-1} = g_L^R$:

$$\boxed{G_{LD}^R = -g_L^R\, \bar{\tau}_L\, G_{DD}^R}$$

### 2.3 Derivation of $G_{RD}^R$ (one equation, same pattern)

Start from equation **(R,D)**:

$$\bar{\tau}_R\, G_{DD} + A_R\, G_{RD} = 0$$

Move the first term to the right:

$$A_R\, G_{RD} = -\bar{\tau}_R\, G_{DD}$$

Left-multiply both sides by $A_R^{-1} = g_R^R$:

$$\boxed{G_{RD}^R = -g_R^R\, \bar{\tau}_R\, G_{DD}^R}$$

### 2.4 Derivation of $G_{DL}^R$ (three equations, eight steps)

This is the harder derivation. Equation **(D,L)** reads:

$$\tau_L\, G_{LL} + A_D\, G_{DL} + \tau_R\, G_{RL} = 0 \tag{D,L}$$

This contains **three** unknowns: $G_{LL}$, $G_{DL}$, and $G_{RL}$. We must eliminate $G_{LL}$ and $G_{RL}$ using other block equations.

**Step 1: Express $G_{RL}$ in terms of $G_{DL}$.**

From equation **(R,L)**:

$$\bar{\tau}_R\, G_{DL} + A_R\, G_{RL} = 0$$

Isolate $G_{RL}$:

$$A_R\, G_{RL} = -\bar{\tau}_R\, G_{DL}$$

Left-multiply by $A_R^{-1} = g_R^R$:

$$G_{RL} = -g_R^R\, \bar{\tau}_R\, G_{DL} \tag{i}$$

**Step 2: Express $G_{LL}$ in terms of $G_{DL}$.**

From equation **(L,L)**:

$$A_L\, G_{LL} + \bar{\tau}_L\, G_{DL} = I$$

Isolate $G_{LL}$:

$$A_L\, G_{LL} = I - \bar{\tau}_L\, G_{DL}$$

Left-multiply by $A_L^{-1} = g_L^R$:

$$G_{LL} = g_L^R\,(I - \bar{\tau}_L\, G_{DL})$$

Distribute $g_L^R$:

$$G_{LL} = g_L^R - g_L^R\, \bar{\tau}_L\, G_{DL} \tag{ii}$$

**Step 3: Substitute (i) and (ii) into equation (D,L).**

Replace $G_{LL}$ using (ii) and $G_{RL}$ using (i):

$$\tau_L\,\bigl[\,g_L^R - g_L^R\, \bar{\tau}_L\, G_{DL}\,\bigr] + A_D\, G_{DL} + \tau_R\,\bigl[\,-g_R^R\, \bar{\tau}_R\, G_{DL}\,\bigr] = 0$$

**Step 4: Distribute $\tau_L$ and $\tau_R$ into the brackets.**

$$\tau_L\, g_L^R \;-\; \tau_L\, g_L^R\, \bar{\tau}_L\, G_{DL} \;+\; A_D\, G_{DL} \;-\; \tau_R\, g_R^R\, \bar{\tau}_R\, G_{DL} = 0$$

**Step 5: Separate constant terms from $G_{DL}$ terms.**

Move the term without $G_{DL}$ to the right-hand side:

$$-\tau_L\, g_L^R\, \bar{\tau}_L\, G_{DL} \;+\; A_D\, G_{DL} \;-\; \tau_R\, g_R^R\, \bar{\tau}_R\, G_{DL} = -\tau_L\, g_L^R$$

**Step 6: Factor out $G_{DL}$ on the left.**

All three terms on the left multiply $G_{DL}$ from the left, so:

$$\bigl[\;A_D \;-\; \tau_L\, g_L^R\, \bar{\tau}_L \;-\; \tau_R\, g_R^R\, \bar{\tau}_R\;\bigr]\, G_{DL} = -\tau_L\, g_L^R$$

**Step 7: Identify the self-energies and $G_{DD}^{-1}$.**

The self-energy from lead $\alpha$ is $\Sigma_\alpha = \tau_\alpha\, g_\alpha^R\, \bar{\tau}_\alpha$, so:

$$\bigl[\;A_D - \Sigma_L - \Sigma_R\;\bigr]\, G_{DL} = -\tau_L\, g_L^R$$

The device retarded Green's function is defined as $G_{DD}^R = [A_D - \Sigma_L - \Sigma_R]^{-1}$, so the bracket on the left is $(G_{DD}^R)^{-1}$:

$$(G_{DD}^R)^{-1}\, G_{DL} = -\tau_L\, g_L^R$$

**Step 8: Left-multiply both sides by $G_{DD}^R$.**

$$\boxed{G_{DL}^R = -G_{DD}^R\, \tau_L\, g_L^R}$$

### 2.5 Derivation of $G_{DR}^R$ (three equations, same pattern)

Equation **(D,R)** reads:

$$\tau_L\, G_{LR} + A_D\, G_{DR} + \tau_R\, G_{RR} = 0 \tag{D,R}$$

This also has three unknowns ($G_{LR}$, $G_{DR}$, $G_{RR}$) and follows the same elimination pattern.

**Step 1: Express $G_{LR}$ in terms of $G_{DR}$** using equation **(L,R)**:

$$A_L\, G_{LR} + \bar{\tau}_L\, G_{DR} = 0 \;\;\Longrightarrow\;\; G_{LR} = -g_L^R\, \bar{\tau}_L\, G_{DR} \tag{iii}$$

**Step 2: Express $G_{RR}$ in terms of $G_{DR}$** using equation **(R,R)**:

$$\bar{\tau}_R\, G_{DR} + A_R\, G_{RR} = I$$

$$A_R\, G_{RR} = I - \bar{\tau}_R\, G_{DR}$$

$$G_{RR} = g_R^R - g_R^R\, \bar{\tau}_R\, G_{DR} \tag{iv}$$

**Step 3: Substitute (iii) and (iv) into equation (D,R).**

$$\tau_L\,[-g_L^R\, \bar{\tau}_L\, G_{DR}] + A_D\, G_{DR} + \tau_R\,[g_R^R - g_R^R\, \bar{\tau}_R\, G_{DR}] = 0$$

**Step 4: Expand.**

$$-\tau_L\, g_L^R\, \bar{\tau}_L\, G_{DR} + A_D\, G_{DR} + \tau_R\, g_R^R - \tau_R\, g_R^R\, \bar{\tau}_R\, G_{DR} = 0$$

**Step 5: Isolate $G_{DR}$ terms.**

$$[A_D - \tau_L\, g_L^R\, \bar{\tau}_L - \tau_R\, g_R^R\, \bar{\tau}_R]\, G_{DR} = -\tau_R\, g_R^R$$

**Step 6: Recognize $[A_D - \Sigma_L - \Sigma_R] = (G_{DD}^R)^{-1}$, then left-multiply by $G_{DD}^R$.**

$$\boxed{G_{DR}^R = -G_{DD}^R\, \tau_R\, g_R^R}$$

### 2.6 Remaining blocks (back-substitution)

These are not needed for the cross-term derivation but complete the picture.

**Diagonal blocks** -- substitute $G_{DL}$ and $G_{DR}$ back into (ii) and (iv):

$$G_{LL}^R = g_L^R - g_L^R\, \bar{\tau}_L\, G_{DL}^R = g_L^R + g_L^R\, \bar{\tau}_L\, G_{DD}^R\, \tau_L\, g_L^R$$

$$G_{RR}^R = g_R^R - g_R^R\, \bar{\tau}_R\, G_{DR}^R = g_R^R + g_R^R\, \bar{\tau}_R\, G_{DD}^R\, \tau_R\, g_R^R$$

**Cross-lead blocks** -- substitute into (i) and (iii):

$$G_{RL}^R = -g_R^R\, \bar{\tau}_R\, G_{DL}^R = g_R^R\, \bar{\tau}_R\, G_{DD}^R\, \tau_L\, g_L^R$$

$$G_{LR}^R = -g_L^R\, \bar{\tau}_L\, G_{DR}^R = g_L^R\, \bar{\tau}_L\, G_{DD}^R\, \tau_R\, g_R^R$$

### 2.7 Summary table

| Block | Formula | Derived from |
|---|---|---|
| $G_{DD}$ | $[A_D - \Sigma_L - \Sigma_R]^{-1}$ | Standard Dyson equation |
| $G_{LD}$ | $-g_L\, \bar{\tau}_L\, G_{DD}$ | Eq. (L,D) alone |
| $G_{RD}$ | $-g_R\, \bar{\tau}_R\, G_{DD}$ | Eq. (R,D) alone |
| $G_{DL}$ | $-G_{DD}\, \tau_L\, g_L$ | Eqs. (D,L) + (L,L) + (R,L) |
| $G_{DR}$ | $-G_{DD}\, \tau_R\, g_R$ | Eqs. (D,R) + (L,R) + (R,R) |
| $G_{LL}$ | $g_L + g_L\, \bar{\tau}_L\, G_{DD}\, \tau_L\, g_L$ | Back-sub into (L,L) |
| $G_{RR}$ | $g_R + g_R\, \bar{\tau}_R\, G_{DD}\, \tau_R\, g_R$ | Back-sub into (R,R) |
| $G_{LR}$ | $g_L\, \bar{\tau}_L\, G_{DD}\, \tau_R\, g_R$ | Back-sub into (L,R) |
| $G_{RL}$ | $g_R\, \bar{\tau}_R\, G_{DD}\, \tau_L\, g_L$ | Back-sub into (R,L) |

**Structural pattern:** The asymmetry between "lead-to-device" and "device-to-lead" blocks:

- $G_{LD}$, $G_{RD}$ have the structure: $g_{\text{lead}}$ ... $G_{DD}$ (lead GF on the left, device GF on the right)
- $G_{DL}$, $G_{DR}$ have the structure: $G_{DD}$ ... $g_{\text{lead}}$ (device GF on the left, lead GF on the right)

This arises because $G_{LD}$ is solved from the lead row (where $A_L$ is independently invertible), while $G_{DL}$ is solved from the device row (where $A_D$ is not invertible alone -- it needs the self-energies $\Sigma_L$, $\Sigma_R$ to become $G_{DD}^{-1}$).

### 2.8 Consistency check: $G^A = (G^R)^\dagger$ at the block level

Take the adjoint of $G_{DL}^R = -G_{DD}^R\, \tau_L\, g_L^R$. The adjoint reverses the order and daggers each factor:

$$(G_{DL}^R)^\dagger = -(g_L^R)^\dagger\, \tau_L^\dagger\, (G_{DD}^R)^\dagger = -g_L^A\, \tau_L^\dagger\, G_{DD}^A$$

Since $G^A = (G^R)^\dagger$, this must equal the $(L,D)$ block of $G^A$: $(G_{DL}^R)^\dagger \stackrel{?}{=} G_{LD}^A$.

Verify independently from the advanced equation of motion $(E^* \mathbf{S} - \mathbf{H})\mathbf{G}^A = \mathbf{I}$. The $(L,D)$ block is:

$$(E^* S_{LL} - H_{LL})\, G_{LD}^A + (E^* S_{LD} - H_{LD})\, G_{DD}^A = 0$$

The first factor: $(E^* S_{LL} - H_{LL})^{-1} = [(E S_{LL} - H_{LL})^\dagger]^{-1} = [(E S_{LL} - H_{LL})^{-1}]^\dagger = (g_L^R)^\dagger = g_L^A$.

The second factor: $E^* S_{LD} - H_{LD} = E^* S_{DL}^\dagger - H_{DL}^\dagger = (E S_{DL} - H_{DL})^\dagger = \tau_L^\dagger$.

So the advanced equation gives:

$$G_{LD}^A = -g_L^A\, \tau_L^\dagger\, G_{DD}^A = (G_{DL}^R)^\dagger \quad \checkmark$$

---

## 3. Spectral Function Cross-Terms

The spectral function is $\mathbf{A} = i(\mathbf{G}^R - \mathbf{G}^A)$. We derive the off-diagonal block $A_{DL}$ step by step and factor it into a form suitable for electron-count integration.

### 3.1 Ingredients

The device-to-left spectral block is:

$$A_{DL} = i(G_{DL}^R - G_{DL}^A)$$

From Section 2 we have:

$$G_{DL}^R = -G_{DD}^R\, \tau_L\, g_L^R \tag{from 2.4}$$

For $G_{DL}^A$, use $G^A = (G^R)^\dagger$, so $G_{DL}^A = (G_{LD}^R)^\dagger$:

$$G_{LD}^R = -g_L^R\, \bar{\tau}_L\, G_{DD}^R \tag{from 2.2}$$

Take the adjoint (reverse order, dagger each factor):

$$(G_{LD}^R)^\dagger = -(G_{DD}^R)^\dagger\, \bar{\tau}_L^\dagger\, (g_L^R)^\dagger = -G_{DD}^A\, \bar{\tau}_L^\dagger\, g_L^A$$

**On the real axis** ($E$ real): $\bar{\tau}_L = \tau_L^\dagger$, so $\bar{\tau}_L^\dagger = (\tau_L^\dagger)^\dagger = \tau_L$. Therefore:

$$G_{DL}^A\big|_{\text{real axis}} = -G_{DD}^A\, \tau_L\, g_L^A$$

The rest of this section works on the real axis (the complex contour version is handled in Section 7 via the $Q^{\mathrm{sym}}$ approach).

### 3.2 Substitution into $A_{DL}$

$$A_{DL} = i\bigl(G_{DL}^R - G_{DL}^A\bigr)$$

Substitute $G_{DL}^R = -G_{DD}^R\, \tau_L\, g_L^R$ and $G_{DL}^A = -G_{DD}^A\, \tau_L\, g_L^A$:

$$A_{DL} = i\bigl(-G_{DD}^R\, \tau_L\, g_L^R - (-G_{DD}^A\, \tau_L\, g_L^A)\bigr)$$

Simplify the double negative:

$$A_{DL} = i\bigl(-G_{DD}^R\, \tau_L\, g_L^R + G_{DD}^A\, \tau_L\, g_L^A\bigr)$$

Factor out $-1$:

$$A_{DL} = -i\bigl(G_{DD}^R\, \tau_L\, g_L^R - G_{DD}^A\, \tau_L\, g_L^A\bigr) \tag{*}$$

### 3.3 Factorization using device and lead spectral functions

Equation $(*)$ has four distinct Green's functions ($G_{DD}^R$, $G_{DD}^A$, $g_L^R$, $g_L^A$). We want to express it in terms of the spectral functions $A_{DD} = i(G_{DD}^R - G_{DD}^A)$ and $a_L = i(g_L^R - g_L^A)$. This requires an add-and-subtract trick.

**Step 1:** Insert zero $= -G_{DD}^A\, \tau_L\, g_L^R + G_{DD}^A\, \tau_L\, g_L^R$ inside the parentheses:

$$(*) = -i\bigl(\underbrace{G_{DD}^R\, \tau_L\, g_L^R - G_{DD}^A\, \tau_L\, g_L^R}_{\text{group 1}} + \underbrace{G_{DD}^A\, \tau_L\, g_L^R - G_{DD}^A\, \tau_L\, g_L^A}_{\text{group 2}}\bigr)$$

**Step 2:** Factor each group. Both groups share $\tau_L$ in the middle.

Group 1: factor out $(G_{DD}^R - G_{DD}^A)$ on the left:

$$\text{Group 1} = (G_{DD}^R - G_{DD}^A)\, \tau_L\, g_L^R$$

Group 2: factor out $G_{DD}^A\, \tau_L$ on the left:

$$\text{Group 2} = G_{DD}^A\, \tau_L\, (g_L^R - g_L^A)$$

**Step 3:** Substitute both groups back:

$$(*) = -i\bigl[(G_{DD}^R - G_{DD}^A)\, \tau_L\, g_L^R + G_{DD}^A\, \tau_L\, (g_L^R - g_L^A)\bigr]$$

**Step 4:** Distribute the $-i$ into each group, recognizing $A = i(G^R - G^A)$:

$$= -\bigl[\underbrace{i(G_{DD}^R - G_{DD}^A)}_{= A_{DD}}\, \tau_L\, g_L^R + G_{DD}^A\, \tau_L\, \underbrace{i(g_L^R - g_L^A)}_{= a_L}\bigr]$$

$$\boxed{A_{DL} = -\bigl(A_{DD}\, \tau_L\, g_L^R + G_{DD}^A\, \tau_L\, a_L\bigr)}$$

**Alternative factorization:** Instead of adding/subtracting $G_{DD}^A\, \tau_L\, g_L^R$ in Step 1, we could add/subtract $G_{DD}^R\, \tau_L\, g_L^A$. Following the same procedure:

$$(*) = -i\bigl(G_{DD}^R\, \tau_L\, g_L^R - G_{DD}^R\, \tau_L\, g_L^A + G_{DD}^R\, \tau_L\, g_L^A - G_{DD}^A\, \tau_L\, g_L^A\bigr)$$

$$= -\bigl(G_{DD}^R\, \tau_L\, a_L + A_{DD}\, \tau_L\, g_L^A\bigr)$$

Both forms are equivalent; the first is used in the $\delta N$ derivation (Section 7).

### 3.4 Cross-term contribution to Mulliken electron count

The Mulliken electron count for the device region involves $\delta N_L = \mathrm{Tr}(P_{DL}\, S_{LD})$ (see Section 4). Using $P_{DL} = \frac{1}{2\pi} \int f(E)\, A_{DL}\, dE$, this becomes:

$$\delta N_L = \frac{1}{2\pi} \int f(E)\, \mathrm{Tr}(A_{DL}\, S_{LD})\, dE$$

Substituting the unfactored spectral function from equation $(*)$:

$$\mathrm{Tr}(A_{DL}\, S_{LD}) = -i\,\mathrm{Tr}\!\bigl[(G_{DD}^R\, \tau_L\, g_L^R - G_{DD}^A\, \tau_L\, g_L^A)\, S_{LD}\bigr]$$

This is the scalar integrand that must be co-accumulated during contour integration. On the complex contour, $G_{DD}^A$ is not available (only $G_{DD}^R$ is computed), so the symmetrized $Q$ approach in Section 7 handles both terms using only $G^R$.

---

## 4. Density Matrix Cross-Terms

### Equilibrium density matrix

In equilibrium at chemical potential $\mu$ and temperature $T$:

$$P_{DL} = \frac{1}{2\pi} \int_{-\infty}^{\infty} f(E, \mu)\, A_{DL}(E)\, dE$$

where $f(E, \mu)$ is the Fermi-Dirac distribution.

Substituting the expression for $A_{DL}$:

$$P_{DL} = \frac{-1}{2\pi} \int f(E)\left[A_{DD}\, \tau_L\, g_L^R + G_{DD}^A\, \tau_L\, a_L\right] dE$$

### Orthogonal limit

When $S_{DL} = S_{DR} = 0$, the coupling reduces to $\tau_\alpha = -H_{D\alpha}$ (pure hopping, no overlap). The cross-term contribution to the electron count is:

$$\mathrm{Tr}(P_{DL}\, S_{LD}) = 0$$

since $S_{LD} = 0$. **The device density matrix $P_{DD}$ and electron count $\mathrm{Tr}(P_{DD} S_{DD}) = \mathrm{Tr}(P_{DD})$ are exact in this limit.** No correction is needed.

### Non-orthogonal case

When $S_{DL} \neq 0$, the correction to the electron count is:

$$\delta N_L = \mathrm{Tr}(P_{DL}\, S_{LD}) = \frac{1}{2\pi} \int f(E)\, \mathrm{Tr}\!\left(A_{DL}\, S_{LD}\right) dE$$

Substituting $G_{DL}^R = -G_{DD}^R \tau_L g_L^R$:

$$\delta N_L = \frac{-i}{2\pi} \int f(E)\, \mathrm{Tr}\!\left[\left(G_{DD}^R\, \tau_L\, g_L^R - G_{DD}^A\, \tau_L\, g_L^A\right) S_{LD}\right] dE$$

And the total corrected electron count is:

$$\boxed{N_D = \mathrm{Tr}(P_{DD}\, S_{DD}) + \delta N_L + \delta N_R}$$

where $\delta N_R$ follows the same formula with $L \to R$.

---

## 5. Impact on DOS Calculation

The current DOS calculation computes:

$$\mathrm{DOS}(E) = -\frac{1}{\pi}\, \mathrm{Im}\!\left[\mathrm{Tr}(G_{DD}^R)\right]$$

### Exact DOS from the spectral function

The exact device DOS in a non-orthogonal basis is defined through the spectral function:

$$\mathrm{DOS}_D(E) = \frac{1}{2\pi}\, \mathrm{Tr}\!\left(A_{DD}\, S_{DD} + A_{DL}\, S_{LD} + A_{DR}\, S_{RD}\right)$$

where $A_{XY} = i(G_{XY}^R - G_{XY}^A)$. This is the quantity whose integral with $f(E)$ gives the exact electron count $N_D$ from Section 4.

### Approximate DOS using $\mathrm{Im}(G^R)$

For a full (un-restricted) trace, the identity $-(1/\pi)\,\mathrm{Im}\,\mathrm{Tr}(G^R S) = (1/2\pi)\,\mathrm{Tr}(AS)$ holds because $\mathrm{Tr}(G^A S) = [\mathrm{Tr}(G^R S)]^*$ (via Hermiticity of $S$ and the cyclic property of the full trace). However, **this identity does not hold exactly for the device-restricted trace** because cyclic rearrangement under a partial trace is not valid in general.

Despite this, the $\mathrm{Im}(G^R)$ form provides a useful approximation:

$$\mathrm{DOS}_D(E) \approx -\frac{1}{\pi}\, \mathrm{Im}\!\left[\mathrm{Tr}\!\left(G_{DD}^R\, \tilde{S}(E)\right)\right]$$

where the **effective overlap** is:

$$\tilde{S}(E) = S_{DD} - \tau_L\, g_L^R\, S_{LD} - \tau_R\, g_R^R\, S_{RD}$$

This approximation is exact in the orthogonal limit ($S_{DL} = S_{DR} = 0$, $S_{DD} = I$), where it reduces to $\mathrm{DOS} = -\mathrm{Im}[\mathrm{Tr}(G_{DD}^R)] / \pi$.

### Practical recommendation

The DOS in density.py is used only for step-size estimation during the Fermi search (not for electron counting). The $\mathrm{Im}(G^R \tilde{S})$ approximation is sufficient for this purpose. The **electron count correction** (Section 4) must use the exact spectral function formula.

---

## 6. Impact on the Density Matrix

### The density matrix $P_{DD}$ is unchanged

The device density matrix itself:

$$P_{DD} = \frac{1}{2\pi} \int f(E)\, A_{DD}(E)\, dE$$

remains correct as computed. $G_{DD}^R$ already includes both contacts through $\Sigma_L + \Sigma_R$. The spectral function $A_{DD}$ captures the full device-region spectral weight.

### What changes is how we count electrons from $P_{DD}$

The current code computes $N = \mathrm{Tr}(P_{DD} S_{DD})$. The correction adds the leakage terms $\delta N_L + \delta N_R$ that account for electrons shared between device and lead orbitals through basis function overlap.

### When does $P_{DD}$ itself need correction?

For the SCF cycle, the charge density in the device region is:

$$\rho(\mathbf{r}) = \sum_{\mu, \nu \in D} (P_{DD})_{\mu\nu}\, \phi_\mu(\mathbf{r})\, \phi_\nu(\mathbf{r}) + \sum_{\mu \in D, \nu \in L} (P_{DL})_{\mu\nu}\, \phi_\mu(\mathbf{r})\, \phi_\nu(\mathbf{r}) + \text{h.c.} + \ldots$$

The cross-terms involve lead basis functions $\phi_\nu$ evaluated in the device region. In our DFT-NEGF workflow, the DFT calculation does **not** have access to $P_{DL}$ -- the Fock and overlap matrices are constructed purely from $P_{DD}$. This means the cross-terms in the charge density are missing from the SCF entirely.

For the current workflow, this is acceptable: the $P_{DL}$ contribution to real-space charge density is a second-order effect in the coupling strength and decays with distance from the lead surface. The dominant error from neglecting cross-terms is in the **electron count** (Fermi search), not in the density matrix feeding the DFT.

If self-consistent contacts are used (updating lead parameters from the Fock matrix via `setF()`), the missing $P_{DL}$ in the charge density could introduce a small systematic error in the self-consistent cycle. This is expected to be small but should be monitored.

---

## 7. Summary of Required Changes

### Must fix (affects Fermi search accuracy)

1. **Electron count correction:** Add $\delta N_L + \delta N_R$ to the electron count in all Fermi search methods (`calcFermi`, `calcFermiBisect`, `calcFermiSecant`, `calcFermiMuller`, `calcFermiPolyFit`, `getFermiContact`)

2. **DOS correction (optional):** Update `_compute_dos_at_energy` to use the overlap-weighted trace $\mathrm{Tr}(G_{DD}^R \tilde{S})$ instead of $\mathrm{Tr}(G_{DD}^R)$. This is only used for step-size estimation in the Fermi search, so the current formula is acceptable as an approximation.

### Implementation approach: co-accumulated contour integral

The cross-term correction must be computed alongside the main $G_{DD}^R$ integration, not as a post-hoc correction. The Fermi search needs the corrected electron count at every trial $\mu$ to converge to the correct Fermi energy.

#### Scalar reduction via symmetrized $Q$ matrix

Define the **symmetrized cross-term overlap matrix** for contact $\alpha$:

$$Q_\alpha^{\mathrm{sym}}(E) = \frac{1}{2}\left(\tau_\alpha\, g_\alpha^R\, S_{LD,\alpha} + S_{DL,\alpha}\, g_\alpha^R\, \bar{\tau}_\alpha\right)$$

where $S_{LD,\alpha} = S_{DL,\alpha}^\dagger$ and $\bar{\tau}_\alpha = E\, S_{LD,\alpha} - H_{LD,\alpha}$ (unconjugated $E$, from the equation of motion). This is an $(N_D \times N_D)$ matrix (device-size), sparse -- nonzero only on the contact orbital indices.

The cross-term electron count becomes:

$$\delta N_\alpha = -\frac{1}{\pi}\, \mathrm{Im}\!\left[\sum_k w_k\, \mathrm{Tr}\!\left(G_{DD}^R(z_k)\, Q_\alpha^{\mathrm{sym}}(z_k)\right)\right]$$

**Proof:** Starting from $\delta N_\alpha = \mathrm{Tr}(P_{DL}\, S_{LD})$ and the contour integral formula for the off-diagonal block, $P_{DL} = \frac{-i}{2\pi}(\text{lineInt}_{DL} - \text{lineInt}_{LD}^\dagger)$:

$$\delta N_\alpha = \frac{-i}{2\pi}\left[\mathrm{Tr}(\text{lineInt}_{DL}\, S_{LD}) - \mathrm{Tr}(\text{lineInt}_{LD}^\dagger\, S_{LD})\right]$$

**Important:** The second term involves $\text{lineInt}_{LD}^\dagger$, NOT $\text{lineInt}_{DL}^\dagger$. These are different blocks of the full system line integral, related by:

$$\text{lineInt}_{DL} = -\sum_k w_k\, G_{DD}^R\, \tau\, g^R, \qquad \text{lineInt}_{LD} = -\sum_k w_k\, g^R\, \bar{\tau}\, G_{DD}^R$$

Define:

$$c_{DL} = \mathrm{Tr}(\text{lineInt}_{DL}\, S_{LD}) = -\sum_k w_k\, \mathrm{Tr}(G_{DD}^R\, Q), \qquad Q = \tau\, g^R\, S_{LD}$$

For the second term, using $\mathrm{Tr}(A^\dagger B) = [\mathrm{Tr}(B^\dagger A)]^*$:

$$\mathrm{Tr}(\text{lineInt}_{LD}^\dagger\, S_{LD}) = \left[\mathrm{Tr}(S_{DL}\, \text{lineInt}_{LD})\right]^* = \left[-\sum_k w_k\, \mathrm{Tr}(G_{DD}^R\, Q_{\mathrm{rev}})\right]^*$$

where $Q_{\mathrm{rev}} = S_{DL}\, g^R\, \bar{\tau}$ (note the reversed order compared to $Q$). This gives:

$$\delta N_\alpha = \frac{-i}{2\pi}\left(-c_{Q} + c_{Q_{\mathrm{rev}}}^*\right)$$

Expanding with $c_Q = a + bi$ and $c_{Q_{\mathrm{rev}}} = c + di$:

$$\frac{-i}{2\pi}\left(-c_Q + c_{Q_{\mathrm{rev}}}^*\right) = \frac{-i}{2\pi}\left[(c - a) + (-b - d)i\right] = \frac{1}{2\pi}\left[-(b + d) + i(a - c)\right]$$

This expression has two components:

- **Real part** $= -(b + d)/(2\pi)$: the physical $\delta N$ (electron count correction)
- **Imaginary part** $= (a - c)/(2\pi)$: must vanish for $\delta N$ to be real, requiring $\mathrm{Re}(c_Q) = \mathrm{Re}(c_{Q_{\mathrm{rev}}})$

Extracting only the electron count:

$$\delta N_\alpha = -\frac{1}{2\pi}\left[\mathrm{Im}(c_Q) + \mathrm{Im}(c_{Q_{\mathrm{rev}}})\right] = -\frac{1}{2\pi}\,\mathrm{Im}\!\left[c_Q + c_{Q_{\mathrm{rev}}}\right]$$

$$= -\frac{1}{2\pi}\,\mathrm{Im}\!\left[\sum_k w_k\, \mathrm{Tr}\!\left(G_{DD}^R\,(Q + Q_{\mathrm{rev}})\right)\right]$$

Defining $Q^{\mathrm{sym}} = (Q + Q_{\mathrm{rev}})/2$:

$$\boxed{\delta N_\alpha = -\frac{1}{\pi}\, \mathrm{Im}\!\left[\sum_k w_k\, \mathrm{Tr}\!\left(G_{DD}^R(z_k)\, Q_\alpha^{\mathrm{sym}}(z_k)\right)\right]}$$

**Why only $G^R$ is needed (no $G^A$):** The real-axis spectral formula $A_{DL} = i(G_{DL}^R - G_{DL}^A)$ involves both retarded and advanced Green's functions. However, the complex contour integral avoids this: $\text{lineInt}_{DL} = \oint_C f(z)\, G_{DL}^R(z)\, dz$ uses only $G^R$ (analytic in the upper half-plane), while the $G^A$ contribution is captured by $\text{lineInt}_{LD}^\dagger$ through contour deformation to the lower half-plane. The symmetrized $Q^{\mathrm{sym}}$ encodes both blocks (DL from $Q_{\mathrm{fwd}}$, LD from $Q_{\mathrm{rev}}$), so no explicit $G^A$ or $g^A$ construction is needed.

**Why $\mathrm{Re}(c_{\mathrm{sym}}) \neq 0$ is expected:** The accumulated scalar $c_{\mathrm{sym}} = \sum_k w_k\, \mathrm{Tr}(G_{DD}^R\, Q^{\mathrm{sym}})$ has $\mathrm{Re}(c_{\mathrm{sym}}) = (a + c)/2$, which is generically nonzero. This is the same phenomenon as the main DD block: $\mathrm{Tr}(\text{lineInt}_{DD}\, S_{DD})$ also has a nonzero real part that does not contribute to $N_{DD} = (1/\pi)\,\mathrm{Im}[\mathrm{Tr}(\text{lineInt}_{DD}\, S_{DD})]$. For the DD block, the Re cancels because $\text{lineInt}_{DD}^\dagger$ is the conjugate of the *same* block. For the cross-term, $\text{lineInt}_{DL}$ and $\text{lineInt}_{LD}$ are *different* blocks, so the Re parts add (giving $a + c$) rather than cancel. The $\mathrm{Im}[\cdot]$ extraction in the boxed formula correctly discards this non-physical component. Since $c_{\mathrm{sym}}$ is a scalar (the trace has already been taken), $\mathrm{Im}(c_{\mathrm{sym}})$ is equivalent to `cross_scalar.imag` in code.

**Why symmetrization is needed:** On the real axis, $Q = Q_{\mathrm{rev}}$ only when all matrices commute (scalar/1x1 case). In the general matrix case, $\tau\, g^R\, S_{LD} \neq S_{DL}\, g^R\, \bar{\tau}$ because the coupling $\tau$ and overlap $S_{DL}$ do not commute with $g^R$. The naive formula using unsymmetrized $Q$ would be exact on the real axis only if the Im[...] trick were valid, but on the complex contour, the DL and LD blocks of the line integral are independent and must both be accounted for.

**Orthogonal limit check:** When $S_{DL} = 0$, both $Q$ and $Q_{\mathrm{rev}}$ vanish (since $\tau = -H_{DL}$ and $S_{LD} = 0$), giving $\delta N = 0$ as expected.

This reduces the cross-term accumulation to a **single complex scalar** per energy point (summed over contacts), rather than accumulating full $G_{DL}^R$ matrices.

#### Integration co-accumulation

At each energy point in `GrInt`, alongside accumulating $w_k \cdot G_{DD}^R$:

1. Call `g.crossTermQTot(E)` to get $Q^{\mathrm{sym}}_{\mathrm{tot}}$ (sum over all contacts)
2. Form $\mathrm{Tr}(G_{DD}^R \cdot Q^{\mathrm{sym}}_{\mathrm{tot}})$ -- cheap since $Q$ is sparse on contact indices
3. Accumulate the weighted scalar: `cross_accum += w_k * Tr(G_R @ Q_tot)`

The scan accumulator in `_GInt` changes from a single matrix to a `(matrix, scalar)` tuple. Both the `vmap` and `lax.scan` paths need to handle the tuple return. Density functions return `(P, delta_N)` where `delta_N = -(1/pi) * Im(cross_accum)`.

#### What changes in the Fermi search

All Fermi search methods currently compute:

```python
N_curr = np.trace(P @ g.S).real
```

This becomes:

```python
N_curr = np.trace(P @ g.S).real + delta_N
```

where `delta_N` is the second return value from the density function.

### Pre-existing bug found during review

In `calcFermiBisect` (density.py line 1181), the arguments to `_compute_dos_at_energy` have F and S swapped:

```python
# Current (WRONG):
dos = _compute_dos_at_energy(E, g.S, g.F, g.sigmaTot(E))
# Should be:
dos = _compute_dos_at_energy(E, g.F, g.S, g.sigmaTot(E))
```

This affects the DOS-based step-size estimation during bounds finding but not the final Fermi energy accuracy (since it only controls step sizes).

### Not needed for current workflow

- Full $P_{DL}$ matrix storage (only the scalar $\mathrm{Tr}(P_{DL} S_{LD})$ is needed)
- Density matrix correction for SCF ($P_{DD}$ feeds DFT, DFT does not use $P_{DL}$)

---

## 8. surfG Interface Specification

### Problem: extracting cross-term data from existing classes

The cross-term requires $Q_\alpha^{\mathrm{sym}} = \frac{1}{2}(\tau_\alpha \, g_\alpha^R \, S_{LD,\alpha} + S_{DL,\alpha} \, g_\alpha^R \, \bar{\tau}_\alpha)$ at each energy point. Currently, $\tau$ and $g$ are consumed internally within `sigma()` and discarded. We cannot recover them from $\Sigma = \tau\, g\, \bar{\tau}$ without knowing $\tau$ separately (the factorization is not unique).

Each surfG class computes $g$ differently:

| Class | How $g_\alpha^R$ is obtained | How $\tau$ is built |
|-------|----------------------------|---------------------|
| `surfG` (1D) | Iterative Dyson: `self.g(E, i)` | `E*stau - tau` (or `-tau` if orthogonal) |
| `surfGAt3D` (3D) | 2D k-mesh inverse FT: `gSurf(E)` returns $G_{AB}$ propagator | `[(E+i*eta)*Slist[a] - Vlist[a]]` concatenated over active dirs |
| `surfGBAt` (Bethe) | Implicit in self-consistent iteration: `inv(A - sigTot + sigK[pair])` | `(E+i*eta)*Slist[k] - Vlist[k]` per direction |
| `surfGTest` | None (constant sigma) | None |

A new method is needed. Each class implements it using its own internal representation.

### Two-level Protocol design

There are two levels of surfG objects:

1. **Atomic-level** (`surfGAt3D`, `surfGBAt`): compute self-energy for a single atom's orbital space (typically $9 \times 9$ for $s$+$p$+$d$)
2. **Wrapper-level** (`surfG`, `surfG3`, `surfGB`, `surfGTest`): map atomic results to the full device basis; this is what `density.py` calls

Both levels get a `crossTermQ` method.

#### `SurfGAtomicProtocol` (atomic level)

```python
from typing import Protocol, Optional
import numpy as np

class SurfGAtomicProtocol(Protocol):
    """Protocol for atomic-level surface Green's function calculators."""

    def sigma(self, E: complex, ...) -> np.ndarray:
        """Self-energy in the atom's orbital basis."""
        ...

    def crossTermQ(self, E: complex, ...) -> Optional[np.ndarray]:
        """Symmetrized cross-term matrix Q_sym in the atom's orbital basis.

        Q_sym = (tau @ g_surf @ S_LD + S_DL @ g_surf @ bar_tau) / 2

        where bar_tau = E * S_LD - H_LD (unconjugated E, from the equation of motion).

        Returns
        -------
        Q_sym : ndarray of shape (dim, dim) or None
            None if the contact basis is orthogonal (S_DL = 0).
            Otherwise, Q_sym such that the cross-term trace contribution is
            Tr(G_DD^R @ Q_sym) at this energy point.
        """
        ...
```

**Implementations:**

- `surfGAt3D.crossTermQ(E, active_dirs, conv, mix, G_AB)`:
  ```
  tau = hstack([(E+i*eta)*Slist[a] - Vlist[a] for a in active_dirs])  # (dim, dim*n_dirs)
  G_sub = G_AB[active slice]                                           # (dim*n_dirs, dim*n_dirs)
  T = tau @ G_sub                                                      # (dim, dim*n_dirs)
  S_LD = vstack([Slist[a].conj().T for a in active_dirs])              # (dim*n_dirs, dim)
  S_DL = hstack([Slist[a] for a in active_dirs])                       # (dim, dim*n_dirs)
  tau_dag = vstack([(E+i*eta).conj()*Slist[a].conj().T - Vlist[a].conj().T
                     for a in active_dirs])                             # (dim*n_dirs, dim)
  Q_fwd = T @ S_LD                                                     # (dim, dim)
  Q_rev = S_DL @ G_sub @ tau_dag                                       # (dim, dim)
  return (Q_fwd + Q_rev) / 2
  ```
  Note: `G_AB` can be passed in (already computed for `sigma`) to avoid recomputation.

- `surfGBAt.crossTermQ(E, conv, mix)`:
  The surface self-consistent iteration in `sigma()` converges a set of surface
  self-energies `sigSurf[k]` for $k \in [0..8]$. After convergence, all directions
  share a single Green's function `g_surf = inv(A - sum(sigSurf))`:
  ```
  # After sigma() convergence:
  sigTot = sum(sigSurf[k] for k in surface_dirs)   # converged total
  g_surf = inv(A - sigTot)                          # single GF, shared by all dirs
  Q = zeros(dim, dim)
  for k in surface_dirs:
      B_k = (E+i*eta)*Slist[k] - Vlist[k]
      bar_B_k = (E+i*eta)*Slist[k].conj().T - Vlist[k].conj().T  # bar_tau (unconjugated E)
      Q += (B_k @ g_surf @ Slist[k].conj().T + Slist[k] @ g_surf @ bar_B_k) / 2
  return Q
  ```

#### `SurfGProtocol` (wrapper level -- density.py interface)

```python
class SurfGProtocol(Protocol):
    """Protocol for wrapper-level surface Green's function calculators.

    This is the interface that density.py interacts with.
    """

    F: np.ndarray   # Fock matrix (N x N)
    S: np.ndarray   # Overlap matrix (N x N)

    def sigma(self, E: complex, i: int, conv: float = ...) -> np.ndarray:
        """Self-energy for contact i in full device basis (N x N)."""
        ...

    def sigmaTot(self, E: complex, conv: float = ...) -> np.ndarray:
        """Total self-energy from all contacts (N x N)."""
        ...

    def setF(self, F: np.ndarray, mu1: float, mu2: float) -> None:
        """Update Fock matrix and contact chemical potentials."""
        ...

    @property
    def num_contacts(self) -> int:
        """Number of contacts."""
        ...

    def crossTermQ(self, E: complex, i: int, conv: float = ...) -> Optional[np.ndarray]:
        """Symmetrized cross-term overlap matrix Q_sym_i in full device basis (N x N).

        Q_sym = (tau @ g_surf @ S_LD + S_DL @ g_surf @ bar_tau) / 2

        where bar_tau = E * S_LD - H_LD (unconjugated E, from the equation of motion).

        Returns
        -------
        Q_sym : ndarray of shape (N, N) or None
            None if contact i has orthogonal coupling (S_DL = 0).
            Otherwise, Q_sym_i embedded in the full device basis at
            the appropriate orbital indices.

            Q_sym is sparse: nonzero only on the contact orbital block
            (indices given by indsList[i]).

        Notes
        -----
        The cross-term electron count correction is:
            delta_N_i = -(1/pi) * Im(sum_k w_k * Tr(G_DD^R(z_k) @ Q_sym_i(z_k)))
        accumulated during contour integration alongside G_DD^R.

        Symmetrization is required because on the complex contour,
        tau @ g^R @ S_LD != S_DL @ g^R @ bar_tau in general.
        See Section 7 proof for derivation.
        """
        ...

    def crossTermQTot(self, E: complex, conv: float = ...) -> Optional[np.ndarray]:
        """Sum of Q_sym over all contacts. Convenience for integration loop.

        Returns sum_i crossTermQ(E, i), or None if all contacts are orthogonal.
        This is the method called by GrInt at each energy point.
        """
        ...
```

### Per-class implementation notes

#### `surfG` (1D) -- `crossTermQ(E, i)`

```
if stauList[i] is None:
    return None                          # orthogonal contact
stau = stauList[i]                       # S_DL, shape (n, n)
tau_raw = tauList[i]                     # H_DL, shape (n, n)
t = E * stau - tau_raw                   # energy-dependent coupling
t_reg = t @ CList[i][len(t):-len(t), len(t):-len(t)]  # regularized (same as sigma)
g_surf = self.g(E, i)                   # surface Green's function (n, n)
T = t_reg @ g_surf                       # (n, n)
Q_fwd = T @ stau.conj().T               # tau @ g @ S_LD, shape (n, n)
bar_t = E * stau.conj().T - tau.conj().T     # bar_tau (unconjugated E)
bar_t_reg = C_mid.conj().T @ bar_t           # regularized bar_tau
Q_rev = stau @ g_surf @ bar_t_reg            # S_DL @ g @ bar_tau, shape (n, n)
Q_raw = (Q_fwd + Q_rev) / 2             # symmetrized
Q = zeros(N, N, dtype=complex)
Q[ix_(inds, inds)] = Q_raw              # embed in full device basis
return Q
```

#### `surfG3` (3D wrapper) -- `crossTermQ(E, i)`

```
# For contact i, loop over atoms and assemble atomic Q blocks
# (same pattern as surfG3.sigma which loops over atoms)
Q = zeros(N, N, dtype=complex)
for atom_index, (nInds, Finds) in enumerate(zip(nIndLists_i, indsLists_i)):
    active_dirs = <neighbor directions for this atom>
    Q_atomic = gList[atom_index].crossTermQ(E, active_dirs, ...)  # (dim, dim)
    if Q_atomic is not None:
        Q[ix_(Finds, Finds)] += Q_atomic

# De-orthonormalize AFTER assembling all atoms (matches sigma pattern):
if Sdict['sss'] == 0:   # orthonormal Slater-Koster params
    Q = Xi @ Q @ Xi
return Q if any nonzero else None
```

Note: The de-orthonormalization must apply to the full assembled matrix, NOT per-atom. This matches how `surfG3.sigma()` works: it assembles all atomic self-energies first, then applies `Xi @ sig @ Xi` to the full result.

#### `surfGB` (Bethe wrapper) -- `crossTermQ(E, i)`

Same pattern as `surfG3`: get `Q_atomic` from `surfGBAt`, map to full device basis with de-orthonormalization.

#### `surfGTest` (tester) -- `crossTermQ(E, i)`

```
return None    # constant sigma, no overlap, orthogonal test case
```

### Avoiding redundant computation

For `surfGAt3D` and `surfGBAt`, computing `crossTermQ` requires the same intermediates as `sigma()` (the surface Green's function and coupling matrices). Two approaches to avoid double computation:

1. **Cache intermediates in `sigma()`:** Have `sigma()` store the surface Green's function (or $G_{AB}$) and coupling matrices as instance state, then `crossTermQ()` reuses them. Requires that `crossTermQ` is always called at the same energy as the preceding `sigma()` call.

2. **Combined method:** `sigmaAndCrossTermQ(E, i)` returns `(sigma, Q)` in a single call, sharing the Dyson iteration. This is cleaner but changes the calling pattern.

Recommendation: approach 1 (caching) is simpler to implement and does not require changing how `sigma()` is called. The cache only needs to hold the last energy point's data. For `surfGAt3D`, `G_AB` can be passed as an optional parameter (already supported: `sigma(E, active_dirs, ..., G_AB=None)`).

**JAX/JIT consideration:** Several `sigma()` methods are JIT-compiled (`surfG1D.sigma` at line 311, `surfGBAt.sigma` at line 947). Mutable instance state cannot be updated inside a JIT-traced function. The caching must happen outside the JIT boundary -- either by structuring `crossTermQ` as a separate non-JIT call that reuses cached intermediates stored by the JIT'd `sigma()` via `jax.experimental.io_callback`, or by having the integration loop call `sigma()` and `crossTermQ()` sequentially outside JIT (the simpler approach, since the integration loop itself is not JIT-compiled).

### Performance analysis

Per energy point, per contact:
- $Q^{\mathrm{sym}}$ computation: four matrix multiplies ($\tau \cdot g$, $T \cdot S_{LD}$, $S_{DL} \cdot g$, $\cdot \bar{\tau}$), plus averaging -- roughly 2x the cost of a single self-energy
- For surfG1D: $(n \times n)$ matrices where $n$ = contact orbital count (small)
- For surfGAt3D/surfGBAt: $(9 \times 9)$ matrices (negligible)
- $\mathrm{Tr}(G_{DD}^R \cdot Q^{\mathrm{sym}})$: $O(N \cdot n)$ since $Q$ is sparse on contact indices
- The dominant cost remains the $N \times N$ inversion for $G_{DD}^R$
- Total overhead: approximately 2 extra `sigma()`-equivalent computations per energy point (for the symmetrized $Q$)

---

## 9. JIT Recompilation Fix: Fermi Shift as Energy Shift

### Problem

During the NEGF-DFT self-consistent cycle, the Fermi energy shifts at each iteration. Currently, `updateH(fermi)` in `surfGAt3D` and `surfGBAt` mutates `self.H` and `self.Vlist`:

```python
# surfGBAt.updateH (line 970-978):
self.H = self.H + dFermi * jnp.eye(self.dim)
for j, S in enumerate(self.Slist):
    self.Vlist[j] = self.Vlist[j] + dFermi * S
```

Since `sigma()` and `sigmaK()` are JIT-compiled with `self` captured in the closure (lines 946-947), mutating `self.H` and `self.Vlist` changes the closure constants, forcing JAX to retrace the entire function on every SCF iteration. For the contour integration with ~100 energy points, each requiring a Dyson iteration, this recompilation cost dominates the SCF wall time after the first cycle.

### Mathematical equivalence: Fermi shift = energy shift

The Fermi shift applies uniformly to all orbitals:

$$H \to H_0 + \delta\mu \cdot I, \qquad V_k \to V_{0,k} + \delta\mu \cdot S_k$$

where $\delta\mu = \mu_{\mathrm{current}} - \mu_{\mathrm{ref}}$ and $H_0$, $V_{0,k}$ are the reference (initial) values. Substituting into the key matrices:

**Onsite term $A$** (used in `sigmaK`, `gSurf`, `gBulk`):

$$A = (E + i\eta)I - H = (E + i\eta)I - (H_0 + \delta\mu\, I) = (E - \delta\mu + i\eta)I - H_0$$

**Coupling term $B$** (used in Dyson iteration and self-energy):

$$B_k = (E + i\eta)S_k - V_k = (E + i\eta)S_k - (V_{0,k} + \delta\mu\, S_k) = (E - \delta\mu + i\eta)S_k - V_{0,k}$$

**Coupling $\tau$** (used in `sigma`):

$$\tau_a = (E + i\eta)S_a - V_a = (E - \delta\mu + i\eta)S_a - V_{0,a}$$

In every case, the Fermi shift is **exactly equivalent to replacing $E$ with $E - \delta\mu$**. This is a gauge shift: the physics is invariant under a uniform energy reference change.

This equivalence extends to the k-space sums in `gSurf` and `gBulk`, since the shift applies identically at every k-point:

$$F_{ak} = \sum_{\text{in-plane}} e^{ik \cdot R_j} V_j + H \quad\to\quad F_{ak} + \delta\mu\, S_{ak}$$

$$A_k = (E + i\eta)S_{ak} - F_{ak} \quad\to\quad (E - \delta\mu + i\eta)S_{ak} - F_{0,ak}$$

### Implementation

#### Atomic-level classes (`surfGAt3D`, `surfGBAt`)

1. **Store reference values as immutable:** `self.H0`, `self.Vlist0`, `self.fermi0` are set once in `__init__` and never modified.

2. **`updateH(fermi)` becomes trivial:**
   ```python
   def updateH(self, fermi=None):
       if fermi is not None:
           self.dFermi = fermi - self.fermi0
       # Rebuild H0x/S0x for extended system (uses reference values)
       H0x = jnp.kron(jnp.eye(self.NN+1), self.H0)
       ...
   ```
   No matrix mutation. `self.dFermi` is a plain float, not a JAX array, so it doesn't affect the JIT trace.

3. **JIT-compiled methods are unchanged** -- they continue to use `self.H0` (constant) and `E` (traced argument). Since `self.H0` never changes, the JIT cache is never invalidated.

#### Wrapper-level classes (`surfG3`, `surfGB`)

The wrapper shifts E before calling atomic methods:

```python
class surfG3:
    def sigma(self, E, i, conv=1e-4):
        ...
        E_shifted = E - self.gList[atom_idx].dFermi
        sigAtom = self.gList[atom_idx].sigma(E_shifted, active_dirs, conv=conv, G_AB=G_AB)
        ...
```

This applies uniformly to all atomic-level calls: `sigma()`, `sigmaK()`, `gSurf()`, `gBulk()`, `sigmaBulk()`, `DOS()`, and the new `crossTermQ()`.

#### Protocol addition

The `SurfGAtomicProtocol` gains a `dFermi` attribute:

```python
class SurfGAtomicProtocol(Protocol):
    H0: np.ndarray           # reference onsite Hamiltonian (immutable)
    dFermi: float             # current Fermi shift from reference

    def updateH(self, fermi: float = None) -> None:
        """Update Fermi shift. Does NOT mutate H0 or Vlist0."""
        ...

    def sigma(self, E: complex, ...) -> np.ndarray:
        """Self-energy at energy E (caller responsible for shifting E by dFermi)."""
        ...

    def crossTermQ(self, E: complex, ...) -> Optional[np.ndarray]:
        """Cross-term Q_sym at energy E (caller responsible for shifting E by dFermi)."""
        ...
```

The wrapper `SurfGProtocol` handles the shift internally -- `density.py` passes the physical energy E, and the wrapper subtracts `dFermi` before calling atomic methods. Density.py does not need to know about `dFermi`.

### Impact on `crossTermQ`

The cross-term matrices use the same coupling $\tau$ and surface Green's function $g^R$ as `sigma()`. With the energy-shift approach:

$$Q^{\mathrm{sym}}(E) = \frac{1}{2}\left(\tau(E_{\mathrm{eff}})\, g^R(E_{\mathrm{eff}})\, S_{LD} + S_{DL}\, g^R(E_{\mathrm{eff}})\, \bar{\tau}(E_{\mathrm{eff}})\right)$$

where $E_{\mathrm{eff}} = E - \delta\mu$. Since the wrapper already shifts E for `sigma()`, the same shifted E is passed to `crossTermQ()`. No additional logic is needed.

### Classes NOT affected

- **`surfG` (1D):** Does not JIT-compile `sigma()`. The Fermi shift is handled differently (contact parameters extracted from the Fock matrix via `setF`). No change needed.
- **`surfGTest`:** Constant self-energy, no Fermi shift. No change needed.

### Performance improvement

| SCF iteration | Current (with recompilation) | After fix |
|---------------|------------------------------|-----------|
| 1st           | Full JIT trace + execution   | Full JIT trace + execution |
| 2nd+          | Full JIT retrace + execution | Cached execution only |

The JIT trace for the Bethe self-consistent iteration (`sigmaK` with `lax.while_loop`) and the k-space surface Green's function (`gSurf` with `vmap` over k-points) are the most expensive compilations. Eliminating retracing after the first SCF cycle should provide a significant speedup for NEGF-DFT calculations.
