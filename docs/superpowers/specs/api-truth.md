# GauNEGF API Truth Table

Generated: 2026-05-09
Source: gauNEGF/*.py
Method: AST-equivalent inspection by haiku agent

## Per-Module Detail

### Module: gauNEGF.scf

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 1
- **Public functions:** 0

#### class NEGF

Signature: `NEGF(fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE)`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE)` -- docstring: present
- `runDFT(self, fullSCF=True)` -- docstring: present
- `updateN(self)` -- docstring: present
- `setFock(self, F_)` -- docstring: present
- `setDen(self, P_, enableSpinLock=False, spinLockList=None)` -- docstring: present
- `getHOMOLUMO(self)` -- docstring: present
- `setVoltage(self, qV, fermi=None, Emin=None, Eminf=None)` -- docstring: present
- `setContacts(self, lContact=None, rContact=None)` -- docstring: present
- `setSigma(self, lContact=None, rContact=None, sig=-0.1j, sig2=None)` -- docstring: present
- `getSigma(self, E=0)` -- docstring: present
- `FockToP(self)` -- docstring: present
- `PMix(self, damping, Pulay=False)` -- docstring: present
- `PToFock(self)` -- docstring: present
- `SCF(self, conv=SCF_CONVERGENCE_TOL, damping=SCF_DAMPING, maxcycles=SCF_MAX_CYCLES, checkpoint=True, pulay=True)` -- docstring: present
- `writeChk(self)` -- docstring: present
- `saveMAT(self, matfile="out.mat")` -- docstring: present

Dynamic attributes:
- `self.sigma1` (set by `setSigma`)
- `self.sigma2` (set by `setSigma`)
- `self.sigma12` (set by `setSigma`)
- `self.Gam1` (set by `setSigma`)
- `self.Gam2` (set by `setSigma`)

---

### Module: gauNEGF.scfE

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 1
- **Public functions:** 0

#### class NEGFE

Signature: `NEGFE(fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE)`
Inherits constructor from: NEGF.__init__(fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE)
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `setContactBethe(self, contactList, latFile='Au', eta=ETA, T=TEMPERATURE)` -- docstring: present
  - Parameter shapes: `contactList` is nested list [[atom1, atom2, ...], [atom3, ...]] (one inner list per contact)
- `setContact1D(self, contactList, tauList=None, stauList=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, neList=None, muList=None, eta=ETA, T=TEMPERATURE, symmetrize_contacts=None)` -- docstring: present
- `_symmetrize_F(self)` -- docstring: present
- `setSigma(self, lContact=None, rContact=None, sig=-0.1j, sig2=None, T=TEMPERATURE)` -- docstring: present
- `setVoltage(self, qV, fermi=None, Emin=None, Eminf=None, fermiMethod=None)` -- docstring: present
- `setIntegralLimits(self, N1=None, N2=None, Nnegf=None, tol=ADAPTIVE_INTEGRATION_TOL, Emin=None)` -- docstring: present
- `integralCheck(self, cycles=10, damp=0.02, pauseFermi=False)` -- docstring: present
- `getSigma(self, E, E2=None)` -- docstring: present
- `spawnNEGF(self, mu1=None, mu2=None)` -- docstring: present
- `FockToP(self)` -- docstring: present
- `PToFock(self)` -- docstring: present

Dynamic attributes:
- `self.g` (set by `setContactBethe`, `setContact1D`, `setSigma`)

---

### Module: gauNEGF.density

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 0
- **Public functions:** 29

#### function fermi

Signature: `fermi(E, mu, T)`
Docstring: present
Deprecated: false
Decorators: []

#### function getANTPoints

Signature: `getANTPoints(N)`
Docstring: present
Deprecated: false
Decorators: []

#### function integratePoints

Signature: `integratePoints(computePointFunc, numPoints, parallel=False, numWorkers=None, chunkSize=None, debug=False)`
Docstring: present
Deprecated: false
Decorators: []

#### function integratePointsAdaptiveANT

Signature: `integratePointsAdaptiveANT(computePoint, tol=ADAPTIVE_INTEGRATION_TOL, maxN=MAX_GRID_POINTS, debug=False)`
Docstring: present
Deprecated: false
Decorators: []

#### function density

Signature: `density(V, Vc, D, Gam, Emin, mu)`
Docstring: present
Deprecated: false
Decorators: [@jit]

#### function bisectFermi

Signature: `bisectFermi(V, Vc, D, Gam, Nexp, conv=FERMI_CALCULATION_TOL, Eminf=ENERGY_MIN)`
Docstring: present
Deprecated: false
Decorators: []

#### function densityRealN

Signature: `densityRealN(F, S, g, Emin, mu, N=100, T=TEMPERATURE, showText=True)`
Docstring: present
Deprecated: false
Decorators: []

#### function densityReal

Signature: `densityReal(F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False)`
Docstring: present
Deprecated: false
Decorators: []

#### function densityGridN

Signature: `densityGridN(F, S, g, mu1, mu2, ind=None, N=100, T=TEMPERATURE, showText=True)`
Docstring: present
Deprecated: false
Decorators: []

#### function densityGridTrap

Signature: `densityGridTrap(F, S, g, mu1, mu2, ind=None, N=100, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

#### function densityGrid

Signature: `densityGrid(F, S, g, mu1, mu2, ind=None, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False)`
Docstring: present
Deprecated: false
Decorators: []

#### function densityComplexN

Signature: `densityComplexN(F, S, g, Emin, mu, N=100, T=TEMPERATURE, showText=True, method='ant')`
Docstring: present
Deprecated: false
Decorators: []

#### function densityComplex

Signature: `densityComplex(F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcEmin

Signature: `calcEmin(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES, Emin=None)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcTSW

Signature: `calcTSW(F, S, g, tol=FERMI_CALCULATION_TOL, maxN=FERMI_SEARCH_CYCLES, Eminf=None, TSW=None)`
Docstring: present
Deprecated: false
Decorators: []

#### function integralFit

Signature: `integralFit(F, S, g, mu, Eminf=ENERGY_MIN, tol=FERMI_CALCULATION_TOL, T=TEMPERATURE, maxN=MAX_CYCLES)`
Docstring: present
Deprecated: false
Decorators: []

#### function integralFitNEGF

Signature: `integralFitNEGF(F, S, g, fermi, qV, Eminf=ENERGY_MIN, tol=FERMI_CALCULATION_TOL, T=TEMPERATURE, maxGrid=MAX_GRID_POINTS)`
Docstring: present
Deprecated: false
Decorators: []

#### function getFermiContact

Signature: `getFermiContact(g, ne, Emin=None, lBound=None, uBound=None, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcFermi

Signature: `calcFermi(g, ne, Emin, Ef, lBound=None, uBound=None, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcFermiBisect

Signature: `calcFermiBisect(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE, uBound=None, lBound=None)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcFermiSecant

Signature: `calcFermiSecant(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcFermiMuller

Signature: `calcFermiMuller(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

#### function calcFermiPolyFit

Signature: `calcFermiPolyFit(g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE, order=3)`
Docstring: present
Deprecated: false
Decorators: []

---

### Module: gauNEGF.transport

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 1
- **Public functions:** 18

#### class SigmaCalculator

Signature: `SigmaCalculator(sig1, sig2=None, energy_dependent=None)`
Docstring: present
Deprecated: false
Decorators: []
Overloads: `sig1` accepts ndarray (static self-energy matrix) OR surfG-type object (duck-typed via hasattr('sigma', 'sigmaTot') -> energy_dependent=True branch)

Methods:
- `__init__(self, sig1, sig2=None, energy_dependent=None)` -- docstring: present
- `get_sigma_total(self, E, spin=None, matrix_size=None)` -- docstring: present
- `get_Q_tot(self, E, spin=None, matrix_size=None)` -- docstring: present
- `get_sigma(self, E, contact_index, spin=None, matrix_size=None)` -- docstring: present
- `get_gamma(self, E, contact_index, spin=None, matrix_size=None)` -- docstring: present

#### function transmission_single_energy

Signature: `transmission_single_energy(E, F_jax, S_jax, sigma_calc, spin=None)`
Docstring: present
Deprecated: false
Decorators: []

#### function dos_single_energy

Signature: `dos_single_energy(E, F_jax, S_jax, sigma_calc, spin=None)`
Docstring: present
Deprecated: false
Decorators: []

#### function calculate_transmission

Signature: `calculate_transmission(F, S, sigma_calculator, energy_list, spin=None, checkpoint_file=None, checkpoint_interval=10)`
Docstring: present
Deprecated: false
Decorators: []
Variable returns: spin_trans is None -> transmission (array); spin_trans is not None -> (transmission, spin_trans) 2-tuple

#### function calculate_dos

Signature: `calculate_dos(F, S, sigma_calculator, energy_list, spin=None, checkpoint_file=None, checkpoint_interval=10)`
Docstring: present
Deprecated: false
Decorators: []
Variable returns: dos_spin is None -> (dos_total, dos_per_site) 2-tuple; dos_spin is not None -> (dos_total, dos_per_site, dos_spin) 3-tuple

#### function calculate_current

Signature: `calculate_current(F, S, sigma_calculator, fermi, qV, T=TEMPERATURE, spin=None, dE=ENERGY_STEP, **kwargs)`
Docstring: present
Deprecated: false
Decorators: []

#### function current

Signature: `current(F, S, sig1, sig2, fermi, qV, T=TEMPERATURE, spin="r", dE=ENERGY_STEP)`
Docstring: present
Deprecated: false
Decorators: []

#### function currentSpin

Signature: `currentSpin(F, S, sig1, sig2, fermi, qV, T=TEMPERATURE, spin="r", dE=ENERGY_STEP)`
Docstring: present
Deprecated: false
Decorators: []

#### function currentE

Signature: `currentE(F, S, g, fermi, qV, T=TEMPERATURE, spin="r", dE=ENERGY_STEP)`
Docstring: present
Deprecated: false
Decorators: []

#### function currentF

Signature: `currentF(fn, dE=ENERGY_STEP, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

#### function cohTrans

Signature: `cohTrans(Elist, F, S, sig1, sig2)`
Docstring: present
Deprecated: false
Decorators: []

#### function cohTransSpin

Signature: `cohTransSpin(Elist, F, S, sig1, sig2, spin='u')`
Docstring: present
Deprecated: false
Decorators: []

#### function DOS

Signature: `DOS(Elist, F, S, sig1, sig2)`
Docstring: present
Deprecated: false
Decorators: []

#### function cohTransE

Signature: `cohTransE(Elist, F, S, g)`
Docstring: present
Deprecated: false
Decorators: []

#### function cohTransSpinE

Signature: `cohTransSpinE(Elist, F, S, g, spin='u')`
Docstring: present
Deprecated: false
Decorators: []

#### function DOSE

Signature: `DOSE(Elist, F, S, g)`
Docstring: present
Deprecated: false
Decorators: []

---

### Module: gauNEGF.surfG1D

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 1
- **Public functions:** 0

#### class surfG

Signature: `surfG(Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA, spin='r')`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA, spin='r')` -- docstring: present
- `_setContacts(self, alphas=None, aOverlaps=None, betas=None, bOverlaps=None)` -- docstring: present
- `_regularizeContacts(self)` -- docstring: present
- `_rejit(self)` -- docstring: present
- `g(self, E, i, conv=SURFACE_GREEN_CONVERGENCE, relFactor=0.5)` -- docstring: present
- `setF(self, F, mu1=None, mu2=None)` -- docstring: present
- `sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present

---

### Module: gauNEGF.surfG3D

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 2
- **Public functions:** 0

#### class surfG3

Signature: `surfG3(F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE)` -- docstring: present
- `genNeighbors(self, plane_normal, first_neighbor)` -- docstring: present
- `readBetheParams(self, filename)` -- docstring: present
- `constructMat(self, Mdict, dirCosines, SOC=False)` -- docstring: present
- `sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `getSigma(self, Elist=[None, None], conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `updateFermi(self, i, Ef)` -- docstring: present
- `setF(self, F, muL, muR)` -- docstring: present
- `testDOrbitalFunctions(self)` -- docstring: present
- `testDOrbitalSymmetry(self)` -- docstring: present
- `testPDInteraction(self)` -- docstring: present
- `testDDInteraction(self)` -- docstring: present
- `testHoppingPhysics(self)` -- docstring: present
- `runAllTests(self)` -- docstring: present

#### class surfGAt3D

Signature: `surfGAt3D(H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False)`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False)` -- docstring: present
- `updateH(self, fermi=None)` -- docstring: present
- `sigmaK(self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `sigmaSurf(self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `crossTermQSurf(self, E, sigInds=None, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `crossTermQBulk(self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `setF(self, F, mu1, mu2)` -- docstring: present
- `sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `DOS(self, E)` -- docstring: present
- `calcFermi(self, ne, tol=FERMI_CALCULATION_TOL)` -- docstring: present

---

### Module: gauNEGF.surfGBethe

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 2
- **Public functions:** 0

#### class surfGB

Signature: `surfGB(F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE)`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE)` -- docstring: present
- `genNeighbors(self, plane_normal, first_neighbor)` -- docstring: present
- `readBetheParams(self, filename)` -- docstring: present
- `constructMat(self, Mdict, dirCosines, SOC=False)` -- docstring: present
- `sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `getSigma(self, Elist=[None, None], conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `updateFermi(self, i, Ef)` -- docstring: present
- `setF(self, F, muL, muR)` -- docstring: present
- `testDOrbitalFunctions(self)` -- docstring: present
- `testDOrbitalSymmetry(self)` -- docstring: present
- `testPDInteraction(self)` -- docstring: present
- `testDDInteraction(self)` -- docstring: present
- `testHoppingPhysics(self)` -- docstring: present
- `runAllTests(self)` -- docstring: present

#### class surfGBAt

Signature: `surfGBAt(H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False)`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False)` -- docstring: present
- `updateH(self, fermi=None)` -- docstring: present
- `sigmaK(self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `sigmaSurf(self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `crossTermQSurf(self, E, sigInds=None, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `crossTermQBulk(self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5)` -- docstring: present
- `setF(self, F, mu1, mu2)` -- docstring: present
- `sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `DOS(self, E)` -- docstring: present
- `calcFermi(self, ne, tol=FERMI_CALCULATION_TOL)` -- docstring: present

---

### Module: gauNEGF.surfGTester

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 1
- **Public functions:** 0

#### class surfGTest

Signature: `surfGTest(Fock, Overlap, indsList, sig1=None, sig2=None, spin='r')`
Docstring: present
Deprecated: false
Decorators: []

Methods:
- `__init__(self, Fock, Overlap, indsList, sig1=None, sig2=None, spin='r')` -- docstring: present
- `sigma(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `sigmaTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `setF(self, F, mu1, mu2)` -- docstring: present
- `crossTermQ(self, E, i, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present
- `crossTermQTot(self, E, conv=SURFACE_GREEN_CONVERGENCE)` -- docstring: present

---

### Module: gauNEGF.matTools

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 0
- **Public functions:** 5

#### function formSigma

Signature: `formSigma(inds, V, nsto, S=0)`
Docstring: present
Deprecated: false
Decorators: []

#### function getDen

Signature: `getDen(bar, spin)`
Docstring: present
Deprecated: false
Decorators: []

#### function getFock

Signature: `getFock(bar, spin)`
Docstring: present
Deprecated: false
Decorators: []

#### function getEnergies

Signature: `getEnergies(bar, spin)`
Docstring: present
Deprecated: false
Decorators: []

#### function storeDen

Signature: `storeDen(bar, P, spin)`
Docstring: present
Deprecated: false
Decorators: []

---

### Module: gauNEGF.integrate

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 0
- **Public functions:** 4

#### function GrInt

Signature: `GrInt(F, S, g, Elist, weights)`
Docstring: present
Deprecated: false
Decorators: []

#### function GrIntCross

Signature: `GrIntCross(F, S, g, Elist, weights)`
Docstring: present
Deprecated: false
Decorators: []

#### function GrLessInt

Signature: `GrLessInt(F, S, g, Elist, weights, ind=None)`
Docstring: present
Deprecated: false
Decorators: []

---

### Module: gauNEGF.spinTools

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 0
- **Public functions:** 7

#### function get_hirshfeld_data

Signature: `get_hirshfeld_data(filename)`
Docstring: present
Deprecated: false
Decorators: []

#### function spinorTF

Signature: `spinorTF(rho_initial, rho_target)`
Docstring: present
Deprecated: false
Decorators: []

#### function genRot

Signature: `genRot(n, omega)`
Docstring: present
Deprecated: false
Decorators: []

#### function genRotsGrid

Signature: `genRotsGrid(spinVec, dphi=np.pi/4)`
Docstring: present
Deprecated: false
Decorators: []

#### function genOrthRots

Signature: `genOrthRots(spinVec)`
Docstring: present
Deprecated: false
Decorators: []

#### function genOrthRotFile

Signature: `genOrthRotFile(filename)`
Docstring: present
Deprecated: false
Decorators: []

#### function genOrthRotGrids

Signature: `genOrthRotGrids(spinVec=None, filename=None, dphi=np.pi/4)`
Docstring: present
Deprecated: false
Decorators: []

#### function constructSOCterm

Signature: `constructSOCterm(lambdas)`
Docstring: present
Deprecated: false
Decorators: []

---

### Module: gauNEGF.utils

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 0
- **Public functions:** 4

#### function fractional_matrix_power

Signature: `fractional_matrix_power(S, power)`
Docstring: present
Deprecated: false
Decorators: [@jit]

#### function inv

Signature: `inv(A)`
Docstring: present
Deprecated: false
Decorators: [@jit]

#### function eig

Signature: `eig(A)`
Docstring: present
Deprecated: false
Decorators: [@jit]

#### function eigh

Signature: `eigh(A)`
Docstring: present
Deprecated: false
Decorators: [@jit]

---

### Module: gauNEGF.protocols

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 1
- **Public functions:** 0

#### class SurfGProtocol

Signature: `SurfGProtocol(Protocol)`
Docstring: present
Deprecated: false
Decorators: [@runtime_checkable]

Methods:
- `sigma(self, E: complex, i: int, conv: float = ...) -> np.ndarray` -- docstring: present
- `sigmaTot(self, E: complex, conv: float = ...) -> np.ndarray` -- docstring: present
- `setF(self, F: np.ndarray, mu1: float, mu2: float) -> None` -- docstring: present
- `crossTermQ(self, E: complex, i: int, conv: float = ...) -> Optional[np.ndarray]` -- docstring: present
- `crossTermQTot(self, E: complex, conv: float = ...) -> Optional[np.ndarray]` -- docstring: present

---

### Module: gauNEGF.config

- **Module docstring:** present
- **Module-level deprecation marker:** none
- **Public classes:** 0
- **Public functions:** 1

#### function shard_array

Signature: `shard_array(array, axis=0)`
Docstring: present
Deprecated: false
Decorators: []

---

### Module: gauNEGF.fermiSearch

- **Module docstring:** present
- **Module-level deprecation marker:** [DEPRECATED]
- **Public classes:** 1
- **Public functions:** 0

#### class DOSFermiSearch

Signature: `DOSFermiSearch(initialEf, nTarget, deltaE=0.01, numPoints=5, debug=False)`
Docstring: present
Deprecated: true
Decorators: []

Methods:
- `__init__(self, initialEf, nTarget, deltaE=0.01, numPoints=5, debug=False)` -- docstring: present
- `getAccuracy(self)` -- docstring: present
- `matrixFiniteDifference(self, dosFunc, E, h, numPoints)` -- docstring: present
- `step(self, dosFunc, nCurr, stepLim=10)` -- docstring: present

---

## Flat Lookup Table

| Symbol | Module | Type | Signature | Docstring | Deprecated |
|--------|--------|------|-----------|-----------|-----------|
| NEGF | gauNEGF.scf | class | (fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE) | present | false |
| NEGF.__init__ | gauNEGF.scf | method | (self, fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE) | present | false |
| NEGF.runDFT | gauNEGF.scf | method | (self, fullSCF=True) | present | false |
| NEGF.updateN | gauNEGF.scf | method | (self) | present | false |
| NEGF.setFock | gauNEGF.scf | method | (self, F_) | present | false |
| NEGF.setDen | gauNEGF.scf | method | (self, P_, enableSpinLock=False, spinLockList=None) | present | false |
| NEGF.getHOMOLUMO | gauNEGF.scf | method | (self) | present | false |
| NEGF.setVoltage | gauNEGF.scf | method | (self, qV, fermi=None, Emin=None, Eminf=None) | present | false |
| NEGF.setContacts | gauNEGF.scf | method | (self, lContact=None, rContact=None) | present | false |
| NEGF.setSigma | gauNEGF.scf | method | (self, lContact=None, rContact=None, sig=-0.1j, sig2=None) | present | false |
| NEGF.getSigma | gauNEGF.scf | method | (self, E=0) | present | false |
| NEGF.FockToP | gauNEGF.scf | method | (self) | present | false |
| NEGF.PMix | gauNEGF.scf | method | (self, damping, Pulay=False) | present | false |
| NEGF.PToFock | gauNEGF.scf | method | (self) | present | false |
| NEGF.SCF | gauNEGF.scf | method | (self, conv=SCF_CONVERGENCE_TOL, damping=SCF_DAMPING, maxcycles=SCF_MAX_CYCLES, checkpoint=True, pulay=True) | present | false |
| NEGF.writeChk | gauNEGF.scf | method | (self) | present | false |
| NEGF.saveMAT | gauNEGF.scf | method | (self, matfile="out.mat") | present | false |
| NEGFE | gauNEGF.scfE | class | (NEGF) | present | false |
| NEGFE.setContactBethe | gauNEGF.scfE | method | (self, contactList, latFile='Au', eta=ETA, T=TEMPERATURE) | present | false |
| NEGFE.setContact1D | gauNEGF.scfE | method | (self, contactList, tauList=None, stauList=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, neList=None, muList=None, eta=ETA, T=TEMPERATURE, symmetrize_contacts=None) | present | false |
| NEGFE._symmetrize_F | gauNEGF.scfE | method | (self) | present | false |
| NEGFE.setSigma | gauNEGF.scfE | method | (self, lContact=None, rContact=None, sig=-0.1j, sig2=None, T=TEMPERATURE) | present | false |
| NEGFE.setVoltage | gauNEGF.scfE | method | (self, qV, fermi=None, Emin=None, Eminf=None, fermiMethod=None) | present | false |
| NEGFE.setIntegralLimits | gauNEGF.scfE | method | (self, N1=None, N2=None, Nnegf=None, tol=ADAPTIVE_INTEGRATION_TOL, Emin=None) | present | false |
| NEGFE.integralCheck | gauNEGF.scfE | method | (self, cycles=10, damp=0.02, pauseFermi=False) | present | false |
| NEGFE.getSigma | gauNEGF.scfE | method | (self, E, E2=None) | present | false |
| NEGFE.spawnNEGF | gauNEGF.scfE | method | (self, mu1=None, mu2=None) | present | false |
| NEGFE.FockToP | gauNEGF.scfE | method | (self) | present | false |
| NEGFE.PToFock | gauNEGF.scfE | method | (self) | present | false |
| fermi | gauNEGF.density | function | (E, mu, T) | present | false |
| getANTPoints | gauNEGF.density | function | (N) | present | false |
| integratePoints | gauNEGF.density | function | (computePointFunc, numPoints, parallel=False, numWorkers=None, chunkSize=None, debug=False) | present | false |
| integratePointsAdaptiveANT | gauNEGF.density | function | (computePoint, tol=ADAPTIVE_INTEGRATION_TOL, maxN=MAX_GRID_POINTS, debug=False) | present | false |
| density | gauNEGF.density | function | (V, Vc, D, Gam, Emin, mu) | present | false |
| bisectFermi | gauNEGF.density | function | (V, Vc, D, Gam, Nexp, conv=FERMI_CALCULATION_TOL, Eminf=ENERGY_MIN) | present | false |
| densityRealN | gauNEGF.density | function | (F, S, g, Emin, mu, N=100, T=TEMPERATURE, showText=True) | present | false |
| densityReal | gauNEGF.density | function | (F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False) | present | false |
| densityGridN | gauNEGF.density | function | (F, S, g, mu1, mu2, ind=None, N=100, T=TEMPERATURE, showText=True) | present | false |
| densityGridTrap | gauNEGF.density | function | (F, S, g, mu1, mu2, ind=None, N=100, T=TEMPERATURE) | present | false |
| densityGrid | gauNEGF.density | function | (F, S, g, mu1, mu2, ind=None, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False) | present | false |
| densityComplexN | gauNEGF.density | function | (F, S, g, Emin, mu, N=100, T=TEMPERATURE, showText=True, method='ant') | present | false |
| densityComplex | gauNEGF.density | function | (F, S, g, Emin, mu, tol=ADAPTIVE_INTEGRATION_TOL, T=TEMPERATURE, debug=False) | present | false |
| calcEmin | gauNEGF.density | function | (F, S, g, tol=FERMI_CALCULATION_TOL, maxN=MAX_CYCLES, Emin=None) | absent | false |
| calcTSW | gauNEGF.density | function | (F, S, g, tol=FERMI_CALCULATION_TOL, maxN=FERMI_SEARCH_CYCLES, Eminf=None, TSW=None) | present | false |
| integralFit | gauNEGF.density | function | (F, S, g, mu, Eminf=ENERGY_MIN, tol=FERMI_CALCULATION_TOL, T=TEMPERATURE, maxN=MAX_CYCLES) | present | false |
| integralFitNEGF | gauNEGF.density | function | (F, S, g, fermi, qV, Eminf=ENERGY_MIN, tol=FERMI_CALCULATION_TOL, T=TEMPERATURE, maxGrid=MAX_GRID_POINTS) | present | false |
| getFermiContact | gauNEGF.density | function | (g, ne, Emin=None, lBound=None, uBound=None, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE) | present | false |
| calcFermi | gauNEGF.density | function | (g, ne, Emin, Ef, lBound=None, uBound=None, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE) | present | false |
| calcFermiBisect | gauNEGF.density | function | (g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE, uBound=None, lBound=None) | present | false |
| calcFermiSecant | gauNEGF.density | function | (g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE) | present | false |
| calcFermiMuller | gauNEGF.density | function | (g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE) | present | false |
| calcFermiPolyFit | gauNEGF.density | function | (g, ne, Emin, Ef, N, tol=ADAPTIVE_INTEGRATION_TOL, conv=FERMI_CALCULATION_TOL, maxcycles=FERMI_SEARCH_CYCLES, T=TEMPERATURE, order=3) | present | false |
| SigmaCalculator | gauNEGF.transport | class | (sig1, sig2=None, energy_dependent=None) | present | false |
| SigmaCalculator.__init__ | gauNEGF.transport | method | (self, sig1, sig2=None, energy_dependent=None) | present | false |
| SigmaCalculator.get_sigma_total | gauNEGF.transport | method | (self, E, spin=None, matrix_size=None) | present | false |
| SigmaCalculator.get_Q_tot | gauNEGF.transport | method | (self, E, spin=None, matrix_size=None) | present | false |
| SigmaCalculator.get_sigma | gauNEGF.transport | method | (self, E, contact_index, spin=None, matrix_size=None) | present | false |
| SigmaCalculator.get_gamma | gauNEGF.transport | method | (self, E, contact_index, spin=None, matrix_size=None) | present | false |
| transmission_single_energy | gauNEGF.transport | function | (E, F_jax, S_jax, sigma_calc, spin=None) | present | false |
| dos_single_energy | gauNEGF.transport | function | (E, F_jax, S_jax, sigma_calc, spin=None) | present | false |
| calculate_transmission | gauNEGF.transport | function | (F, S, sigma_calculator, energy_list, spin=None, checkpoint_file=None, checkpoint_interval=10) | present | false |
| calculate_dos | gauNEGF.transport | function | (F, S, sigma_calculator, energy_list, spin=None, checkpoint_file=None, checkpoint_interval=10) | present | false |
| calculate_current | gauNEGF.transport | function | (F, S, sigma_calculator, fermi, qV, T=TEMPERATURE, spin=None, dE=ENERGY_STEP, **kwargs) | present | false |
| current | gauNEGF.transport | function | (F, S, sig1, sig2, fermi, qV, T=TEMPERATURE, spin="r", dE=ENERGY_STEP) | present | false |
| currentSpin | gauNEGF.transport | function | (F, S, sig1, sig2, fermi, qV, T=TEMPERATURE, spin="r", dE=ENERGY_STEP) | present | false |
| currentE | gauNEGF.transport | function | (F, S, g, fermi, qV, T=TEMPERATURE, spin="r", dE=ENERGY_STEP) | present | false |
| currentF | gauNEGF.transport | function | (fn, dE=ENERGY_STEP, T=TEMPERATURE) | present | false |
| cohTrans | gauNEGF.transport | function | (Elist, F, S, sig1, sig2) | present | false |
| cohTransSpin | gauNEGF.transport | function | (Elist, F, S, sig1, sig2, spin='u') | present | false |
| DOS | gauNEGF.transport | function | (Elist, F, S, sig1, sig2) | present | false |
| cohTransE | gauNEGF.transport | function | (Elist, F, S, g) | present | false |
| cohTransSpinE | gauNEGF.transport | function | (Elist, F, S, g, spin='u') | present | false |
| DOSE | gauNEGF.transport | function | (Elist, F, S, g) | present | false |
| surfG | gauNEGF.surfG1D | class | (Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA, spin='r') | present | false |
| surfG.__init__ | gauNEGF.surfG1D | method | (self, Fock, Overlap, indsList, taus=None, staus=None, alphas=None, aOverlaps=None, betas=None, bOverlaps=None, eta=ETA, spin='r') | present | false |
| surfG._setContacts | gauNEGF.surfG1D | method | (self, alphas=None, aOverlaps=None, betas=None, bOverlaps=None) | present | false |
| surfG._regularizeContacts | gauNEGF.surfG1D | method | (self) | present | false |
| surfG._rejit | gauNEGF.surfG1D | method | (self) | present | false |
| surfG.g | gauNEGF.surfG1D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE, relFactor=0.5) | present | false |
| surfG.setF | gauNEGF.surfG1D | method | (self, F, mu1=None, mu2=None) | present | false |
| surfG.sigma | gauNEGF.surfG1D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG.crossTermQ | gauNEGF.surfG1D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG.crossTermQTot | gauNEGF.surfG1D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG.sigmaTot | gauNEGF.surfG1D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG3 | gauNEGF.surfG3D | class | (F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE) | present | false |
| surfG3.__init__ | gauNEGF.surfG3D | method | (self, F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE) | present | false |
| surfG3.genNeighbors | gauNEGF.surfG3D | method | (self, plane_normal, first_neighbor) | present | false |
| surfG3.readBetheParams | gauNEGF.surfG3D | method | (self, filename) | present | false |
| surfG3.constructMat | gauNEGF.surfG3D | method | (self, Mdict, dirCosines, SOC=False) | present | false |
| surfG3.sigma | gauNEGF.surfG3D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG3.sigmaTot | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG3.crossTermQ | gauNEGF.surfG3D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG3.crossTermQTot | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG3.getSigma | gauNEGF.surfG3D | method | (self, Elist=[None, None], conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfG3.updateFermi | gauNEGF.surfG3D | method | (self, i, Ef) | present | false |
| surfG3.setF | gauNEGF.surfG3D | method | (self, F, muL, muR) | present | false |
| surfG3.testDOrbitalFunctions | gauNEGF.surfG3D | method | (self) | present | false |
| surfG3.testDOrbitalSymmetry | gauNEGF.surfG3D | method | (self) | present | false |
| surfG3.testPDInteraction | gauNEGF.surfG3D | method | (self) | present | false |
| surfG3.testDDInteraction | gauNEGF.surfG3D | method | (self) | present | false |
| surfG3.testHoppingPhysics | gauNEGF.surfG3D | method | (self) | present | false |
| surfG3.runAllTests | gauNEGF.surfG3D | method | (self) | present | false |
| surfGAt3D | gauNEGF.surfG3D | class | (H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False) | present | false |
| surfGAt3D.__init__ | gauNEGF.surfG3D | method | (self, H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False) | present | false |
| surfGAt3D.updateH | gauNEGF.surfG3D | method | (self, fermi=None) | present | false |
| surfGAt3D.sigmaK | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGAt3D.sigmaSurf | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGAt3D.crossTermQSurf | gauNEGF.surfG3D | method | (self, E, sigInds=None, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGAt3D.crossTermQBulk | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGAt3D.setF | gauNEGF.surfG3D | method | (self, F, mu1, mu2) | present | false |
| surfGAt3D.sigmaTot | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGAt3D.sigma | gauNEGF.surfG3D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGAt3D.crossTermQ | gauNEGF.surfG3D | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGAt3D.crossTermQTot | gauNEGF.surfG3D | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGAt3D.DOS | gauNEGF.surfG3D | method | (self, E) | present | false |
| surfGAt3D.calcFermi | gauNEGF.surfG3D | method | (self, ne, tol=FERMI_CALCULATION_TOL) | present | false |
| surfGB | gauNEGF.surfGBethe | class | (F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE) | present | false |
| surfGB.__init__ | gauNEGF.surfGBethe | method | (self, F, S, contacts, bar, latFile='Au', spin='r', eta=ETA, T=TEMPERATURE) | present | false |
| surfGB.genNeighbors | gauNEGF.surfGBethe | method | (self, plane_normal, first_neighbor) | present | false |
| surfGB.readBetheParams | gauNEGF.surfGBethe | method | (self, filename) | present | false |
| surfGB.constructMat | gauNEGF.surfGBethe | method | (self, Mdict, dirCosines, SOC=False) | present | false |
| surfGB.sigma | gauNEGF.surfGBethe | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGB.sigmaTot | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGB.crossTermQ | gauNEGF.surfGBethe | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGB.crossTermQTot | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGB.getSigma | gauNEGF.surfGBethe | method | (self, Elist=[None, None], conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGB.updateFermi | gauNEGF.surfGBethe | method | (self, i, Ef) | present | false |
| surfGB.setF | gauNEGF.surfGBethe | method | (self, F, muL, muR) | present | false |
| surfGB.testDOrbitalFunctions | gauNEGF.surfGBethe | method | (self) | present | false |
| surfGB.testDOrbitalSymmetry | gauNEGF.surfGBethe | method | (self) | present | false |
| surfGB.testPDInteraction | gauNEGF.surfGBethe | method | (self) | present | false |
| surfGB.testDDInteraction | gauNEGF.surfGBethe | method | (self) | present | false |
| surfGB.testHoppingPhysics | gauNEGF.surfGBethe | method | (self) | present | false |
| surfGB.runAllTests | gauNEGF.surfGBethe | method | (self) | present | false |
| surfGBAt | gauNEGF.surfGBethe | class | (H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False) | present | false |
| surfGBAt.__init__ | gauNEGF.surfGBethe | method | (self, H, Slist, Vlist, eta, T=TEMPERATURE, SOC=False) | present | false |
| surfGBAt.updateH | gauNEGF.surfGBethe | method | (self, fermi=None) | present | false |
| surfGBAt.sigmaK | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGBAt.sigmaSurf | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGBAt.crossTermQSurf | gauNEGF.surfGBethe | method | (self, E, sigInds=None, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGBAt.crossTermQBulk | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE, mix=0.5) | present | false |
| surfGBAt.setF | gauNEGF.surfGBethe | method | (self, F, mu1, mu2) | present | false |
| surfGBAt.sigmaTot | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGBAt.sigma | gauNEGF.surfGBethe | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGBAt.crossTermQ | gauNEGF.surfGBethe | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGBAt.crossTermQTot | gauNEGF.surfGBethe | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGBAt.DOS | gauNEGF.surfGBethe | method | (self, E) | present | false |
| surfGBAt.calcFermi | gauNEGF.surfGBethe | method | (self, ne, tol=FERMI_CALCULATION_TOL) | present | false |
| surfGTest | gauNEGF.surfGTester | class | (Fock, Overlap, indsList, sig1=None, sig2=None, spin='r') | present | false |
| surfGTest.__init__ | gauNEGF.surfGTester | method | (self, Fock, Overlap, indsList, sig1=None, sig2=None, spin='r') | present | false |
| surfGTest.sigma | gauNEGF.surfGTester | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGTest.sigmaTot | gauNEGF.surfGTester | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGTest.setF | gauNEGF.surfGTester | method | (self, F, mu1, mu2) | present | false |
| surfGTest.crossTermQ | gauNEGF.surfGTester | method | (self, E, i, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| surfGTest.crossTermQTot | gauNEGF.surfGTester | method | (self, E, conv=SURFACE_GREEN_CONVERGENCE) | present | false |
| formSigma | gauNEGF.matTools | function | (inds, V, nsto, S=0) | present | false |
| getDen | gauNEGF.matTools | function | (bar, spin) | present | false |
| getFock | gauNEGF.matTools | function | (bar, spin) | present | false |
| getEnergies | gauNEGF.matTools | function | (bar, spin) | present | false |
| storeDen | gauNEGF.matTools | function | (bar, P, spin) | present | false |
| GrInt | gauNEGF.integrate | function | (F, S, g, Elist, weights) | present | false |
| GrIntCross | gauNEGF.integrate | function | (F, S, g, Elist, weights) | present | false |
| GrLessInt | gauNEGF.integrate | function | (F, S, g, Elist, weights, ind=None) | present | false |
| get_hirshfeld_data | gauNEGF.spinTools | function | (filename) | present | false |
| spinorTF | gauNEGF.spinTools | function | (rho_initial, rho_target) | present | false |
| genRot | gauNEGF.spinTools | function | (n, omega) | present | false |
| genRotsGrid | gauNEGF.spinTools | function | (spinVec, dphi=np.pi/4) | present | false |
| genOrthRots | gauNEGF.spinTools | function | (spinVec) | present | false |
| genOrthRotFile | gauNEGF.spinTools | function | (filename) | present | false |
| genOrthRotGrids | gauNEGF.spinTools | function | (spinVec=None, filename=None, dphi=np.pi/4) | present | false |
| constructSOCterm | gauNEGF.spinTools | function | (lambdas) | present | false |
| fractional_matrix_power | gauNEGF.utils | function | (S, power) | present | false |
| inv | gauNEGF.utils | function | (A) | present | false |
| eig | gauNEGF.utils | function | (A) | present | false |
| eigh | gauNEGF.utils | function | (A) | present | false |
| SurfGProtocol | gauNEGF.protocols | class | (Protocol) | present | false |
| SurfGProtocol.sigma | gauNEGF.protocols | method | (self, E: complex, i: int, conv: float = ...) -> np.ndarray | present | false |
| SurfGProtocol.sigmaTot | gauNEGF.protocols | method | (self, E: complex, conv: float = ...) -> np.ndarray | present | false |
| SurfGProtocol.setF | gauNEGF.protocols | method | (self, F: np.ndarray, mu1: float, mu2: float) -> None | present | false |
| SurfGProtocol.crossTermQ | gauNEGF.protocols | method | (self, E: complex, i: int, conv: float = ...) -> Optional[np.ndarray] | present | false |
| SurfGProtocol.crossTermQTot | gauNEGF.protocols | method | (self, E: complex, conv: float = ...) -> Optional[np.ndarray] | present | false |
| shard_array | gauNEGF.config | function | (array, axis=0) | present | false |
| DOSFermiSearch | gauNEGF.fermiSearch | class | (initialEf, nTarget, deltaE=0.01, numPoints=5, debug=False) | present | true |
| DOSFermiSearch.__init__ | gauNEGF.fermiSearch | method | (self, initialEf, nTarget, deltaE=0.01, numPoints=5, debug=False) | present | true |
| DOSFermiSearch.getAccuracy | gauNEGF.fermiSearch | method | (self) | present | true |
| DOSFermiSearch.matrixFiniteDifference | gauNEGF.fermiSearch | method | (self, dosFunc, E, h, numPoints) | present | true |
| DOSFermiSearch.step | gauNEGF.fermiSearch | method | (self, dosFunc, nCurr, stepLim=10) | present | true |

---

## Summary

- **Total modules:** 15
- **Total classes:** 13 (1 deprecated) -- surfGBethe has 2 classes (surfGB, surfGBAt); total corrected from Phase I
- **Total functions:** 67 (0 deprecated)
- **Total deprecated symbols:** 5 (DOSFermiSearch and its 4 methods)
- **Total absent docstrings:** 0 (Phase II added docstrings for calcEmin, inv, eig, eigh, surfG1D module-level)

Modules: scf, scfE, density, transport, surfG1D, surfG3D, surfGBethe, surfGTester, matTools,
integrate, spinTools, utils, protocols, config, fermiSearch. All modules have module-level
docstrings (Phase II added surfG1D module docstring). Deprecated symbols are all in
fermiSearch. Phase II filled all 5 docstring gaps (calcEmin, inv, eig, eigh, surfG1D module).

---

## Upgrade Conventions Applied

Generated: 2026-05-11
Upgrade pass applied to all 15 modules per Phase II Task 12.

### Inheritance constructors detected: 1

- gauNEGF.scfE: NEGFE inherits NEGF.__init__ (no explicit __init__ in class body)
  Inlined signature: NEGFE(fn, basis="chkbasis", func="hf", spin="r", fullSCF=True, route=None, section=None, nPulay=PULAY_MIXING_SIZE)

### Dynamic attribute sets detected: 2

- gauNEGF.scf NEGF: self.sigma1, self.sigma2, self.sigma12, self.Gam1, self.Gam2 (set by setSigma)
- gauNEGF.scfE NEGFE: self.g (set by setContactBethe, setContact1D, setSigma)

### Variable-shape returns detected: 2

- gauNEGF.transport calculate_transmission: spin_trans None -> single array; not None -> (transmission, spin_trans)
- gauNEGF.transport calculate_dos: dos_spin None -> (dos_total, dos_per_site); not None -> (dos_total, dos_per_site, dos_spin)

### Duck-typed overloads detected: 1

- gauNEGF.transport SigmaCalculator.__init__: sig1 accepts ndarray (static) or surfG-type object (duck-typed via hasattr('sigma', 'sigmaTot'))

### Nested-list parameter conventions detected: 1

- gauNEGF.scfE NEGFE.setContactBethe: contactList is [[atom1, atom2, ...], [atom3, ...]] (one inner list per contact)
