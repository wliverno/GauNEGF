# GaussianNEGF

A Python package for performing Non-Equilibrium Green's Function (NEGF) calculations integrated with Gaussian quantum chemistry software.

## Overview

GaussianNEGF is a powerful tool that combines density functional theory (DFT) calculations from Gaussian with NEGF formalism to simulate quantum transport in molecular devices. The package enables calculations of electronic properties and transport characteristics of molecular junctions under bias voltage.

This package builds upon the foundation laid by the open-source ANT.Gaussian package. Specifically, the Bethe lattice implementation is adapted from ANT.Gaussian's Fortran implementation, and the numerical quadrature methods for integration are also derived from ANT.Gaussian. We gratefully acknowledge the contributions of the ANT.Gaussian developers to the field of molecular electronics.

## Technical Implementation Details

### Density Matrix Calculations (Confidence: High)
- Multiple integration methods supported:
  - Real axis integration with Legendre quadrature
  - Complex contour integration with ANT-style quadrature points
  - Adaptive integration grid fitting
- Parallel processing capabilities for large systems
- Temperature effects included via Fermi-Dirac statistics
- Robust convergence handling with multiple Fermi search algorithms

### SCF Implementation (Confidence: High)
- Base SCF implementation in `scf.py`
- Extended energy-dependent implementation in `scfE.py`
- Features:
  - Pulay mixing for convergence acceleration
  - Multiple Fermi energy search methods (Müller, secant, prediction)
  - Automated integration parameter optimization
  - Robust error handling and convergence checks

### Surface Green's Functions (Confidence: High)
- Bethe lattice implementation for metal contacts
  - Direct port of ANT.Gaussian's proven implementation
  - Supports FCC [111] surface geometry
  - Complete Slater-Koster parameterization
- 1D chain contacts supported
- Test contacts for validation

### Transport Calculations (Confidence: Medium-High)
- Transmission function calculation
- Current integration under bias
- Temperature broadening effects
- Adaptive energy grid selection

### Integration with Gaussian (Confidence: High)
- Robust interface through gau-open
- Handles all matrix operations:
  - Fock matrix extraction and modification
  - Density matrix updates
  - Overlap matrix handling
- Supports multiple basis sets and DFT functionals

## Core Modules

### surfGBethe.py (Confidence: High)
- Implements Bethe lattice surface Green's function calculations (adapted from ANT.Gaussian's Fortran implementation)
- Handles contact-device interactions
- Supports FCC [111] surface geometry only
- Uses Slater-Koster parameterization for s, p, and d orbitals (imported from a custom file)
- Includes temperature and broadening effects

### scfE.py (Confidence: High)
- Extension of scf.py for energy-dependent contacts 
- Includes temperature effects and broadening
- Uses multiple methods for Fermi energy calculation
- Automated integration grid fitting
- Robust convergence handling with multiple algorithms

### density.py (Confidence: High)
- Comprehensive density matrix calculations
- Multiple integration schemes:
  - Real axis (Legendre quadrature)
  - Complex contour (ANT-style points)
  - Adaptive grid selection
- Parallel processing support
- Temperature effects via Fermi-Dirac statistics

### transport.py (Confidence: Medium-High)
- Transmission calculations
- Current integration
- DOS calculations
- Bias voltage handling

### matTools.py (Confidence: High)
- Matrix operation utilities
- Basis set transformations
- Helper functions for Green's function calculations

## Integration Methods

### Quadrature Implementations (Confidence: High)
1. Real Axis Integration
   - Legendre quadrature for equilibrium calculations
   - Adaptive grid selection based on DOS
   - Temperature broadening support

2. Complex Contour Integration (Confidence: High)
   - ANT-style quadrature points
   - Optimized for equilibrium calculations
   - Efficient pole handling

3. Non-equilibrium Integration (Confidence: Medium-High)
   - Real-axis grid methods
   - Adaptive grid selection
   - Bias window handling

## Performance Optimizations

1. Integration Strategies (Confidence: High)
   - Automated grid point selection
   - Adaptive error control
   - Parallel processing support

2. SCF Convergence (Confidence: High)
   - Multiple mixing schemes
   - Automated parameter selection
   - Robust error handling

3. Memory Management (Confidence: Medium)
   - Efficient matrix operations
   - Parallel processing options
   - Resource monitoring

## Installation

1. Ensure you have Gaussian installed and properly configured
2. Install required Python dependencies:
   ```bash
   numpy
   scipy
   matplotlib
   ```
3. Install the gau-open package for Gaussian integration

## Usage

### Basic NEGF Calculation

```python
from scf import NEGF

# Initialize NEGF calculation
negf = NEGF("molecule", basis="lanl2dz", func="b3pw91")

# Set contacts
negf.setContacts(lContact=[1], rContact=[2])

# Set voltage bias
negf.setVoltage(qV=1.0)  # 1.0V bias

# Run SCF calculation
negf.SCF(conv=1e-5, maxcycles=100)
```

### Surface Green's Function Setup

```python
from surfGBethe import surfGB

# Initialize surface Green's function
surf = surfGB(F, S, contacts, bar, latFile='Au')

# Calculate self-energy
sigma = surf.sigma(E)
```

## Key Parameters

### NEGF Class
- `basis`: Gaussian basis set (default: "lanl2dz")
- `func`: DFT functional (default: "b3pw91")
- `spin`: Spin configuration ("r" for restricted, "u" for unrestricted)
- `fullSCF`: Whether to run full SCF or use Harris approximation

### surfGB Class
- `latFile`: Metal contact parameter file (e.g., 'Au.bethe')
- `spin`: Spin configuration
- `eta`: Broadening parameter
- `T`: Temperature in Kelvin

## Integration with Gaussian

The package uses the gau-open interface to interact with Gaussian:

1. Reads Gaussian input files (.gjf)
2. Manages checkpoint files (.chk)
3. Extracts Fock and overlap matrices
4. Updates density matrices
5. Handles basis set information

## File Formats

### Bethe Lattice Parameters (.bethe)
Contains Slater-Koster parameters for metal contacts:
- Onsite energies (es, ep, edd, edt)
- Hopping integrals (sss, sps, pps, etc.)
- Overlap parameters (Ssss, Ssps, etc.)

### Gaussian Files
- `.gjf`: Input geometry and calculation parameters
- `.chk`: Checkpoint file for matrices and wavefunctions
- `.log`: Output file with calculation results

## Best Practices

1. Always check SCF convergence carefully
2. Use appropriate basis sets for both molecule and contacts
3. Ensure proper contact atom selection
4. Monitor electron count throughout calculations
5. Verify energy conservation in transport calculations

## Error Handling

The package includes various error checks:
- Matrix dimension consistency
- Electron count verification
- Convergence monitoring
- Contact setup validation

## Performance Tips

1. Use Pulay mixing for faster convergence
2. Optimize integration parameters for accuracy vs. speed
3. Choose appropriate convergence criteria
4. Monitor memory usage for large systems

## Known Limitations

1. Currently supports up to two terminals
2. Limited to specific metal surface orientations
3. Requires Gaussian installation
4. Memory intensive for large systems

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

See LICENSE file for details.

## Contact

For support and questions, please open an issue in the repository.

## References

1. ANT.Gaussian package: The original implementation of the Bethe lattice approach and numerical integration methods used in this package.
   - Original Fortran implementation of surface Green's functions
   - Numerical quadrature methods for transport calculations
   - Website: https://github.com/juanjosepalacios/ANT.Gaussian