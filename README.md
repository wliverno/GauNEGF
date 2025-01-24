# GaussianNEGF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for performing Non-Equilibrium Green's Function (NEGF) calculations integrated with Gaussian quantum chemistry software.

## Features

- Integration with Gaussian quantum chemistry package
- Self-consistent field (SCF) calculations with NEGF formalism
- Surface Green's function calculations using Bethe lattice approach
- Transport calculations for molecular junctions
- Support for spin-polarized calculations
- Parallel processing capabilities

## Requirements

### Required Software
- Python 3.7 or higher
- Gaussian quantum chemistry package (licensed copy required)
- gau-open Python interface for Gaussian

### Python Dependencies
- numpy
- scipy
- matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GaussianNEGF.git
cd GaussianNEGF

# Install in development mode
pip install -e .
```

## Quick Start

```python
from gauNEGF import scf, transport

# Initialize NEGF calculation
negf = scf.NEGF("molecule", basis="lanl2dz", func="b3pw91")

# Set contacts - left contact on atom 1, right contact on atom 2
negf.setContacts([1], [2])

# Set voltage bias
negf.setVoltage(0.1)  # 0.1V bias

# Run SCF calculation
negf.SCF(1e-3, 0.02) # convergence @ 1e-3, damping=0.02

# Calculate current
I = transport.quickCurrent(negf.F, negf.S, negf.sig1, negf.sig2, negf.fermi, negf.qV)

```

## Documentation

See the `DOCUMENTATION.md` file for detailed usage instructions and API documentation.

For usage examples, see the files in the `examples/` directory.

## Citation

If you use GaussianNEGF in your research, please cite:

```bibtex
@software{gaussianNEGF2024,
  author       = {Your Name},
  title        = {GaussianNEGF: A Python package for Non-Equilibrium Green's Function calculations},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## Acknowledgments

This package builds upon the foundation laid by the open-source ANT.Gaussian package. The Bethe lattice implementation and numerical quadrature methods are adapted from ANT.Gaussian's Fortran implementation.

Note: This package requires a licensed copy of Gaussian quantum chemistry software to run. Gaussian is a registered trademark of Gaussian, Inc.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 