# GauNEGF


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15178667.svg)](https://doi.org/10.5281/zenodo.15178667)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

***Note: This package is currently in testing and uses the Gaussian Development Version and Gaussian Python Interface, both of which are not publicly available. Please check the [Gaussian website](https://gaussian.com/news/) for updates on the forthcoming releases.***

A Python package for performing Non-Equilibrium Green's Function (NEGF) calculations integrated with Gaussian quantum chemistry software.

Documentation: [https://wliverno.github.io/GauNEGF/](https://wliverno.github.io/GauNEGF/)

This package builds upon the foundation laid by the open-source ANT.Gaussian package. The Bethe lattice implementation and numerical quadrature methods are adapted from ANT.Gaussian's Fortran implementation (Palacios et al., 2002; Jacob & Palacios, 2011).

The package incorporates methods from several key works in the field:
- NEGF-DFT integration based on Damle et al. (2002)
- SCF convergence acceleration using Pulay's method (1980) as implemented in ANT.Gaussian
- Spin-dependent transport calculations inspired by Zöllner et al. (2020)

Note: This package requires a licensed copy of Gaussian quantum chemistry software to run. Gaussian is a registered trademark of Gaussian, Inc.

## Features

- Integration with Gaussian quantum chemistry package
- Self-consistent field (SCF) calculations with NEGF formalism
- Surface Green's function calculations using Bethe lattice approach
- Transport calculations for molecular junctions
- Support for spin-polarized calculations
- Parallel processing capabilities

## Requirements

### Required Software
- gauopen 3.0.0 Python interface for Gaussian

### Python Dependencies
- numpy
- scipy
- matplotlib

## Installation

```bash
# Clone the repository
git clone git@github.com:wliverno/GauNEGF.git
cd GauNEGF

# Install using pip
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

Full API documentation with examples has been compiled and deployed with Github Pages: [https://wliverno.github.io/GauNEGF/](https://wliverno.github.io/GauNEGF/)

For usage examples, see the files in the `examples/` directory.

## Citation

If you use GaussianNEGF in your research, please cite:

```bibtex
@software{gauNEGF2025
  author       = {Livernois, William},
  license      = {MIT},
  month        = April,
  year         = {2025},
  title        = {{GauNEGF}},
  url          = {https://github.com/wliverno/GauNEGF},
  version      = {v0.1.0-alpha},
  doi          = {10.5281/zenodo.15178667},
  url          = {https://doi.org/10.5281/zenodo.15178667}
}
```

## Acknowledgments

* Prof. M. P. Anantram (University of Washington) - Debugging and testing underlying NEGF framework and quantum transport physics
* Dr. Mike Frisch (Gaussian Inc.) - Debugging and patching issues within Gaussian, theory and implementation for non-collinear spin treatment
* Prof. Juan José Palacios (Universidad Autónoma de Madrid) - Theory and implementation for Fermi energy calculation and energy-dependent contacts
* Alex Bernson - Technical editing and AI meta-framework development for documentation

## References

1. Damle, P., Ghosh, A. W., & Datta, S. (2002). First-principles analysis of molecular conduction using quantum chemistry software. *Chemical Physics*, 281(2-3), 171-187. https://doi.org/10.1016/S0301-0104(02)00496-2
2. Jacob, D., & Palacios, J. J. (2011). Critical comparison of electrode models in density functional theory based quantum transport calculations. *The Journal of Chemical Physics*, 134(4), 044118. https://doi.org/10.1063/1.3526044
3. Palacios, J. J., Pérez-Jiménez, A. J., Louis, E., SanFabián, E., & Vergés, J. A. (2002). First-principles approach to electrical transport in atomic-scale nanostructures. *Physical Review B*, 66(3), 035322. https://doi.org/10.1103/PhysRevB.66.035322
4. Pulay, P. (1980). Convergence acceleration of iterative sequences. the case of scf iteration. *Chemical Physics Letters*, 73(2), 393-398. https://doi.org/10.1016/0009-2614(80)80396-4
5. Zöllner, M. S., Varela, S., Medina, E., Mujica, V., & Herrmann, C. (2020). Insight into the Origin of Chiral-Induced Spin Selectivity from a Symmetry Analysis of Electronic Transmission. *Journal of Chemical Theory and Computation*, 16(5), 2914-2929. https://doi.org/10.1021/acs.jctc.9b01078

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
