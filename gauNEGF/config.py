"""
Global configuration settings for gauNEGF calculations.

This module provides centralized default parameters used throughout the gauNEGF package.
"""

# Physical Parameters
TEMPERATURE = 0.0               # Kelvin - ambient temperature
ETA = 1e-6                      # eV - broadening parameter  
ENERGY_STEP = 0.001             # eV - default energy step size

# Contact Tolerances
FERMI_CALCULATION_TOL = 1e-3        # Fermi energy calculation tolerance
FERMI_SEARCH_CYCLES = 10            # Number of cycles to run search before returning
SURFACE_GREEN_CONVERGENCE = 1e-5    # Surface Green's function convergence
SURFACE_RELAXATION_FACTOR = 0.1     # Relaxation/Mixing factor for Green's function convergence

# Integration Parameters
ADAPTIVE_INTEGRATION_TOL = 1e-4     # Adaptive integration tolerance
N_KT = 10                           # Number of kT for integration limits
ENERGY_MIN = -1e6                   # eV - lower bound for energy integration
MAX_CYCLES = 1000                   # Maximum iteration cycles
MAX_GRID_POINTS = 1000              # Maximum number of grid points

# SCF Parameters
SCF_DAMPING = 0.02              # SCF damping parameter
SCF_CONVERGENCE_TOL = 1e-3      # SCF convergence tolerance
SCF_MAX_CYCLES = 100            # Maximum SCF cycles
PULAY_MIXING_SIZE = 4           # Number of iterations for Pulay mixing

# GPU/CPU Integration Logging Configuration
LOG_LEVEL = 'DEBUG'              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_PERFORMANCE = True           # Enable GPU performance logging

