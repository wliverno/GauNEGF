"""
Global configuration settings for gauNEGF calculations.

This module provides centralized default parameters used throughout the gauNEGF package.
"""

# Physical Parameters
TEMPERATURE = 0.0               # Kelvin - ambient temperature
ETA = 1e-5                      # eV - broadening parameter  
ENERGY_STEP = 0.001             # eV - default energy step size

# Contact Tolerances
FERMI_CALCULATION_TOL = 1e-3        # Fermi energy calculation tolerance
OVERLAP_EIGENVALUE_RATIO = 1e-6     # Floor overlap eigenvalues below this fraction of max
FERMI_SEARCH_CYCLES = 10            # Number of cycles to run search before returning
SURFACE_GREEN_CONVERGENCE = 1e-5    # Surface Green's function convergence
SURFACE_RELAXATION_FACTOR = 0.1     # Relaxation/Mixing factor for Green's function convergence
FERMI_DEBUG=True                    # Debugging for fermi search functions

# Integration Parameters
ADAPTIVE_INTEGRATION_TOL = 1e-4     # Adaptive integration tolerance
N_KT = 10                           # Number of kT for integration limits
ENERGY_MIN = -1e6                   # eV - lower bound for energy integration - MUST BE NEGATIVE
MAX_CYCLES = 100                    # Maximum iteration cycles
MAX_GRID_POINTS = 500               # Maximum number of grid points

# SCF Parameters
SCF_DAMPING = 0.02              # SCF damping parameter
SCF_CONVERGENCE_TOL = 1e-3      # SCF convergence tolerance
SCF_MAX_CYCLES = 100            # Maximum SCF cycles
PULAY_MIXING_SIZE = 4           # Number of iterations for Pulay mixing

# GPU/CPU Integration Logging Configuration
LOG_LEVEL = 'DEBUG'              # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_PERFORMANCE = False          # Enable GPU performance logging

# JAX Configuration
JAX_ENABLE_X64 = True            # Use 64-bit precision (required for NEGF calculations)
JAX_GPU_MEMORY_FRACTION = 0.8    # Fraction of GPU memory to preallocate (0.0-1.0)
JAX_PREALLOCATE = False          # Preallocate GPU memory (False = allocate as needed, reduces OOM)
JAX_PLATFORM = None              # Force specific platform: 'cpu', 'gpu', or None for auto-detect

# Configure JAX on module import
import os
import multiprocessing

# GPU memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(JAX_PREALLOCATE).lower()
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(JAX_GPU_MEMORY_FRACTION)
# On older GPUs the BFC allocator can OOM on large matrix ops despite sufficient VRAM;
# use cudaMalloc directly instead by uncommenting the line below (set in job script instead
# to avoid impacting performance on newer GPUs where BFC is preferable):
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# CPU threading settings (may improve multi-core utilization)
CPU_THREADS = multiprocessing.cpu_count()
os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={CPU_THREADS}'

import jax

# Enable 64-bit precision
jax.config.update('jax_enable_x64', JAX_ENABLE_X64)

# Platform selection
if JAX_PLATFORM is not None:
    jax.config.update('jax_platform_name', JAX_PLATFORM)

# Clear compilation cache function
jax.config.update('jax_compilation_cache_dir', None)  # Disable persistent cache

# Detect backend and report
_backend = jax.default_backend()
_devices = jax.devices()
if _backend == 'gpu':
    print(f"JAX GPU backend: {len(_devices)} device(s), Memory fraction: {JAX_GPU_MEMORY_FRACTION}, Preallocate: {JAX_PREALLOCATE}")
else:
    print(f"JAX {_backend.upper()} backend: {len(_devices)} device(s)")

# Create global device mesh for automatic sharding
# This enables parallel computation across all available devices (CPU/GPU/TPU)
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh

# Create 1D mesh along 'data' axis using all available devices
DEVICE_MESH = Mesh(create_device_mesh((len(_devices),)), ('data',))

# Helper function to shard arrays across the device mesh
def shard_array(array, axis=0):
    """
    Shard an array across all devices along the specified axis.
    Use this for input arrays before passing to jitted functions.

    Args:
        array: JAX array to shard
        axis: Which axis to shard (default: 0, typically batch dimension)

    Returns:
        Sharded array distributed across devices
    """
    import jax
    # Create partition spec: shard along specified axis, replicate others
    pspec = [None] * array.ndim
    pspec[axis] = 'data'
    sharding = NamedSharding(DEVICE_MESH, P(*pspec))
    return jax.device_put(array, sharding)

