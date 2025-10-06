Installation Guide
=================

This guide covers installation of gauNEGF and its dependencies.

Prerequisites
-----------
* **Gaussian Development Version**
* **gauopen v2 Python Interface**
* Python 3.7 or higher
* NumPy 1.26 or lower (for gauopen-3.0.0 compatibility)
* JAX (for parallelization and GPU support)

Installation Steps
---------------

1. **Install Python Dependencies**

   .. code-block:: bash

      pip install numpy<=1.26 matplotlib jax

To install JAX with GPU support, you will need to install CUDA and also install the [cudaXX] extra, where XX is the CUDA version you are using (e.g. [cuda12] for CUDA 12). 

   .. code-block:: bash
      
      pip install jax[cuda12]

Environment Setup
-----------------
1. Ensure Gaussian is installed and properly licensed

   * Follow Gaussian installation guide for your system
   * Set up required environment variables
   * Test Gaussian installation

2. Set the required environment variables:

   .. code-block:: bash

       export GAUSSIAN_DIR=/path/to/gaussian
       export GAUSS_SCRDIR=/path/to/scratch

Note to use CUDA with JAX you will need to unset the $LD_LIBRARY_PATH environment variable before running Python:

   .. code-block:: bash

      unset $LD_LIBRARY_PATH

3. Set up `gauopen` running "make -f qc.make" in the gauopen directory.

Installing gauNEGF
--------------
Clone the repository and install:

.. code-block:: bash

    git clone https://github.com/wliverno/GauNEGF.git
    cd gauNEGF
    ./install.sh

Verification
----------
Test the installation by first creating a Gaussian input file, for example 'H2.gjf':

.. code-block:: text

    %chk=H2.chk
    #p b3lyp/6-31g(d,p)
    
    H2 molecule for NEGF-DFT
    
    0 1
    H    0.000000    0.000000    0.000000
    H    0.000000    0.000000    0.7414

Then you can run the following code to test the installation:

.. code-block:: python

    from gauNEGF.scf import NEGF
    
    # Initialize NEGF object
    negf = NEGF(fn='H2', func='b3lyp', basis='6-31g(d,p)')


This will initialize the NEGF object by reading from the checkpoint file or running a new SCF in Gaussian.

Common Issues
-----------

1. **Gaussian Not Found**

   * Check PATH settings
   * Verify environment variables
   * Test gdv command directly

2. **Import Errors**

   * Check Python version
   * Verify all dependencies installed
   * Check for version conflicts

