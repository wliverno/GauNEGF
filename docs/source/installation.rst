Installation Guide
=================

This guide covers installation of gauNEGF and its dependencies.

Prerequisites
-----------
* **Gaussian Development Version**
* **gauopen v2 Python Interface**
* Python 3.7 or higher
* Gaussian 16 or newer
* NumPy and SciPy
* Matplotlib (for plotting)

Installation Steps
---------------

1. **Install Python Dependencies**

   .. code-block:: bash

      pip install numpy scipy matplotlib

Gaussian Setup
------------
1. Ensure Gaussian is installed and properly licensed

   * Follow Gaussian installation guide for your system
   * Set up required environment variables
   * Test Gaussian installation

2. Set the required environment variables:

   .. code-block:: bash

       # Linux/MacOS
       export GAUSSIAN_DIR=/path/to/gaussian
       export GAUSS_SCRDIR=/path/to/scratch
       
       # Windows (PowerShell)
       $env:GAUSSIAN_DIR = "C:\path\to\gaussian"
       $env:GAUSS_SCRDIR = "C:\path\to\scratch"

3. Set up `gauopen` following the directions on the Gaussian website (*TODO*)

Installing gauNEGF
--------------
Clone the repository and install:

.. code-block:: bash

    git clone https://github.com/your-username/gauNEGF.git
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
   * Test g16 command directly

2. **Import Errors**

   * Check Python version
   * Verify all dependencies installed
   * Check for version conflicts

