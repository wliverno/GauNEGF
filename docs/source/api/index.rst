API Reference
============

This section provides detailed API documentation for all gauNEGF modules.

Core Modules
===========

NEGF Base Class
--------------

.. automodule:: gauNEGF.scf
   :members:
   :undoc-members:
   :show-inheritance:

Energy-Dependent NEGF
-------------------

.. seealso::

   :doc:`/guides/contacts_1d` -- 1D chain contact setup via NEGFE.setContact1D

   :doc:`/guides/contacts_bethe` -- Bethe lattice contact setup via NEGFE.setContactBethe

   :doc:`/guides/contact_choice` -- Choosing between contact types

.. automodule:: gauNEGF.scfE
   :members:
   :undoc-members:
   :show-inheritance:

Density Module
------------

.. automodule:: gauNEGF.density
   :members:
   :undoc-members:
   :show-inheritance:

Transport Module
--------------

.. seealso::

   :doc:`/guides/workflow_recipes` -- IV curve sweep, checkpointing, multi-temperature workflows

.. automodule:: gauNEGF.transport
   :members:
   :undoc-members:
   :show-inheritance:

Contact Models
============

Bethe Lattice
------------

.. seealso::

   :doc:`/guides/contacts_bethe` -- Full Bethe lattice contact guide (Au, AuSOC, SOC workflows)

.. automodule:: gauNEGF.surfGBethe
   :members:
   :undoc-members:
   :show-inheritance:

1D Chain
-------

.. seealso::

   :doc:`/guides/contacts_1d` -- 1D chain contact setup guide (three usage patterns)

.. automodule:: gauNEGF.surfG1D
   :members:
   :undoc-members:
   :show-inheritance:

3D Contacts
----------

.. automodule:: gauNEGF.surfG3D
   :members:
   :undoc-members:
   :show-inheritance:

Constant Self Energy
-------------------

.. automodule:: gauNEGF.surfGTester
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
=======

Matrix Tools
-----------

.. automodule:: gauNEGF.matTools
   :members:
   :undoc-members:
   :show-inheritance:

Integration Tools
-----------

.. automodule:: gauNEGF.integrate
   :members:
   :undoc-members:
   :show-inheritance:

Spin Tools
---------

.. automodule:: gauNEGF.spinTools
   :members:
   :undoc-members:
   :show-inheritance:

JIT / Linear Algebra Helpers
---------------------------

.. automodule:: gauNEGF.utils
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Reference
======================

.. seealso::

   :doc:`/guides/config_tuning` -- When to change ETA, SCF_DAMPING, fermiMethod, and TEMPERATURE

.. automodule:: gauNEGF.config
   :members:
   :undoc-members:
   :show-inheritance:

Developer / Extensibility Reference
==================================

.. automodule:: gauNEGF.protocols
   :members:
   :undoc-members:
   :show-inheritance:
