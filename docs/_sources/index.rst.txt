.. pySDR documentation master file, created by
   sphinx-quickstart on Thu Feb 16 19:33:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pySDR's documentation!
=================================
**pySDR** (/py'SDR/) is a Python libary for performing local gradient clustering (LGC) based sharpened dimensionality reduction (SDR).
The library uses the `LGCDR_v1 <https://github.com/youngjookim/sdr-nnp/tree/main/Code/LGCDR_v1>`_ code written by Youngjoo Kim 
in C++ as its backend.

`LGCDR_v1 <https://github.com/youngjookim/sdr-nnp/tree/main/Code/LGCDR_v1>`_ uses `nanoflann <https://jlblancoc.github.io/nanoflann/>`_ for local gradient clustering,
`Tapkee <https://tapkee.lisitsyn.me/>`_ for dimensionality reduction 
and `Eigen 3.3.9 <https://gitlab.com/libeigen/eigen/-/releases/3.3.9>`_ for linear algebra. 

Installation
------------
**pySDR** includes shared C libraries that were compiled from the `LGCDR_v1 <https://github.com/youngjookim/sdr-nnp/tree/main/Code/LGCDR_v1>`_ code.
Due to this dependency only 64bit Windows and Linux machines are currently supported.
To install **pySDR**, in your terminal go to the `python` directory within the project directory and run:

.. code-block:: bash

   $ pip install .

Contents
--------
.. toctree::
   :maxdepth: 2

   api



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
