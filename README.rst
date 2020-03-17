sumpy: n-body kernels and translation operators
===============================================

.. image:: https://gitlab.tiker.net/inducer/sumpy/badges/master/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/sumpy/commits/master
.. image:: https://github.com/inducer/sumpy/workflows/CI/badge.svg?branch=master
    :alt: Github Build Status
    :target: https://github.com/inducer/sumpy/actions?query=branch%3Amaster+workflow%3ACI
.. image:: https://badge.fury.io/py/sumpy.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/sumpy/

Sumpy is mainly a 'scaffolding' package for Fast Multipole and quadrature methods.
If you're building one of those and need code generation for the required Multipole
and local expansions, come right on in. Together with boxtree, there is a full,
symbolically kernel-independent FMM implementation here.

Sumpy relies on

* `numpy <http://pypi.python.org/pypi/numpy>`_ for arrays
* `boxtree <http://pypi.python.org/pypi/boxtree>`_ for FMM tree building
* `sumpy <http://pypi.python.org/pypi/sumpy>`_ for expansions and analytical routines
* `loopy <http://pypi.python.org/pypi/loo.py>`_ for fast array operations
* `pytest <http://pypi.python.org/pypi/pytest>`_ for automated testing

and, indirectly,

* `PyOpenCL <http://pypi.python.org/pypi/pyopencl>`_ as computational infrastructure

PyOpenCL is likely the only package you'll have to install
by hand, all the others will be installed automatically.

Resources:

* `documentation <http://documen.tician.de/sumpy>`_
* `source code via git <http://github.com/inducer/sumpy>`_

If you can see inside the UIUC firewall, you may browse
`benchmark results <http://koelsch.d.tiker.net/benchmarks/asv/sumpy/>`_.
