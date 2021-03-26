sumpy: n-body kernels and translation operators
===============================================

.. image:: https://gitlab.tiker.net/inducer/sumpy/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/sumpy/commits/main
.. image:: https://github.com/inducer/sumpy/workflows/CI/badge.svg?branch=main&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/sumpy/actions?query=branch%3Amain+workflow%3ACI+event%3Apush
.. image:: https://badge.fury.io/py/sumpy.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/sumpy/

Sumpy is mainly a 'scaffolding' package for Fast Multipole and quadrature methods.
If you're building one of those and need code generation for the required Multipole
and local expansions, come right on in. Together with boxtree, there is a full,
symbolically kernel-independent FMM implementation here.

Sumpy relies on

* `numpy <https://pypi.org/project/numpy>`_ for arrays
* `boxtree <https://pypi.org/project/boxtree>`_ for FMM tree building
* `sumpy <https://pypi.org/project/sumpy>`_ for expansions and analytical routines
* `loopy <https://pypi.org/project/loopy>`_ for fast array operations
* `pytest <https://pypi.org/project/pytest>`_ for automated testing

and, indirectly,

* `PyOpenCL <https://pypi.org/project/pyopencl>`_ as computational infrastructure

PyOpenCL is likely the only package you'll have to install
by hand, all the others will be installed automatically.

Resources:

* `documentation <https://documen.tician.de/sumpy>`_
* `source code via git <https://github.com/inducer/sumpy>`_
* `benchmarks <https://documen.tician.de/sumpy/benchmarks>`_
