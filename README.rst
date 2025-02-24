sumpy: n-body kernels and translation operators
===============================================

.. image:: https://gitlab.tiker.net/inducer/sumpy/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/sumpy/commits/main
.. image:: https://github.com/inducer/sumpy/actions/workflows/ci.yml/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/sumpy/actions/workflows/ci.yml
.. image:: https://badge.fury.io/py/sumpy.svg
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/sumpy/
.. image:: https://zenodo.org/badge/1856097.svg
    :alt: Zenodo DOI for latest release
    :target: https://zenodo.org/badge/latestdoi/1856097

sumpy is mainly a 'scaffolding' package for Fast Multipole and quadrature methods.
If you're building one of those and need code generation for the required multipole
and local expansions, come right on in. Together with ``boxtree``, there is a full,
symbolically kernel-independent FMM implementation here.

It relies on

* `boxtree <https://pypi.org/project/boxtree>`__ for FMM tree building
* `loopy <https://pypi.org/project/loopy>`__ for fast array operations
* `pytest <https://pypi.org/project/pytest>`__ for automated testing

and, indirectly,

* `PyOpenCL <https://pypi.org/project/pyopencl>`__ as computational infrastructure

Resources:

* `documentation <https://documen.tician.de/sumpy>`__
* `source code via git <https://github.com/inducer/sumpy>`__
* `benchmarks <https://documen.tician.de/sumpy/benchmarks>`__
