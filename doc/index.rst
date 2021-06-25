Welcome to sumpy's documentation!
=================================

.. automodule:: sumpy

Sumpy is mainly a 'scaffolding' package for Fast Multipole and quadrature methods.
If you're building one of those and need code generation for the required Multipole
and local expansions, come right on in. Together with boxtree, there is a full,
symbolically kernel-independent FMM implementation here.

Contents
--------

.. toctree::
    :maxdepth: 2

    kernel
    expansion
    interactions
    codegen
    eval
    misc
    🚀 Github <https://github.com/inducer/sumpy>
    💾 Download Releases <https://pypi.org/project/sumpy>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Example
-------

.. literalinclude:: ../examples/curve-pot.py

