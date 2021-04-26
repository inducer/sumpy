Misc Tools
==========

.. automodule:: sumpy.tools


Installation
============

This command should install :mod:`sumpy`::

    pip install sumpy

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.org/project/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, download the source, unpack it,
and say::

    python setup.py install

In addition, you need to have :mod:`numpy` installed.

Usage
=====

Environment variables
---------------------

+-----------------------------------+-----------------------------------------------------+
| Name                              | Purpose                                             |
+===================================+=====================================================+
| `SUMPY_FORCE_SYMBOLIC_BACKEND`    | Symbolic backend control, see `Symbolic backends`_  |
+-----------------------------------+-----------------------------------------------------+
| `SUMPY_NO_CACHE`                  | If set, disables the on-disk cache                  |
+-----------------------------------+-----------------------------------------------------+
| `SUMPY_NO_OPT`                    | If set, disables performance-oriented :mod:`loopy`  |
|                                   | transformations                                     |
+-----------------------------------+-----------------------------------------------------+

Symbolic backends
-----------------

:mod:`sumpy` supports two symbolic backends: sympy and SymEngine. To use the
SymEngine backend, ensure that the `SymEngine library
<https://github.com/symengine/symengine>`_ and the `SymEngine Python bindings
<https://github.com/symengine/symengine.py>`_ are installed.

By default, :mod:`sumpy` prefers using SymEngine but falls back to sympy if it
detects that SymEngine is not installed. To force the use of a particular
backend, set the environment variable `SUMPY_FORCE_SYMBOLIC_BACKEND` to
`symengine` or `sympy`.

User-visible Changes
====================

Version 2016.1
--------------
.. note::

    This version is currently under development. You can get snapshots from
    sumpy's `git repository <https://github.com/inducer/sumpy>`_

* Initial release.

.. _license:

License
=======

:mod:`sumpy` is licensed to you under the MIT/X Consortium license:

Copyright (c) 2012-16 Andreas Klöckner

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the
`Wiki FAQ page <https://wiki.tiker.net/Sumpy/FrequentlyAskedQuestions>`_.

Acknowledgments
===============

Work on meshmode was supported in part by

* the US National Science Foundation under grant numbers DMS-1418961,
  DMS-1654756, SHF-1911019, and OAC-1931577.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.

The views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
