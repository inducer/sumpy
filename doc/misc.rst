Installation
============

This command should install :mod:`sumpy`::

    pip install sumpy

You may need to run this with :command:`sudo`.
If you don't already have `pip <https://pypi.python.org/pypi/pip>`_,
run this beforehand::

    curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py

For a more manual installation, download the source, unpack it,
and say::

    python setup.py install

In addition, you need to have :mod:`numpy` installed.

Symbolic backends
=================

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
`Wiki FAQ page <http://wiki.tiker.net/Sumpy/FrequentlyAskedQuestions>`_.

Acknowledgments
===============

Andreas Klöckner's work on :mod:`sumpy` was supported in part by

* US Navy ONR grant number N00014-14-1-0117
* the US National Science Foundation under grant numbers DMS-1418961 and CCF-1524433.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.  The
views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
