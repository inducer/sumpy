__copyright__ = "Copyright (C) 2020 Isuru Fernando"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
logger = logging.getLogger(__name__)

import sumpy.symbolic as sym
from sumpy.tools import (fft_toeplitz_upper_triangular,
    matvec_toeplitz_upper_triangular)
import numpy as np


def test_fft():
    k = 5
    v = np.random.rand(k)
    x = np.random.rand(k)

    fft = fft_toeplitz_upper_triangular(v, x)
    matvec = matvec_toeplitz_upper_triangular(v, x)

    for i in range(k):
        assert abs(fft[i] - matvec[i]) < 1e-14


def test_fft_small_floats():
    k = 5
    v = sym.make_sym_vector("v", k)
    x = sym.make_sym_vector("x", k)

    fft = fft_toeplitz_upper_triangular(v, x)
    for expr in fft:
        for f in expr.atoms(sym.Float):
            if f == 0:
                continue
            assert abs(f) > 1e-10
