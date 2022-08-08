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
    matvec_toeplitz_upper_triangular, loopy_fft, fft)
import numpy as np

import pyopencl as cl
import pyopencl.array as cla
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def test_matvec_fft():
    k = 5
    v = np.random.rand(k)
    x = np.random.rand(k)

    fft = fft_toeplitz_upper_triangular(v, x)
    matvec = matvec_toeplitz_upper_triangular(v, x)

    for i in range(k):
        assert abs(fft[i] - matvec[i]) < 1e-14


def test_matvec_fft_small_floats():
    k = 5
    v = sym.make_sym_vector("v", k)
    x = sym.make_sym_vector("x", k)

    fft = fft_toeplitz_upper_triangular(v, x)
    for expr in fft:
        for f in expr.atoms(sym.Float):
            if f == 0:
                continue
            assert abs(f) > 1e-10

def test_fft(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    inp = np.arange(10, dtype=np.complex64)
    inp_dev = cla.to_device(queue, inp)
    out = fft(inp)

    fft_func = loopy_fft(inp.shape, inverse=False, complex_dtype=inp.dtype.type)
    evt, (out_dev,) = fft_func(queue, y=inp_dev)
    assert np.allclose(out_dev.get(), out)
