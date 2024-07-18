from __future__ import annotations


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
import sys

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

import sumpy.symbolic as sym
from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf  # noqa: F401
from sumpy.tools import (
    fft,
    fft_toeplitz_upper_triangular,
    loopy_fft,
    matvec_toeplitz_upper_triangular,
)


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_matvec_fft

def test_matvec_fft():
    k = 5

    rng = np.random.default_rng(42)
    v = rng.random(k)
    x = rng.random(k)

    fft = fft_toeplitz_upper_triangular(v, x)
    matvec = matvec_toeplitz_upper_triangular(v, x)

    for i in range(k):
        assert abs(fft[i] - matvec[i]) < 1e-14

# }}}


# {{{ test_matvec_fft_small_floats

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

# }}}


# {{{ test_fft

@pytest.mark.parametrize("size", [1, 2, 7, 10, 30, 210])
def test_fft(actx_factory, size):
    actx = actx_factory()

    inp = np.arange(size, dtype=np.complex64)
    inp_dev = actx.from_numpy(inp)
    out = fft(inp)

    fft_func = loopy_fft(inp.shape, inverse=False, complex_dtype=inp.dtype.type)
    _evt, (out_dev,) = fft_func(actx.queue, y=inp_dev)

    assert np.allclose(actx.to_numpy(out_dev), out)

# }}}


# You can test individual routines by typing
# $ python test_tools.py 'test_fft(_acf, 30)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
