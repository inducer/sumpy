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

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts

import sumpy.symbolic as sym
from sumpy.array_context import (  # ruff:ignore[unused-import]
    PytestPyOpenCLArrayContextFactory,
    _acf,
)
from sumpy.tools import (
    FFTBackend,
    fft,
    fft_toeplitz_upper_triangular,
    get_opencl_fft_app,
    loopy_fft,
    matvec_toeplitz_upper_triangular,
    run_opencl_fft,
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
def test_fft(actx_factory: ArrayContextFactory, size: int):
    actx = actx_factory()

    inp = np.arange(size, dtype=np.complex64)
    inp_dev = actx.from_numpy(inp)
    out = fft(inp)

    fft_func = loopy_fft(
                inp.shape[-1],
                n_batch_dims=len(inp.shape) - 1,
                inverse=False,
                complex_dtype=inp.dtype.type)
    out_dev = actx.call_loopy(fft_func, y=inp_dev)["x"]

    assert np.allclose(actx.to_numpy(out_dev), out)

# }}}


# {{{ test_real_fft

@pytest.mark.parametrize("size", [2, 7, 10, 15, 16])
def test_real_fft(actx_factory: ArrayContextFactory, size: int):
    pytest.importorskip("pyvkfft")

    actx = actx_factory()

    from pyopencl.characterize import get_pocl_version
    pocl_version = get_pocl_version(actx.queue.device.platform)
    if pocl_version and (7,) <= pocl_version < (7, 2):
        pytest.skip("pocl 7.0/7.1 and pyvkfft don't get along: "
                    "https://github.com/pocl/pocl/issues/2069")

    rng = np.random.default_rng(42)
    # We run one 1D FFT on each row of the data.
    data = rng.standard_normal((3, size))

    forward_app = get_opencl_fft_app(actx, shape=data.shape,
            dtype=data.dtype, inverse=False, backend=FFTBackend.PYVKFFT)
    inverse_app = get_opencl_fft_app(actx, shape=data.shape,
            dtype=data.dtype, inverse=True, backend=FFTBackend.PYVKFFT)

    _, spectrum = run_opencl_fft(actx, forward_app, actx.from_numpy(data))
    _, result = run_opencl_fft(actx, inverse_app, spectrum, inverse=True,
            n=size)

    assert np.max(np.abs(
        actx.to_numpy(spectrum) - np.fft.rfft(data, axis=-1))) < 1e-13
    assert np.max(np.abs(actx.to_numpy(result) - data)) < 1e-13

# }}}


# You can test individual routines by typing
# $ python test_tools.py 'test_fft(_acf, 30)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
