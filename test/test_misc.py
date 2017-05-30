from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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

import numpy as np
import numpy.linalg as la
import sys
import sumpy.toys as t

import pytest
import pyopencl as cl  # noqa: F401
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.parametrize("dim", [2, 3])
def test_pde_check_biharmonic(ctx_factory, dim, order=5):
    from sumpy.kernel import BiharmonicKernel
    tctx = t.ToyContext(ctx_factory(), BiharmonicKernel(dim))

    pt_src = t.PointSources(
            tctx,
            np.random.rand(dim, 50) - 0.5,
            np.ones(50))

    from pytools.convergence import EOCRecorder
    from sumpy.point_calculus import CalculusPatch
    eoc_rec = EOCRecorder()

    for h in [0.1, 0.05, 0.025]:
        cp = CalculusPatch(np.array([1, 0, 0])[:dim], h=h, order=order)
        pot = pt_src.eval(cp.points)

        deriv = cp.laplace(cp.laplace(pot))

        err = la.norm(deriv)
        eoc_rec.add_data_point(h, err)

    print(eoc_rec)
    assert eoc_rec.order_estimate() > order-3-0.1


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pde_check(dim, order=4):
    from sumpy.point_calculus import CalculusPatch
    from pytools.convergence import EOCRecorder

    for iaxis in range(dim):
        eoc_rec = EOCRecorder()
        for h in [0.1, 0.01, 0.001]:
            cp = CalculusPatch(np.array([3, 0, 0])[:dim], h=h, order=order)
            df_num = cp.diff(iaxis, np.sin(10*cp.points[iaxis]))
            df_true = 10*np.cos(10*cp.points[iaxis])

            err = la.norm(df_num-df_true)
            eoc_rec.add_data_point(h, err)

        print(eoc_rec)
        assert eoc_rec.order_estimate() > order-2-0.1


# You can test individual routines by typing
# $ python test_misc.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
