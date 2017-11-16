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

from sumpy.kernel import (LaplaceKernel, HelmholtzKernel,
        BiharmonicKernel, YukawaKernel, StokesKernel)


# {{{ pde check for kernels

class BiharmonicKernelInfo:
    def __init__(self, dim):
        self.kernel = BiharmonicKernel(dim)
        self.extra_kwargs = {}

    @staticmethod
    def pde_func(cp, pot):
        return cp.laplace(cp.laplace(pot[0]))

    nderivs = 4


class YukawaKernelInfo:
    def __init__(self, dim, lam):
        self.kernel = YukawaKernel(dim)
        self.lam = lam
        self.extra_kwargs = {"lam": lam}

    def pde_func(self, cp, pot):
        return cp.laplace(pot[0]) - self.lam**2*pot[0]

    nderivs = 2


class StokesKernelInfo:
    def __init__(self, dim, f, mu):
        self.kernel = StokesKernel(dim)
        self.f = f
        self.mu = mu
        extra_kwargs = dict(("f_{}".format(i), f[i]) for i in range(dim))
        extra_kwargs["mu"] = mu
        self.extra_kwargs = extra_kwargs
        self.dim = dim

    def pde_func(self, cp, pot, num=0):
        if num == self.dim:
            return sum(cp.diff(i, pot[i]) for i in range(self.dim))
        else:
            return (self.mu * sum(cp.diff(i, pot[num], 2) for i in range(self.dim)) +
                    cp.diff(num, pot[self.dim], 1))

    nderivs = 2


@pytest.mark.parametrize("knl_info", [
    BiharmonicKernelInfo(2),
    BiharmonicKernelInfo(3),
    YukawaKernelInfo(2, 5),
    StokesKernelInfo(3, [1, 2, 3], 4),
    ])
def test_pde_check_kernels(ctx_factory, knl_info, order=5):
    dim = knl_info.kernel.dim
    tctx = t.ToyContext(ctx_factory(), knl_info.kernel,
            extra_source_kwargs=knl_info.extra_kwargs)

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

        pde = knl_info.pde_func(cp, pot)

        err = la.norm(pde)
        eoc_rec.add_data_point(h, err)

    print(eoc_rec)
    assert eoc_rec.order_estimate() > order - knl_info.nderivs + 1 - 0.1

# }}}


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


class FakeTree:
    def __init__(self, dimensions, root_extent, stick_out_factor):
        self.dimensions = dimensions
        self.root_extent = root_extent
        self.stick_out_factor = stick_out_factor


@pytest.mark.parametrize("knl", [LaplaceKernel(2), HelmholtzKernel(2)])
def test_order_finder(knl):
    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    ofind = SimpleExpansionOrderFinder(1e-5)

    tree = FakeTree(knl.dim, 200, 0.5)
    orders = [
        ofind(knl, frozenset([("k", 5)]), tree, level)
        for level in range(30)]
    print(orders)


# You can test individual routines by typing
# $ python test_misc.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
