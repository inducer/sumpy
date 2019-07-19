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

from pytools import Record

from sumpy.kernel import (LaplaceKernel, HelmholtzKernel,
        BiharmonicKernel, YukawaKernel)


# {{{ pde check for kernels

class BiharmonicKernelInfo:
    def __init__(self, dim):
        self.kernel = BiharmonicKernel(dim)
        self.extra_kwargs = {}

    @staticmethod
    def pde_func(cp, pot):
        return cp.laplace(cp.laplace(pot))

    nderivs = 4


class YukawaKernelInfo:
    def __init__(self, dim, lam):
        self.kernel = YukawaKernel(dim)
        self.lam = lam
        self.extra_kwargs = {"lam": lam}

    def pde_func(self, cp, pot):
        return cp.laplace(pot) - self.lam**2*pot

    nderivs = 2


@pytest.mark.parametrize("knl_info", [
    BiharmonicKernelInfo(2),
    BiharmonicKernelInfo(3),
    YukawaKernelInfo(2, 5),
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


@pytest.mark.parametrize("knl", [
        LaplaceKernel(2), HelmholtzKernel(2),
        LaplaceKernel(3), HelmholtzKernel(3)])
def test_order_finder(knl):
    from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder

    ofind = SimpleExpansionOrderFinder(1e-5)

    tree = FakeTree(knl.dim, 200, 0.5)
    orders = [
        ofind(knl, frozenset([("k", 5)]), tree, level)
        for level in range(30)]
    print(orders)

    # Order should not increase with level
    assert (np.diff(orders) <= 0).all()


@pytest.mark.parametrize("knl", [
        LaplaceKernel(2), HelmholtzKernel(2),
        LaplaceKernel(3), HelmholtzKernel(3)])
def test_fmmlib_order_finder(knl):
    pytest.importorskip("pyfmmlib")
    from sumpy.expansion.level_to_order import FMMLibExpansionOrderFinder

    ofind = FMMLibExpansionOrderFinder(1e-5)

    tree = FakeTree(knl.dim, 200, 0.5)
    orders = [
        ofind(knl, frozenset([("k", 5)]), tree, level)
        for level in range(30)]
    print(orders)

    # Order should not increase with level
    assert (np.diff(orders) <= 0).all()


# {{{ expansion toys p2e2e2p test cases

def approx_convergence_factor(orders, errors):
    poly = np.polyfit(orders, np.log(errors), deg=1)
    return np.exp(poly[0])


class P2E2E2PTestCase(Record):

    @property
    def dim(self):
        return len(self.source)

    @staticmethod
    def eval(expr, source, center1, center2, target):
        from pymbolic import parse, evaluate
        context = {
                "s": source,
                "c1": center1,
                "c2": center2,
                "t": target,
                "norm": la.norm}

        return evaluate(parse(expr), context)

    def __init__(self,
            source, center1, center2, target, expansion1, expansion2, conv_factor):

        if isinstance(conv_factor, str):
            conv_factor = self.eval(conv_factor, source, center1, center2, target)

        Record.__init__(self,
                source=source,
                center1=center1,
                center2=center2,
                target=target,
                expansion1=expansion1,
                expansion2=expansion2,
                conv_factor=conv_factor)


P2E2E2P_TEST_CASES = (
        # local to local, 3D
        P2E2E2PTestCase(
            source=np.array([3., 4., 5.]),
            center1=np.array([1., 0., 0.]),
            center2=np.array([1., 3., 0.]),
            target=np.array([1., 1., 1.]),
            expansion1=t.local_expand,
            expansion2=t.local_expand,
            conv_factor="norm(t-c1)/norm(s-c1)"),

        # multipole to multipole, 3D
        P2E2E2PTestCase(
            source=np.array([1., 1., 1.]),
            center1=np.array([1., 0., 0.]),
            center2=np.array([1., 0., 3.]),
            target=np.array([3., 4., 5.]),
            expansion1=t.multipole_expand,
            expansion2=t.multipole_expand,
            conv_factor="norm(s-c2)/norm(t-c2)"),

        # multipole to local, 3D
        P2E2E2PTestCase(
            source=np.array([-2., 2., 1.]),
            center1=np.array([-2., 5., 3.]),
            center2=np.array([0., 0., 0.]),
            target=np.array([0., 0., -1]),
            expansion1=t.multipole_expand,
            expansion2=t.local_expand,
            conv_factor="norm(t-c2)/(norm(c2-c1)-norm(c1-s))"),
)

# }}}


ORDERS_P2E2E2P = (3, 4, 5)
RTOL_P2E2E2P = 1e-2


@pytest.mark.parametrize("case", P2E2E2P_TEST_CASES)
def test_toy_p2e2e2p(ctx_getter, case):
    dim = case.dim

    src = case.source.reshape(dim, -1)
    tgt = case.target.reshape(dim, -1)

    if not 0 <= case.conv_factor <= 1:
        raise ValueError(
                "convergence factor not in valid range: %e" % case.conv_factor)

    from sumpy.expansion.local import VolumeTaylorLocalExpansion
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion

    cl_ctx = ctx_getter()
    ctx = t.ToyContext(cl_ctx,
             LaplaceKernel(dim),
             VolumeTaylorMultipoleExpansion,
             VolumeTaylorLocalExpansion)

    errors = []

    src_pot = t.PointSources(ctx, src, weights=np.array([1.]))
    pot_actual = src_pot.eval(tgt).item()

    for order in ORDERS_P2E2E2P:
        expn = case.expansion1(src_pot, case.center1, order=order)
        expn2 = case.expansion2(expn, case.center2, order=order)
        pot_p2e2e2p = expn2.eval(tgt).item()
        errors.append(np.abs(pot_actual - pot_p2e2e2p))

    conv_factor = approx_convergence_factor(1 + np.array(ORDERS_P2E2E2P), errors)
    assert conv_factor <= min(1, case.conv_factor * (1 + RTOL_P2E2E2P)), \
        (conv_factor, case.conv_factor * (1 + RTOL_P2E2E2P))


# You can test individual routines by typing
# $ python test_misc.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
