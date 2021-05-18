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
import sumpy.symbolic as sym

import pytest
import pyopencl as cl  # noqa: F401
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from pytools import Record

from sumpy.kernel import (LaplaceKernel, HelmholtzKernel,
        BiharmonicKernel, YukawaKernel, StokesletKernel, StressletKernel,
        ElasticityKernel, LineOfCompressionKernel)
from sumpy.expansion.diff_op import (make_identity_diff_op, gradient,
        divergence, laplacian, concat, as_scalar_pde, curl, diff)


# {{{ pde check for kernels

class KernelInfo:
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.extra_kwargs = kwargs
        diff_op = self.kernel.get_pde_as_diff_op()
        assert len(diff_op.eqs) == 1
        eq = diff_op.eqs[0]
        self.eq = eq

    def pde_func(self, cp, pot):
        subs_dict = {sym.Symbol(k): v for k, v in self.extra_kwargs.items()}
        result = 0
        for ident, coeff in self.eq.items():
            lresult = pot
            for axis, nderivs in enumerate(ident.mi):
                lresult = cp.diff(axis, lresult, nderivs)
            result += lresult*float(sym.sympify(coeff).xreplace(subs_dict))
        return result

    @property
    def nderivs(self):
        return max(sum(ident.mi) for ident in self.eq.keys())


@pytest.mark.parametrize("knl_info", [
    KernelInfo(BiharmonicKernel(2)),
    KernelInfo(BiharmonicKernel(3)),
    KernelInfo(YukawaKernel(2), lam=5),
    KernelInfo(YukawaKernel(3), lam=5),
    KernelInfo(LaplaceKernel(2)),
    KernelInfo(LaplaceKernel(3)),
    KernelInfo(HelmholtzKernel(2), k=5),
    KernelInfo(HelmholtzKernel(3), k=5),
    KernelInfo(StokesletKernel(2, 0, 1), mu=5),
    KernelInfo(StokesletKernel(2, 1, 1), mu=5),
    KernelInfo(StokesletKernel(3, 0, 1), mu=5),
    KernelInfo(StokesletKernel(3, 1, 1), mu=5),
    KernelInfo(StressletKernel(2, 0, 0, 0), mu=5),
    KernelInfo(StressletKernel(2, 0, 0, 1), mu=5),
    KernelInfo(StressletKernel(3, 0, 0, 0), mu=5),
    KernelInfo(StressletKernel(3, 0, 0, 1), mu=5),
    KernelInfo(StressletKernel(3, 0, 1, 2), mu=5),
    KernelInfo(ElasticityKernel(2, 0, 1), mu=5, nu=0.2),
    KernelInfo(ElasticityKernel(2, 0, 0), mu=5, nu=0.2),
    KernelInfo(ElasticityKernel(3, 0, 1), mu=5, nu=0.2),
    KernelInfo(ElasticityKernel(3, 0, 0), mu=5, nu=0.2),
    KernelInfo(LineOfCompressionKernel(3, 0), mu=5, nu=0.2),
    KernelInfo(LineOfCompressionKernel(3, 1), mu=5, nu=0.2),
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
    assert eoc_rec.order_estimate() > order - knl_info.nderivs + 1 - 0.2

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
def test_toy_p2e2e2p(ctx_factory, case):
    dim = case.dim

    src = case.source.reshape(dim, -1)
    tgt = case.target.reshape(dim, -1)

    if not 0 <= case.conv_factor <= 1:
        raise ValueError(
                "convergence factor not in valid range: %e" % case.conv_factor)

    from sumpy.expansion.local import VolumeTaylorLocalExpansion
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion

    cl_ctx = ctx_factory()
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


def test_cse_matvec():
    from sumpy.expansion import CSEMatVecOperator
    input_coeffs = [
        [(0, 2)],
        [],
        [(1, 1)],
        [(1, 9)],
    ]

    output_coeffs = [
        [],
        [(0, 3)],
        [],
        [(2, 7), (1, 5)],
    ]

    op = CSEMatVecOperator(input_coeffs, output_coeffs, shape=(4, 2))
    m = np.array([[2, 0], [6, 0], [0, 1], [30, 16]])

    vec = np.random.random(2)
    expected_result = m @ vec
    actual_result = op.matvec(vec)
    assert np.allclose(expected_result, actual_result)

    vec = np.random.random(4)
    expected_result = m.T @ vec
    actual_result = op.transpose_matvec(vec)
    assert np.allclose(expected_result, actual_result)


def test_diff_op_stokes():
    from sumpy.symbolic import symbols, Function
    diff_op = make_identity_diff_op(3, 4)
    u = diff_op[:3]
    p = diff_op[3]
    pde = concat(laplacian(u) - gradient(p), divergence(u))

    actual_output = pde.to_sym()
    x, y, z = syms = symbols("x0, x1, x2")
    funcs = symbols("f0, f1, f2, f3", cls=Function)
    u, v, w, p = [f(*syms) for f in funcs]

    eq1 = u.diff(x, x) + u.diff(y, y) + u.diff(z, z) - p.diff(x)
    eq2 = v.diff(x, x) + v.diff(y, y) + v.diff(z, z) - p.diff(y)
    eq3 = w.diff(x, x) + w.diff(y, y) + w.diff(z, z) - p.diff(z)
    eq4 = u.diff(x) + v.diff(y) + w.diff(z)

    expected_output = [eq1, eq2, eq3, eq4]

    assert expected_output == actual_output


def test_as_scalar_pde_stokes():
    diff_op = make_identity_diff_op(3, 4)
    u = diff_op[:3]
    p = diff_op[3]
    pde = concat(laplacian(u) - gradient(p), divergence(u))

    # velocity components in Stokes should satisfy Biharmonic
    for i in range(3):
        print(as_scalar_pde(pde, i))
        print(laplacian(laplacian(u[i])))
        assert as_scalar_pde(pde, i) == laplacian(laplacian(u[0]))

    # pressure should satisfy Laplace
    assert as_scalar_pde(pde, 3) == laplacian(u[0])


def test_as_scalar_pde_maxwell():
    from sumpy.symbolic import symbols
    op = make_identity_diff_op(3, 6, time_dependent=True)
    E = op[:3]  # noqa: N806
    B = op[3:]  # noqa: N806
    mu, epsilon = symbols("mu, epsilon")
    t = (0, 0, 0, 1)

    pde = concat(curl(E) + diff(B, t),  curl(B) - mu*epsilon*diff(E, t),
                 divergence(E), divergence(B))
    as_scalar_pde(pde, 3)

    for i in range(6):
        assert as_scalar_pde(pde, i) == \
            -1/(mu*epsilon)*laplacian(op[0]) + diff(diff(op[0], t), t)


def test_as_scalar_pde_elasticity():

    # Ref: https://doi.org/10.1006/jcph.1996.0102

    diff_op = make_identity_diff_op(2, 5)
    sigma_x = diff_op[0]
    sigma_y = diff_op[1]
    tau = diff_op[2]
    u = diff_op[3]
    v = diff_op[4]

    # Use numeric values as the expressions grow exponentially large otherwise
    lam, mu = 2, 3

    x = (1, 0)
    y = (0, 1)

    exprs = [
        diff(sigma_x, x) + diff(tau, y),
        diff(tau, x) + diff(sigma_y, y),
        sigma_x - (lam + 2*mu)*diff(u, x) - lam*diff(v, y),
        sigma_y - (lam + 2*mu)*diff(v, y) - lam*diff(u, x),
        tau - mu*(diff(u, y) + diff(v, x)),
    ]

    pde = concat(*exprs)
    for i in range(5):
        assert as_scalar_pde(pde, i) == laplacian(laplacian(diff_op[0]))


def test_elasticity_new():
    from pickle import dumps, loads
    stokes_knl = StokesletKernel(3, 0, 1, "mu1", 0.5)
    stokes_knl2 = ElasticityKernel(3, 0, 1, "mu1", 0.5)
    elasticity_knl = ElasticityKernel(3, 0, 1, "mu1", "nu")
    elasticity_helper_knl = LineOfCompressionKernel(3, 0, "mu1", "nu")

    assert isinstance(stokes_knl2, StokesletKernel)
    assert stokes_knl == stokes_knl2
    assert loads(dumps(stokes_knl)) == stokes_knl

    for knl in [elasticity_knl, elasticity_helper_knl]:
        assert not isinstance(knl, StokesletKernel)
        assert loads(dumps(knl)) == knl


# You can test individual routines by typing
# $ python test_misc.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
