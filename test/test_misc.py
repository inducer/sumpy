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

import pytest
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.linalg as la

from arraycontext import pytest_generate_tests_for_array_contexts
from sumpy.array_context import (                                 # noqa: F401
        PytestPyOpenCLArrayContextFactory, _acf)

import sumpy.toys as t
import sumpy.symbolic as sym

from sumpy.kernel import (
    LaplaceKernel,
    HelmholtzKernel,
    BiharmonicKernel,
    YukawaKernel,
    StokesletKernel,
    StressletKernel,
    ElasticityKernel,
    LineOfCompressionKernel,
    AxisTargetDerivative,
    ExpressionKernel)
from sumpy.expansion.diff_op import (
    make_identity_diff_op, concat, as_scalar_pde, diff,
    gradient, divergence, laplacian, curl)
from sumpy.derivative_taker import get_pde_operators

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


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
def test_pde_check_kernels(actx_factory, knl_info, order=5):
    actx = actx_factory()

    dim = knl_info.kernel.dim
    tctx = t.ToyContext(actx.context, knl_info.kernel,
            extra_source_kwargs=knl_info.extra_kwargs)

    rng = np.random.default_rng(42)
    pt_src = t.PointSources(
            tctx,
            rng.random(size=(dim, 50)) - 0.5,
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

    logger.info("eoc:\n%s", eoc_rec)
    assert eoc_rec.order_estimate() > order - knl_info.nderivs + 1 - 0.1

# }}}


# {{{ test_pde_check

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

        logger.info("eoc:\n%s", eoc_rec)
        assert eoc_rec.order_estimate() > order-2-0.1

# }}}


# {{{ test_order_finder

@dataclass(frozen=True)
class FakeTree:
    dimensions: int
    root_extent: float
    stick_out_factor: float


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
    logger.info("orders: %s", orders)

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
    logger.info("orders: %s", orders)

    # Order should not increase with level
    assert (np.diff(orders) <= 0).all()

# }}}


# {{{ expansion toys p2e2e2p test cases

def approx_convergence_factor(orders, errors):
    poly = np.polyfit(orders, np.log(errors), deg=1)
    return np.exp(poly[0])


@dataclass(frozen=True)
class P2E2E2PTestCase:
    source: np.ndarray
    target: np.ndarray
    center1: np.ndarray
    center2: np.ndarray
    expansion1: Callable[..., Any]
    expansion2: Callable[..., Any]
    conv_factor: str

    @property
    def dim(self):
        return len(self.source)


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


# {{{ test_toy_p2e2e2p

ORDERS_P2E2E2P = (3, 4, 5)
RTOL_P2E2E2P = 1e-2


@pytest.mark.parametrize("case", P2E2E2P_TEST_CASES)
def test_toy_p2e2e2p(actx_factory, case):
    dim = case.dim

    src = case.source.reshape(dim, -1)
    tgt = case.target.reshape(dim, -1)

    from pymbolic import parse, evaluate
    case_conv_factor = evaluate(parse(case.conv_factor), {
            "s": case.source,
            "c1": case.center1,
            "c2": case.center2,
            "t": case.target,
            "norm": la.norm,
    })

    if not 0 <= case_conv_factor <= 1:
        raise ValueError(
            f"convergence factor not in valid range: {case_conv_factor}")

    from sumpy.expansion import VolumeTaylorExpansionFactory

    actx = actx_factory()
    ctx = t.ToyContext(actx.context,
             LaplaceKernel(dim),
             expansion_factory=VolumeTaylorExpansionFactory())

    errors = []

    src_pot = t.PointSources(ctx, src, weights=np.array([1.]))
    pot_actual = src_pot.eval(tgt).item()

    for order in ORDERS_P2E2E2P:
        expn = case.expansion1(src_pot, case.center1, order=order)
        expn2 = case.expansion2(expn, case.center2, order=order)
        pot_p2e2e2p = expn2.eval(tgt).item()
        errors.append(np.abs(pot_actual - pot_p2e2e2p))

    conv_factor = approx_convergence_factor(1 + np.array(ORDERS_P2E2E2P), errors)
    assert conv_factor <= min(1, case_conv_factor * (1 + RTOL_P2E2E2P)), \
        (conv_factor, case_conv_factor * (1 + RTOL_P2E2E2P))

# }}}


# {{{ test_cse_matvec

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

    rng = np.random.default_rng(42)
    vec = rng.random(2)
    expected_result = m @ vec
    actual_result = op.matvec(vec)
    assert np.allclose(expected_result, actual_result)

    vec = rng.random(4)
    expected_result = m.T @ vec
    actual_result = op.transpose_matvec(vec)
    assert np.allclose(expected_result, actual_result)

# }}}


# {{{ test_diff_op_stokes

def test_diff_op_stokes():
    from sumpy.symbolic import symbols, Function
    diff_op = make_identity_diff_op(3, 4)
    u = diff_op[:3]
    p = diff_op[3]
    pde = concat(laplacian(u) - gradient(p), divergence(u))

    actual_output = pde.to_sym()
    x, y, z = syms = symbols("x0, x1, x2")
    funcs = symbols("f0, f1, f2, f3", cls=Function)
    u, v, w, p = (f(*syms) for f in funcs)

    eq1 = u.diff(x, x) + u.diff(y, y) + u.diff(z, z) - p.diff(x)
    eq2 = v.diff(x, x) + v.diff(y, y) + v.diff(z, z) - p.diff(y)
    eq3 = w.diff(x, x) + w.diff(y, y) + w.diff(z, z) - p.diff(z)
    eq4 = u.diff(x) + v.diff(y) + w.diff(z)

    expected_output = [eq1, eq2, eq3, eq4]

    assert expected_output == actual_output

# }}}


# {{{ test_as_scalar_pde_stokes

def test_as_scalar_pde_stokes():
    diff_op = make_identity_diff_op(3, 4)
    u = diff_op[:3]
    p = diff_op[3]
    pde = concat(laplacian(u) - gradient(p), divergence(u))

    # velocity components in Stokes should satisfy Biharmonic
    for i in range(3):
        logger.info("pde\n%s", as_scalar_pde(pde, i))
        logger.info("\n%s", laplacian(laplacian(u[i])))
        assert as_scalar_pde(pde, i) == laplacian(laplacian(u[0]))

    # pressure should satisfy Laplace
    assert as_scalar_pde(pde, 3) == laplacian(u[0])

# }}}


# {{{ test_as_scalar_pde_maxwell

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
            laplacian(op[0]) - mu*epsilon*diff(diff(op[0], t), t)

# }}}


# {{{ test_as_scalar_pde_elasticity

def test_as_scalar_pde_elasticity():

    # Ref: https://doi.org/10.1006/jcph.1996.0102

    diff_op = make_identity_diff_op(2, 5)
    sigma_x = diff_op[0]
    sigma_y = diff_op[1]
    tau = diff_op[2]
    u = diff_op[3]
    v = diff_op[4]

    # Use numeric values as the expressions grow exponentially large otherwise
    from sumpy.symbolic import symbols
    lam, mu = symbols("lam, mu")

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
    assert pde.order == 1
    for i in range(5):
        scalar_pde = as_scalar_pde(pde, i)
        assert scalar_pde == laplacian(laplacian(diff_op[0]))
        assert scalar_pde.order == 4

# }}}


# {{{ test_elasticity_new

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

# }}}


# {{{ test_weird_kernel

w = make_identity_diff_op(2)

pdes = [
    diff(w, (1, 1)) + diff(w, (2, 0)),
    diff(w, (1, 1)) + diff(w, (0, 2)),
]


@pytest.mark.parametrize("pde", pdes)
def test_weird_kernel(pde):
    class MyKernel(ExpressionKernel):
        def __init__(self):
            super().__init__(dim=2, expression=1, global_scaling_const=1,
                is_complex_valued=False)

        def get_pde_as_diff_op(self):
            return pde

    from sumpy.expansion import LinearPDEConformingVolumeTaylorExpansion
    from operator import mul
    from functools import reduce

    knl = MyKernel()
    order = 10
    expn = LinearPDEConformingVolumeTaylorExpansion(kernel=knl,
            order=order, use_rscale=False)

    coeffs = expn.get_coefficient_identifiers()
    fft_size = reduce(mul, map(max, *coeffs), 1)

    assert fft_size == order

# }}}


# {{{ test_get_pde_operators

def test_get_pde_operators_laplace_biharmonic():
    dim = 3
    laplace = LaplaceKernel(dim)
    biharmonic = BiharmonicKernel(dim)
    id_op = make_identity_diff_op(dim, 1)

    op1, op2 = get_pde_operators([laplace, biharmonic], 2, {})
    assert op1 == laplacian(id_op) * sym.Rational(1, 2)
    assert op2 == id_op

    d_biharmonic = AxisTargetDerivative(1, biharmonic)
    op1, op2, op3 = get_pde_operators(
            [laplace, biharmonic, d_biharmonic], 2, {})
    assert op1 == laplacian(id_op) * sym.Rational(1, 2)
    assert op2 == id_op
    assert op3 == diff(id_op, [0, 1, 0][:dim])


def test_get_pde_operators_stokes():
    for dim in (2, 3):
        stokes00 = StokesletKernel(dim, 0, 0)
        stokes01 = StokesletKernel(dim, 0, 1)
        stokes11 = StokesletKernel(dim, 1, 1)

        id_op = make_identity_diff_op(dim, 1)

        op1, op2, op3 = get_pde_operators([stokes00, stokes01, stokes11], 2, {})

        assert op1 == laplacian(id_op) - diff(id_op, [2, 0, 0][:dim])
        assert op2 == -diff(id_op, [1, 1, 0][:dim])
        assert op3 == laplacian(id_op) - diff(id_op, [0, 2, 0][:dim])
# }}}


# You can test individual routines by typing
# $ python test_misc.py 'test_pde_check_kernels(_acf,
#       KernelInfo(HelmholtzKernel(2), k=5), order=5)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
