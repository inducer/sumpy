from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
"""

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

import numpy as np
import pytest

from pytools import ProcessTimer

import sumpy.symbolic as sym
from sumpy.kernel import (
    AxisSourceDerivative,
    AxisTargetDerivative,
    BiharmonicKernel,
    BrinkmanletComponentKernel,
    ElasticityComponentKernel,
    HelmholtzKernel,
    LaplaceKernel,
    ScalarKernel,
    StokesletComponentKernel,
    StressletComponentKernel,
    YukawaKernel,
)
from sumpy.kernel_rewrite import (
    LinearOperatorRepresentation,
    RewriteFailedError,
    rewrite_using_base_kernel,
)


logger = logging.getLogger(__name__)


def check_kernel_rewrite(op: LinearOperatorRepresentation) -> None:
    from sumpy.kernel_rewrite import evalf, simplify
    from sumpy.tools import mi_derivative

    dim = op.kernel.dim
    dvec = sym.make_sym_vector("d", dim)

    const = sym.to_symbolic(op.const, symbols=True)
    assert not (const.free_symbols & set(dvec)), f"constant term depends on d: {const}"

    # apply any source/target derivatives to get the actual target expression
    kernel = op.kernel
    derivatives: list[tuple[int, int]] = []
    while isinstance(kernel, (AxisTargetDerivative, AxisSourceDerivative)):
        sign = 1 if isinstance(kernel, AxisTargetDerivative) else -1
        derivatives.append((sign, kernel.axis))
        kernel = kernel.inner_kernel

    kernel_expr = (
        kernel.get_global_scaling_const()
        * kernel.get_expression(dvec))
    for sign, axis in derivatives:
        kernel_expr = sign * kernel_expr.diff(dvec[axis])

    base_expr = (
        op.base_kernel.get_global_scaling_const()
        * op.base_kernel.get_expression(dvec))

    expr = sym.to_symbolic(op.const, symbols=True) + sum((
        sym.to_symbolic(c, symbols=True) * mi_derivative(base_expr, dvec, mi)
        for c, mi in zip(op.coeffs, op.mis, strict=True)
    ), sym.Integer(0))

    result = evalf(simplify(kernel_expr - expr))
    assert abs(result) < 3.0 * 1.0e-16, f"result is not zero: {result!r}"


def make_random_derivatives(
        kernel: ScalarKernel, nderivs: int, *, rng: np.random.Generator,
    ) -> ScalarKernel:
    for _ in range(nderivs):
        axis = int(rng.integers(kernel.dim))
        if rng.random() < 0.5:
            kernel = AxisTargetDerivative(axis, kernel)
        else:
            kernel = AxisSourceDerivative(axis, kernel)

    return kernel


def check_rewrite_all_methods(
        kernel: ScalarKernel,
        base_kernel: ScalarKernel,
        rng: np.random.Generator,
        *,
        nderivs: int = 0,
    ) -> None:
    for method in ("fourier", "lu"):
        with ProcessTimer() as pt:
            result = rewrite_using_base_kernel(
                kernel, base_kernel, method=method, rng=rng)
        logger.info("[%7s] %s (%s)", method, result.pretty(), pt)

        assert result.kernel is kernel
        assert len(result.coeffs) == len(result.mis)
        if nderivs > 0:
            assert result.const == 0

        check_kernel_rewrite(result)


# {{{ test_rewrite_using_base_kernel_lu_conditioning


@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_lu_conditioning(dim: int) -> None:
    """Numerically evaluates the matrix formed in :func:`rewrite_using_base_kernel_lu`
    to check the conditioning for various point clouds.
    """

    from pytools import (
        generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
    )

    from sumpy.kernel_rewrite import (
        _generate_points_shells,
        _make_derivative_matrix,
        _make_expr_derivatives,
    )

    rng = np.random.default_rng(42)
    base_kernel = BiharmonicKernel(dim)

    dvec = sym.make_sym_vector("d", dim)
    base_expr = base_kernel.get_expression(dvec)

    pde = base_kernel.get_pde_as_diff_op()
    mis = list(gnitstam(pde.order, dim))
    pde_mis = [ident.mi for eq in pde.eqs for ident in eq]
    pde_mis = [mi for mi in pde_mis if sum(mi) == pde.order]
    mis.remove(pde_mis[-1])

    mi_to_derivative = _make_expr_derivatives(base_expr, dvec, mis)

    nruns = 16
    kappa = np.empty(nruns)

    for i in range(nruns):
        points = _generate_points_shells(dim, len(mis) + 1, rng=rng)
        mat = _make_derivative_matrix(points, dvec, mis, mi_to_derivative)

        mat = np.array([
            [float(mat[i, j]) for j in range(mat.shape[1])]
            for i in range(mat.shape[0])
        ])

        kappa[i] = np.linalg.cond(mat)
        logger.info("kappa = %.8e", kappa[i])

    logger.info("median: %.8e max %.8e", np.median(kappa), np.max(kappa))
    assert np.max(kappa) < 3.0e+5

# }}}


# {{{ test_rewrite_using_base_kernel_laplace_biharmonic

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nderivs", [0, 2])
def test_rewrite_using_base_kernel_laplace_biharmonic(
        dim: int, nderivs: int) -> None:
    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    kernel = make_random_derivatives(LaplaceKernel(dim), nderivs, rng=rng)
    check_rewrite_all_methods(kernel, base_kernel, rng, nderivs=nderivs)

# }}}


# {{{ test_rewrite_using_base_kernel_stokeslet_biharmonic

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nderivs", [0, 2])
def test_rewrite_using_base_kernel_stokeslet_biharmonic(
        dim: int, nderivs: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    for i, j in gnitb(dim, 2):
        # Biharmonic -> works
        base_kernel = BiharmonicKernel(dim)
        kernel = make_random_derivatives(
            StokesletComponentKernel(dim, i, j, viscosity_mu_name="mu"),
            nderivs, rng=rng)
        check_rewrite_all_methods(kernel, base_kernel, rng, nderivs=nderivs)

        # Laplace -> fails
        base_kernel = LaplaceKernel(dim)
        kernel = make_random_derivatives(
            StokesletComponentKernel(dim, i, j, viscosity_mu_name="mu"),
            nderivs, rng=rng)

        with pytest.raises(RewriteFailedError):
            _ = rewrite_using_base_kernel(
                kernel, base_kernel, method="fourier", rng=rng
            )

# }}}


# {{{ test_rewrite_using_base_kernel_stresslet_biharmonic

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nderivs", [0, 2])
def test_rewrite_using_base_kernel_stresslet_biharmonic(
        dim: int, nderivs: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    for i, j, k in gnitb(dim, 3):
        kernel = make_random_derivatives(
            StressletComponentKernel(dim, i, j, k, viscosity_mu_name="mu"),
            nderivs, rng=rng)

        with pytest.raises(RewriteFailedError):
            _ = rewrite_using_base_kernel(
                kernel, base_kernel, method="fourier", rng=rng)

        with ProcessTimer() as pt:
            result = rewrite_using_base_kernel(
                kernel, base_kernel, method="lu", rng=rng)
        logger.info("[%7s] %s (%s)", "lu", result.pretty(), pt)

        assert result.kernel is kernel
        assert len(result.coeffs) == len(result.mis)
        if nderivs > 0:
            assert result.const == 0

        check_kernel_rewrite(result)

# }}}


# {{{ test_rewrite_using_base_kernel_brinkman_yukawa

# NOTE: this would also work if we define a combined kernel
#   K = (Yukakwa + Laplace) / k^2

@pytest.mark.skip(reason="needs two bases (Laplace and Yukawa)")
@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_brinkman_yukawa(dim: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    base_kernel = YukawaKernel(dim, yukawa_lambda_name="k")
    for i, j in gnitb(dim, 2):
        kernel = BrinkmanletComponentKernel(
            dim, i, j, viscosity_mu_name="mu", darcy_impermeability_name="k")

        check_rewrite_all_methods(kernel, base_kernel, rng)

# }}}


# {{{ test_rewrite_using_base_kernel_elasticity_biharmonic


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("nderivs", [0])
def test_rewrite_using_base_kernel_elasticity_biharmonic(
        dim: int, nderivs: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    for i, j in gnitb(dim, 2):
        kernel = make_random_derivatives(
            ElasticityComponentKernel(
                dim, i, j, viscosity_mu_name="mu", poisson_ratio_name="nu"),
            nderivs, rng=rng)
        check_rewrite_all_methods(kernel, base_kernel, rng, nderivs=nderivs)

# }}}


# {{{ test_rewrite_using_base_kernel_random_derivatives


@pytest.mark.parametrize("base_kernel", [
    BiharmonicKernel(2),
    BiharmonicKernel(3),
    YukawaKernel(2),
    # YukawaKernel(3),
    HelmholtzKernel(2),
    # HelmholtzKernel(3),
])
@pytest.mark.parametrize("nderivs", [2, 3, 4])
def test_rewrite_using_base_kernel_random_derivatives(
        base_kernel: ScalarKernel, nderivs: int
    ) -> None:
    rng = np.random.default_rng(seed=42)

    kernel = make_random_derivatives(base_kernel, nderivs, rng=rng)
    check_rewrite_all_methods(kernel, base_kernel, rng, nderivs=nderivs)

# }}}


# {{{ test_rewrite_using_base_kernel_biharmonic_laplace


@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_biharmonic_laplace(dim: int) -> None:
    from sumpy.kernel_rewrite import RewriteFailedError

    rng = np.random.default_rng(seed=42)

    # Laplace Fourier symbol (-|k|^2) is not divisible by biharmonic (|k|^4)
    base_kernel = LaplaceKernel(dim)
    kernel = BiharmonicKernel(dim)

    with pytest.raises(RewriteFailedError, match="cannot rewrite"):
        rewrite_using_base_kernel(
            kernel, base_kernel, method="lu", rng=rng)

    with pytest.raises(RewriteFailedError, match="cannot rewrite"):
        rewrite_using_base_kernel(kernel, base_kernel, method="fourier")


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
