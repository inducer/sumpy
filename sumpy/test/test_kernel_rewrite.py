from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
"""

import logging

import numpy as np
import pytest

import sumpy.symbolic as sym
from sumpy.kernel import (
    BiharmonicKernel,
    LaplaceKernel,
    StokesletComponentKernel,
    StressletComponentKernel,
)
from sumpy.kernel_rewrite import (
    LinearOperatorRepresentation,
    rewrite_using_base_kernel_fourier,
    rewrite_using_base_kernel_lu,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mi_derivative(expr: sym.Expr, x: sym.Matrix, mi: tuple[int, ...]) -> sym.Expr:
    result = expr
    for i, n in enumerate(mi):
        result = result.diff(x[i], n)

    return result


def check_kernel_rewrite(op: LinearOperatorRepresentation) -> None:
    from sumpy.kernel_rewrite import evalf, simplify
    from sumpy.symbolic import PymbolicToSympyMapperWithSymbols

    dim = op.target_kernel.dim
    dvec = sym.make_sym_vector("d", dim)
    to_sympy = PymbolicToSympyMapperWithSymbols()

    target_expr = (
        op.target_kernel.get_global_scaling_const()
        * op.target_kernel.get_expression(dvec))
    base_expr = (
        op.base_kernel.get_global_scaling_const()
        * op.base_kernel.get_expression(dvec))

    expr = to_sympy(op.coeffs[0]) + sum((
        to_sympy(c) * mi_derivative(base_expr, dvec, mi)
        for c, mi in zip(op.coeffs[1:], op.mis, strict=True)
    ), sym.Integer(0))

    result = evalf(simplify(target_expr - expr))
    assert abs(result) < 3.0 * 1.0e-16


# {{{ test_rewrite_using_base_kernel_lu_conditioning


@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_lu_conditioning(dim: int) -> None:
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
    assert np.max(kappa) < 2.0e+5

# }}}


# {{{ test_rewrite_using_base_kernel_lu_laplace_biharmonic

@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_lu_laplace_biharmonic(dim: int) -> None:
    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    target_kernel = LaplaceKernel(dim)
    result = rewrite_using_base_kernel_lu(target_kernel, base_kernel, rng=rng)

    print(result.pretty())
    check_kernel_rewrite(result)

# }}}


# {{{ test_rewrite_using_base_kernel_lu_stokeslet_biharmonic

@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_lu_stokeslet_biharmonic(dim: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    for i, j in gnitb(dim, 2):
        target_kernel = StokesletComponentKernel(dim, i, j, viscosity_mu_name="mu")
        result = rewrite_using_base_kernel_lu(target_kernel, base_kernel, rng=rng)
        print(result.pretty())
        check_kernel_rewrite(result)


# }}}


# {{{ test_rewrite_using_base_kernel_lu_stresslet_biharmonic

@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_lu_stresslet_biharmonic(dim: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    for i, j, k in gnitb(dim, 3):
        target_kernel = StressletComponentKernel(dim, i, j, k, viscosity_mu_name="mu")
        result = rewrite_using_base_kernel_lu(target_kernel, base_kernel, rng=rng)
        print(result.pretty())
        check_kernel_rewrite(result)


# }}}


# {{{ test_rewrite_using_base_kernel_fourier_laplace_biharmonic


@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_fourier_laplace_biharmonic(dim: int) -> None:
    """Test that the Fourier-based algorithm recovers Laplace from biharmonic."""
    base_kernel = BiharmonicKernel(dim)
    target_kernel = LaplaceKernel(dim)
    result = rewrite_using_base_kernel_fourier(target_kernel, base_kernel)

    logger.info(result.pretty())
    check_kernel_rewrite(result)


# }}}


# {{{ test_rewrite_using_base_kernel_fourier_stokeslet_biharmonic


@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_fourier_stokeslet_biharmonic(dim: int) -> None:
    """Test that the Fourier-based algorithm recovers the Stokeslet from biharmonic."""
    from itertools import product

    base_kernel = BiharmonicKernel(dim)

    for i, j in product(range(dim), repeat=2):
        target_kernel = StokesletComponentKernel(dim, i, j, viscosity_mu_name="mu")
        result = rewrite_using_base_kernel_fourier(target_kernel, base_kernel)

    logger.info(result.pretty())
    check_kernel_rewrite(result)

# }}}


# {{{ test_rewrite_using_base_kernel_fourier_indivisible


@pytest.mark.parametrize("dim", [2, 3])
def test_rewrite_using_base_kernel_fourier_biharmonic_laplace(dim: int) -> None:
    from sumpy.kernel_rewrite import (
        RewriteFailedError,
        rewrite_using_base_kernel_fourier,
    )

    # Laplace Fourier symbol (-|k|^2) is not divisible by biharmonic (|k|^4)
    base_kernel = LaplaceKernel(dim)
    target_kernel = BiharmonicKernel(dim)

    with pytest.raises(RewriteFailedError, match="cannot rewrite"):
        rewrite_using_base_kernel_fourier(target_kernel, base_kernel)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
