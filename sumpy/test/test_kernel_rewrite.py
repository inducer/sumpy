from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
"""

import logging

import numpy as np
import pytest

import sumpy.symbolic as sym
from sumpy.kernel import BiharmonicKernel, LaplaceKernel, StokesletComponentKernel
from sumpy.kernel_rewrite import (
    LinearOperatorRepresentation,
    rewrite_using_base_kernel_fourier,
    rewrite_using_base_kernel_lu,
)


logger = logging.getLogger(__name__)


def mi_derivative(expr: sym.Expr, x: sym.Matrix, mi: tuple[int, ...]) -> sym.Expr:
    result = expr
    for i, n in enumerate(mi):
        result = result.diff(x[i], n)

    return result


def check_kernel_rewrite(op: LinearOperatorRepresentation) -> None:
    from sumpy.kernel_rewrite import evalf

    dim = op.target_kernel.dim
    dvec = sym.make_sym_vector("d", dim)

    target_expr = (
        op.target_kernel.get_global_scaling_const()
        * op.target_kernel.get_expression(dvec))
    base_expr = (
        op.base_kernel.get_global_scaling_const()
        * op.base_kernel.get_expression(dvec))

    expr = sym.to_sympy(op.coeffs[0]) + sum((
        sym.to_sympy(c) * mi_derivative(base_expr, dvec, mi)
        for c, mi in zip(op.coeffs[1:], op.mis, strict=True)
    ), sym.Integer(0))

    result = evalf(sym.simplify(target_expr - expr))
    assert abs(result) < 3.0 * 1.0e-16


# {{{ test_rewrite_using_base_kernel_lu

@pytest.mark.parametrize("dim", [2, 3])
def test_laplace_biharmonic_rewrite(dim: int) -> None:
    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    target_kernel = LaplaceKernel(dim)
    result = rewrite_using_base_kernel_lu(target_kernel, base_kernel, rng=rng)

    logger.info(result.pretty())
    check_kernel_rewrite(result)


@pytest.mark.skip()
@pytest.mark.parametrize("dim", [2, 3])
def test_stokeslet_biharmonic_rewrite(dim: int) -> None:
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    rng = np.random.default_rng(seed=42)

    base_kernel = BiharmonicKernel(dim)
    for i, j in gnitb(dim, 2):
        if not i <= j:
            continue

        target_kernel = StokesletComponentKernel(dim, i, j, viscosity_mu_name="mu")
        result = rewrite_using_base_kernel_lu(target_kernel, base_kernel, rng=rng)
        print(result.pretty())


@pytest.mark.skip()
@pytest.mark.parametrize("dim", [2, 3])
def test_stresslet_biharmonic_rewrite(dim: int) -> None:
    pass


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


# {{{ test_stokeslet_biharmonic_fourier_rewrite


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
def test_rewrite_using_base_kernel_fourier_indivisible(dim: int) -> None:
    """Test that a ValueError is raised when P_base is not divisible by P_target."""
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
