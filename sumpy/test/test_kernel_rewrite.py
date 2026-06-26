from __future__ import annotations


__copyright__ = """
Copyright (C) 2020 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
"""

import logging

import numpy as np
import pytest

import sumpy.symbolic as sym
from sumpy.kernel import BiharmonicKernel, LaplaceKernel, StokesletKernel
from sumpy.kernel_rewrite import (
    LinearOperatorRepresentation,
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

        target_kernel = StokesletKernel(dim, i, j, viscosity_mu=1)
        result = rewrite_using_base_kernel_lu(target_kernel, base_kernel, rng=rng)
        print(result.pretty())


@pytest.mark.skip()
@pytest.mark.parametrize("dim", [2, 3])
def test_stresslet_biharmonic_rewrite(dim: int) -> None:
    pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
