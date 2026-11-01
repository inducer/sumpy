from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sumpy.kernel import ScalarKernel

__doc__ = r"""Derive a kernel ODE from its PDE.

Assumptions:

- Linear PDE with polynomial coefficients in :math:`z`.
- The scalar kernel is :math:`G(z) = g(q)`, with :math:`q = |z|^2`.
- The ODE's coefficients are polynomials in :math:`q`.

TODO: support ODE whose coefficients may have coordinate dependence.

.. autofunction:: kernel_ode
.. autofunction:: chain_rule_weights
.. autofunction:: pde_to_ode
"""

# ODE variable q = |z|^2.
q = sp.Symbol("q")


def chain_rule_weights(
        mi: tuple[int, ...], z: sp.Matrix,
) -> dict[int, sp.Expr]:
    r"""
    Return the chain-rule weights :math:`\{i: w_i(z)\}` of
    :math:`\partial_z^{\mathrm{mi}} G(z) = \sum_i w_i(z) g^{(i)}(q)`.

    For example, in 2D,
    :math:`\partial_z^{(2,0)} G(z) = 2g^{(1)}(q) + 4z_0^2 g^{(2)}(q)`,
    so :math:`w_1(z)=2` and :math:`w_2(z)=4z_0^2`.
    """
    g = sp.Function("g")
    r2 = z.dot(z)
    derivative = sp.diff(
        g(r2), *((z[k], count) for k, count in enumerate(mi))).expand()
    # w_i(z) is the chain rule weight of g^(i)(|z|^2).
    weights = {}
    for i in range(sum(mi)+1):
        g_derivative = sp.diff(g(q), q, i).subs(q, r2)
        weight = derivative.coeff(g_derivative)
        if weight != 0:
            weights[i] = weight
    return weights


def primitive(coeffs: Sequence[sp.Expr | int]) -> list[sp.Expr]:
    r"""
    Return the normalized polynomial ODE coefficients :math:`a_i(q)`.
    """
    coeffs = list(coeffs)
    # Drop trailing zeros: [0, 2*q, 0] -> [0, 2*q].
    while coeffs[-1] == 0:
        coeffs.pop()
    # Remove the common factor: [0, 6, 4*q] -> [0, 3, 2*q].
    common_factor = sp.gcd_list(coeffs)
    coeffs = [sp.cancel(coeff / common_factor) for coeff in coeffs]
    # Remove the leading coefficient's numeric factor.
    # [1, 0, 3*k*q] -> [1/3, 0, k*q].
    leading_coeff = sp.Poly(coeffs[-1], q).LC()
    numeric_factor = leading_coeff.as_coeff_Mul()[0]
    return [sp.expand(coeff / numeric_factor) for coeff in coeffs]


def pde_to_ode(
        pde_coeffs: Mapping[tuple[int, ...], sp.Expr | int],
        z: sp.Matrix,
) -> list[sp.Expr]:
    r"""
    Return the ODE :math:`\sum_i a_i(q)g^{(i)}(q)=0` from the PDE.
    """
    r2 = z.dot(z)
    pde_order = max(sum(mi) for mi in pde_coeffs)
    ode_coeffs: list[sp.Expr] = [sp.Integer(0)]*(pde_order + 1)
    for mi, pde_coeff in pde_coeffs.items():
        for i, weight in chain_rule_weights(mi, z).items():
            ode_coeffs[i] += pde_coeff*weight

    ode_coeffs = [sp.rem(coeff, r2 - q, z[0]) for coeff in ode_coeffs]
    if any(coeff.has(*z) for coeff in ode_coeffs):
        raise NotImplementedError("the ODE in q depends on direction")
    if all(coeff == 0 for coeff in ode_coeffs):
        raise ValueError("every g(q) satisfies this PDE")
    return primitive(ode_coeffs)


def kernel_ode(
        kernel: ScalarKernel, q0: sp.Expr,
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    r"""
    Return the kernel ODE :math:`\sum_i a_i(q)g^{(i)}(q)=0`
    with initial conditions :math:`g^{(i)}(q_0)`.
    """
    # Check that the kernel is of the form G(z) = g(|z|^2).
    axis = sp.Matrix([sp.sqrt(q)] + [0]*(kernel.dim-1))
    g = kernel.get_expression(axis)
    z = sp.Matrix(sp.symbols(f"z0:{kernel.dim}"))
    if kernel.get_expression(z) != g.subs(q, z.dot(z)):
        raise NotImplementedError("kernel is not G(z) = g(|z|^2)")

    # Derive the ODE from the PDE.
    pde_coeffs = {}
    diff_op = kernel.get_pde_as_diff_op()
    for ident, coeff in diff_op.eqs[0].items():
        pde_coeffs[ident.mi] = coeff
    ode_coeffs = pde_to_ode(pde_coeffs, z)
    ode_order = len(ode_coeffs)-1

    init_conds = []
    for i in range(ode_order):
        g_derivative = sp.diff(g, q, i)
        init_conds.append(g_derivative.subs(q, q0))
    return ode_coeffs, init_conds
