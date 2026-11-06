from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import sumpy.symbolic as sym


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sumpy.kernel import ScalarKernel

__doc__ = r"""
Kernel ODE
==========

Derive an ODE from the PDE's kernel :math:`G(z)`.

Notation:

- :math:`q = |z|^2` and :math:`g^{(i)}(q) = \frac{d^i g(q)}{dq^i}`.
- The ODE is :math:`\sum_i a_i(q) g^{(i)}(q) = 0`, with ascending coefficients
  :math:`[a_0, \dots, a_n]`.

Assumptions:

- The kernel has the form :math:`G(z) = g(q)`.
- Its PDE is linear with polynomial coefficients in :math:`z`.
- The derived ODE's coefficients are polynomials in :math:`q`.

.. autofunction:: kernel_ode
.. autofunction:: chain_rule_weights
.. autofunction:: pde_to_ode
"""


def chain_rule_weights(
        mi: tuple[int, ...], z: sym.Matrix,
) -> dict[int, sym.Expr]:
    r"""
    Return the chain-rule weights :math:`w_i(z)` of

    .. math::

        \partial_z^{\mathrm{mi}} G(z) = \sum_i w_i(z) g^{(i)}(q)

    as ``{i: w_i(z)}``. For example, in 2D,
    :math:`\partial_z^{(2,0)} G(z) = 2 g^{(1)}(q) + 4 z_0^2 g^{(2)}(q)`,
    so :math:`w_1 = 2` and :math:`w_2 = 4 z_0^2`.
    """
    # \partial_{z_k}(w_i * g^{(i)}(q))
    #   = (\partial_{z_k} w_i) * g^{(i)}(q)
    #     + 2*z_k*w_i * g^{(i+1)}(q).
    weights = {0: sym.Integer(1)}
    for axis, degree in enumerate(mi):
        for _ in range(degree):
            new_weights = defaultdict(lambda: 0)
            for i, weight in weights.items():
                # Differentiate w_i(z).
                weight_derivative = weight.diff(z[axis])
                new_weights[i] += weight_derivative
                # Apply the chain rule to g^{(i)}(q).
                new_weights[i+1] += 2*z[axis]*weight
            weights = new_weights
    return weights


def _primitive(coeffs: list[sym.Expr]) -> tuple[sym.Expr, ...]:
    r"""
    Return the ODE coefficients :math:`a_i(q)` without trailing zeros,
    common factor or numeric factor in the leading coefficient.
    """
    # SymEngine has no polynomial gcd.
    import sympy as sp

    q = sp.Symbol("q")
    # Drop trailing zeros: [0, 2*q, 0] -> [0, 2*q].
    while coeffs[-1] == 0:
        coeffs.pop()
    # Remove the common factor: [0, 6, 4*q] -> [0, 3, 2*q].
    common_factor = sp.gcd_list(coeffs)
    ode_coeffs = []
    for coeff in coeffs:
        ode_coeffs.append(sp.cancel(coeff/common_factor))
    # Remove the leading coefficient's numeric factor.
    # [1, 0, 3*k*q] -> [1/3, 0, k*q].
    leading_coeff = sp.Poly(ode_coeffs[-1], q).LC()
    numeric_factor = leading_coeff.as_coeff_Mul()[0]
    for i, coeff in enumerate(ode_coeffs):
        coeff = sp.expand(coeff/numeric_factor)
        ode_coeffs[i] = sym.sympify(coeff)
    return tuple(ode_coeffs)


def pde_to_ode(
        pde_coeffs: Mapping[tuple[int, ...], sym.Expr | int],
        z: sym.Matrix,
) -> tuple[sym.Expr, ...]:
    r"""
    Return the ODE coefficients :math:`[a_0, \dots, a_n]` from the
    given PDE.
    """
    # SymEngine has no polynomial remainder.
    import sympy as sp

    pde_order = max(sum(mi) for mi in pde_coeffs)
    ode_coeffs: list[sym.Expr] = [sym.Integer(0)]*(pde_order+1)
    for mi, pde_coeff in pde_coeffs.items():
        for i, weight in chain_rule_weights(mi, z).items():
            ode_coeffs[i] += pde_coeff*weight
    # TODO: support ODEs whose coefficients depend on direction.
    q = sp.Symbol("q")
    z = sp.sympify(z)
    for i, coeff in enumerate(ode_coeffs):
        coeff = sp.rem(coeff, z.dot(z)-q, z[0])
        if coeff.has(*z):
            raise NotImplementedError("the ode in q depends on direction")
        ode_coeffs[i] = coeff
    return _primitive(ode_coeffs)


def kernel_ode(
        kernel: ScalarKernel, q0: sym.Expr,
) -> tuple[tuple[sym.Expr, ...], tuple[sym.Expr, ...]]:
    r"""
    Return the kernel's ODE coefficients :math:`[a_0, \dots, a_n]` and the
    initial conditions :math:`g^{(i)}(q_0)` for :math:`i = 0, \dots, n-1`.
    """
    q = sym.Symbol("q")
    # Check that the kernel is of the form G(z) = g(|z|^2).
    axis = sym.Matrix([sym.sqrt(q)] + [0]*(kernel.dim-1))
    g = kernel.get_expression(axis)
    z = sym.Matrix(sym.symbols(f"z0:{kernel.dim}"))
    kernel_expr = kernel.get_expression(z)
    if (kernel_expr-g.subs(q, z.dot(z))).expand() != 0:
        raise NotImplementedError("kernel is not G(z) = g(|z|^2)")

    pde_coeffs = {}
    pde_eq, = kernel.get_pde_as_diff_op().eqs
    for ident, coeff in pde_eq.items():
        pde_coeffs[ident.mi] = coeff
    ode_coeffs = pde_to_ode(pde_coeffs, z)
    ode_order = len(ode_coeffs)-1

    init_conds = []
    for i in range(ode_order):
        g_derivative = g.diff(q, i)
        init_conds.append(g_derivative.subs(q, q0))
    return ode_coeffs, tuple(init_conds)
