from __future__ import annotations

from typing import TYPE_CHECKING

import sumpy.symbolic as sym


if TYPE_CHECKING:
    from collections.abc import Sequence

__doc__ = r"""
Line Recurrence
===============

Use the kernel ODE to derive Taylor recurrences along a line.

Notation:

- The kernel function is :math:`G(z) = g(q)`, where :math:`q = |z|^2`.
  The function :math:`g` satisfies the ODE
  :math:`\sum_{i=0}^{n} a_i(q)g^{(i)}(q) = 0`.
- Define :math:`U_i(q(\tau)) := \left.\frac{d^i g(q)}{dq^i}\right|_{q=q(\tau)}`,
  where :math:`q(\tau) = q_0 + q_1\tau + q_2\tau^2`.
  The :math:`j`-th Taylor coefficient of :math:`U_i(q(\tau))` is
  :math:`U[i, j] = [\tau^j]g^{(i)}(q(\tau))`.

Assumptions:

- The ODE has order :math:`n \ge 1` and polynomial coefficients in :math:`q`.
- :math:`a_n(q_0)` is nonzero.

.. autofunction:: ode_differentiation
"""


def ode_differentiation(
        ode_coeffs: Sequence[sym.Expr], init_conds: Sequence[sym.Expr],
        max_derivative_order: int,
) -> tuple[tuple[sym.Expr, ...], tuple[sym.Expr, ...]]:
    r"""
    Given an ODE of order :math:`n` = ``len(ode_coeffs) - 1`` with its
    initial conditions.
    Define :math:`M = \max(n-1, \mathtt{max\_derivative\_order})`.
    It returns the coefficients of an order-:math:`(M+1)` ODE and its
    initial conditions.
    """
    ode_coeffs = tuple(ode_coeffs)
    ode_order = len(ode_coeffs)-1
    if max_derivative_order < ode_order:
        return ode_coeffs, tuple(init_conds)

    q = sym.Symbol("q")
    q0 = sym.Symbol("q0")
    init_conds = list(init_conds)
    leading_coeff_at_q0 = ode_coeffs[-1].subs(q, q0)
    for _ in range(ode_order, max_derivative_order+1):
        g_derivative = (
            -sym.Add(*(
                coeff.subs(q, q0)*init_cond
                for coeff, init_cond in zip(ode_coeffs[:-1], init_conds, strict=True)))
            / leading_coeff_at_q0)
        init_conds.append(g_derivative)

        # ODE coefficients after one differentiation:
        # [0, a_0, a_1, a_2] + d/dq [a_0, a_1, a_2, 0]
        #     = [a_0', a_0 + a_1', a_1 + a_2', a_2].
        ode_coeffs = tuple(
            coeff.diff(q) + previous
            for coeff, previous in zip(
                (*ode_coeffs, sym.Integer(0)), (0, *ode_coeffs), strict=True))
    return ode_coeffs, tuple(init_conds)
