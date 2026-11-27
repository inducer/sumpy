from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import sumpy.symbolic as sym
from sumpy.derivative_recurrence import _polynomial_coeffs


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
.. autofunction:: recurrence
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


def recurrence(
        ode_coeffs: Sequence[sym.Expr],
) -> tuple[dict[tuple[int, int], sym.Expr], ...]:
    r"""
    Return the updates for :math:`U[i, j]`, :math:`0 \leq i \leq M`, :math:`j \ge 1`,
    using the order-:math:`(M+1)` ODE with coefficients :math:`[a_0, \dots, a_{M+1}]`.
    """
    q = sym.Symbol("q")
    q0, q1, q2 = sym.symbols("q0:3")
    tau = sym.Symbol("tau")
    j = sym.Symbol("j")
    M = len(ode_coeffs)-2  # ruff:ignore[non-lowercase-variable-in-function]

    q_tau = q0 + q1*tau + q2*tau**2
    dq_tau = q_tau.diff(tau)

    # ===== coefficient matching for the top state i = M =====
    #
    # a_{M+1}(q(tau)) * \partial_tau g^{(M)}(q(tau))
    #   = -\partial_tau q(tau) * \sum_{i=0}^{M} a_i(q(tau)) * g^{(i)}(q(tau)).
    #
    # Expand the LHS:
    # a_{M+1}(q(tau)) * \partial_tau g^{(M)}(q(tau))
    #   = a_{M+1}(q0) * \partial_tau g^{(M)}(q(tau))
    #     + (a_{M+1}(q(tau)) - a_{M+1}(q0)) * \partial_tau g^{(M)}(q(tau))
    #   = a_{M+1}(q0) * \sum_{j' >= 1} j' * U[M, j'] * tau^{j'-1}
    #     + (a_{M+1}(q(tau)) - a_{M+1}(q0))
    #         * \sum_{j' >= 1} j' * U[M, j'] * tau^{j'-1}.
    #
    # Expand the RHS:
    # -(q1 + 2*q2*tau) * \sum_{i=0}^{M} (
    #     a_i(q(tau)) * \sum_{j' >= 0} U[i, j'] * tau^{j'}).
    #
    # Then, we match the coefficients for [tau^{j-1}] on both sides.

    leading_coeff_at_q0 = ode_coeffs[-1].subs(q, q0)
    leading_coeffs = _polynomial_coeffs(
        ode_coeffs[-1].subs(q, q_tau), sym.Matrix([tau]))

    # [tau^{j-1}]: a_{M+1}(q0) * j * U[M, j]
    #   = -\sum_{(i, shift)} rest[i, shift] * U[i, j+shift].
    # U[M, j] couples with the top state and lower states.

    rest = defaultdict(lambda: 0)
    # Coupling with the top state.
    for (k,), weight in leading_coeffs.items():
        if k != 0:
            # [tau^{j-1}] tau^k * \sum_{j' >= 1} j' * U[M, j'] * tau^{j'-1}
            #   = (j-k) * U[M, j-k].
            rest[M, -k] += weight*(j-k)

    for i, coeff in enumerate(ode_coeffs[:-1]):
        lower_coeffs = _polynomial_coeffs(
            dq_tau*coeff.subs(q, q_tau), sym.Matrix([tau]))
        for (k,), weight in lower_coeffs.items():
            # [tau^{j-1}] tau^k * \sum_{j' >= 0} U[i, j'] * tau^{j'}
            #   = U[i, j-1-k].
            rest[i, -1-k] += weight

    top_state_update = {key: -weight/(leading_coeff_at_q0*j)
                        for key, weight in rest.items() if weight != 0}

    # ===== coefficient matching for the lower states, i < M =====
    #
    # \partial_tau g^{(i)}(q(tau))
    #   = g^{(i+1)}(q(tau)) * \partial_tau q(tau)
    #   = (q1 + 2*q2*tau) * g^{(i+1)}(q(tau)).
    #
    # Matching the coefficients for [tau^{j-1}] on both sides:
    # LHS: j * U[i, j],
    # RHS: q1 * U[i+1, j-1] + 2*q2 * U[i+1, j-2].
    # Then,
    #   U[i, j] = (q1 * U[i+1, j-1] + 2*q2 * U[i+1, j-2]) / j.

    lower_state_updates = [{(i+1, -1): q1/j, (i+1, -2): 2*q2/j} for i in range(M)]
    return (*lower_state_updates, top_state_update)
