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

Derive recurrences for Taylor coefficients along a line.

Notation:

- The kernel is :math:`G(z) = g(q)`, with :math:`q = |z|^2`.
- Along a line, :math:`q(\tau) = q_0 + q_1\tau + q_2\tau^2`.
- :math:`U[i, j] = [\tau^j]g^{(i)}(q(\tau))`, where :math:`i` is the
  :math:`q`-derivative order and :math:`j` is the Taylor degree.
- The ODE is :math:`\sum_{i=0}^{\mathrm{ode\_order}} a_i(q)g^{(i)}(q) = 0`,
  with ascending coefficients :math:`[a_0, \dots, a_{\mathrm{ode\_order}}]`.

Assumptions:

- The ODE has positive order and polynomial coefficients in :math:`q`.
- The ODE's leading coefficient is nonzero at :math:`q_0`.

.. autofunction:: recurrence
"""


def _ode_differentiation(
        ode_coeffs: Sequence[sym.Expr], order: int,
) -> tuple[sym.Expr, ...]:
    r"""
    Return the coefficients after differentiating the ODE in :math:`q`
    until its order is at least ``order``.
    """
    q = sym.Symbol("q")
    ode_coeffs = tuple(ode_coeffs)
    # [0, a_0, a_1, a_2] + d/dq [a_0, a_1, a_2, 0]
    #     = [a_0', a_0 + a_1', a_1 + a_2', a_2].
    for _ in range(len(ode_coeffs)-1, order):
        ode_coeffs = tuple(
            (coeff.diff(q) + previous).expand()
            for coeff, previous in zip(
                (*ode_coeffs, sym.Integer(0)), (0, *ode_coeffs), strict=True))
    return ode_coeffs


def recurrence(
        ode_coeffs: Sequence[sym.Expr],
        max_derivative_order: int,
        *, init_conds: Sequence[sym.Expr],
        q0: sym.Expr, q1: sym.Expr, q2: sym.Expr,
) -> tuple[tuple[sym.Expr, ...], tuple[dict[tuple[int, int], sym.Expr], ...]]:
    r"""
    Return extended initial conditions and the update formulas for
    :math:`U[i, j] = [\tau^j]g^{(i)}(q(\tau))` with :math:`j \ge 1`.

    Let :math:`M = \max(\mathrm{max\_derivative\_order}, \mathrm{ode\_order}-1)`.
    For :math:`0 \le i < M`, the chain rule gives

    .. math::

        U[i, j] = \frac{q_1 U[i+1, j-1] + 2 q_2 U[i+1, j-2]}{j}.

    For :math:`i = M`, the ODE gives

    .. math::

        U[M, j] = \sum_{(i',\mathrm{shift})} c_{i',\mathrm{shift}}(j)
            U[i', j+\mathrm{shift}],

    where :math:`\mathrm{shift} < 0`. Each formula is returned as
    ``{(i', shift): weight}``, with weights in the symbol ``j``.
    """
    q = sym.Symbol("q")
    tau = sym.Symbol("tau")
    j = sym.Symbol("j")
    ode_order = len(ode_coeffs)-1
    M = max(max_derivative_order, ode_order-1)  # ruff:ignore[non-lowercase-variable-in-function]

    init_conds = list(init_conds)
    leading_coeff_at_q0 = ode_coeffs[-1].subs(q, q0)
    # For i >= ode_order, we use the initial conditions
    # {g^(0)(q0), g^(1)(q0), ..., g^(ode_order-1)(q0)} to obtain
    # the initial conditions g^(ode_order)(q0),... for higher-order
    # ODE derivatives.
    for i in range(ode_order, M+1):
        ode_coeffs = _ode_differentiation(ode_coeffs, i)
        # g^(i)(q0) = -(a_0*g^(0)(q0) + ... + a_{i-1}*g^(i-1)(q0))/a_i.
        g_derivative = (
            -sym.Add(*(
                coeff.subs(q, q0)*init_cond
                for coeff, init_cond in zip(ode_coeffs[:-1], init_conds, strict=True)))
            / leading_coeff_at_q0)
        init_conds.append(g_derivative)

    ode_coeffs = _ode_differentiation(ode_coeffs, M+1)

    q_tau = q0 + q1*tau + q2*tau**2
    dq_tau = q_tau.diff(tau)

    # ===== coefficient matching for the top state i = M =====
    # a_{M+1}(q(tau)) * \partial_tau g^{(M)}(q(tau))
    #   = -\partial_tau q(tau) * \sum_{i=0}^{M} a_i(q(tau)) * g^{(i)}(q(tau)).
    #
    # LHS:
    # a_{M+1}(q(tau)) * \partial_tau (\sum_{j >= 0} U[M, j] * tau^j)
    #   = a_{M+1}(q(tau)) * \sum_{j >= 1} j * U[M, j] * tau^{j-1}.
    #
    # RHS:
    # -(q1 + 2*q2*tau) * \sum_{i=0}^{M} a_i(q(tau))
    #   * \sum_{j >= 0} U[i, j] * tau^j.
    #
    # Match the coefficients of tau^{j-1}, j >= 1.

    leading_coeffs = _polynomial_coeffs(
        ode_coeffs[-1].subs(q, q_tau), sym.Matrix([tau]))
    rest = defaultdict(lambda: 0)
    for (k,), weight in leading_coeffs.items():
        if k != 0:
            # tau^k * \partial_tau (U[M, j-k] * tau^{j-k})
            #   = (j-k) * U[M, j-k] * tau^{j-1}.
            rest[M, -k] += weight*(j-k)

    for i, coeff in enumerate(ode_coeffs[:-1]):
        lower_coeffs = _polynomial_coeffs(
            dq_tau*coeff.subs(q, q_tau), sym.Matrix([tau]))
        for (k,), weight in lower_coeffs.items():
            # tau^k * U[i, j-1-k] * tau^{j-1-k}
            #   = U[i, j-1-k] * tau^{j-1}.
            rest[i, -1-k] += weight

    # a_{M+1}(q0) * j * U[M, j]
    #   = -\sum_{(i, shift)} rest[i, shift] * U[i, j+shift].
    top_state_update = {key: -weight/(leading_coeff_at_q0*j)
                        for key, weight in rest.items() if weight != 0}

    # ===== coefficient matching for the lower states, i < M =====
    #
    # \partial_tau g^{(i)}(q(tau))
    #   = g^{(i+1)}(q(tau)) * \partial_tau q(tau)
    #   = (q1 + 2*q2*tau) * g^{(i+1)}(q(tau)).
    #
    # LHS term contributing to tau^{j-1}:
    #   \partial_tau (U[i, j] * tau^j) = j * U[i, j] * tau^{j-1}.
    #
    # RHS terms contributing to tau^{j-1}:
    #     q1 * U[i+1, j-1]
    #   + 2*q2 * U[i+1, j-2].
    #
    # Hence,
    #   U[i, j] = (q1 * U[i+1, j-1] + 2*q2 * U[i+1, j-2]) / j.

    lower_state_updates = [{(i+1, -1): q1/j, (i+1, -2): 2*q2/j} for i in range(M)]
    return tuple(init_conds), (*lower_state_updates, top_state_update)
