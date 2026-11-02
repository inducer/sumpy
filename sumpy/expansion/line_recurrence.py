from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import sympy as sp


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__doc__ = r"""Derive recurrences for Taylor coefficients along a line.

Assumptions:

- Linear ODE with polynomial coefficients in :math:`q`.
- Along a line, :math:`q(\tau) = q_0 + q_1\tau + q_2\tau^2`.
- The ODE's leading coefficient is nonzero at :math:`q_0`.

.. autofunction:: recurrence
.. autofunction:: line_recurrence_table
"""

q = sp.Symbol("q")
q0, q1, q2 = sp.symbols("q0:3")
tau = sp.Symbol("tau")

# U[i, j]: [tau^j] g^(i)(q(tau)).
j = sp.Symbol("j", integer=True, nonnegative=True)
U = sp.IndexedBase("U")


def _ode_differentiation(
        ode_coeffs: Sequence[sp.Expr], order: int,
) -> tuple[sp.Expr, ...]:
    r"""
    Return the coefficients after differentiating the ODE in :math:`q`
    until its order is at least ``order``.
    """
    ode_coeffs = tuple(ode_coeffs)
    # [0, a_0, a_1, a_2] + d/dq [a_0, a_1, a_2, 0]
    #     = [a_0', a_0 + a_1', a_1 + a_2', a_2].
    for _ in range(len(ode_coeffs)-1, order):
        ode_coeffs = tuple(
            sp.expand(sp.diff(coeff, q) + previous)
            for coeff, previous in zip(
                (*ode_coeffs, 0), (0, *ode_coeffs), strict=False))
    return ode_coeffs


def recurrence(
        ode_coeffs: Sequence[sp.Expr],
        max_derivative_order: int,
        *, init_conds: Sequence[sp.Expr] | None = None,
        q0: sp.Expr = q0, q1: sp.Expr = q1, q2: sp.Expr = q2,
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    r"""
    Return extended initial conditions and update expressions for the states
    :math:`U_{i,j} = [\tau^j]g^{(i)}(q(\tau))`.
    """
    ode_order = len(ode_coeffs)-1
    if ode_order < 1:
        raise ValueError("the ODE must have positive order")
    if ode_coeffs[-1] == 0:
        raise ValueError("the ODE needs a nonzero leading coefficient")

    M = max(max_derivative_order, ode_order-1)  # ruff:ignore[non-lowercase-variable-in-function]

    if init_conds is None:
        init_conds = [sp.Symbol(f"s{i}") for i in range(ode_order)]
    else:
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
            -sp.Add(*(
                coeff.subs(q, q0)*init_cond
                for coeff, init_cond in zip(ode_coeffs, init_conds, strict=False)))
            / leading_coeff_at_q0)
        init_conds.append(g_derivative)

    ode_coeffs = _ode_differentiation(ode_coeffs, M+1)

    # Example: 2D Laplace, q*g^(2)(q) + g^(1)(q) = 0, with M = 1.
    # Let U_1(tau) = g^(1)(q(tau)). Then,
    # d/dtau U_1(tau) = q'(tau)*g^(2)(q(tau)) = -q'(tau)*U_1(tau)/q(tau).
    # The ODE in tau is
    #     q(tau)*(d/dtau)U_1(tau) + q'(tau)*U_1(tau) = 0,
    # where q(tau) = q0 + q1*tau + q2*tau^2.
    #
    # Therefore, the coefficient of tau^(j-1) in the LHS is
    #  q0*j*U[M, j]                                     (unknown)
    #  + q1*(j-1)*U[M, j-1] + q2*(j-2)*U[M, j-2]        (rest = sum(...))
    #  + q1*U[M, j-1] + 2*q2*U[M, j-2]                  (rest += sum(...))
    #
    # Hence, U[M, j] = -(q1*U[M, j-1] + q2*U[M, j-2])/q0.

    q_tau = q0 + q1*tau + q2*tau**2
    dq_tau = sp.diff(q_tau, tau)
    leading_coeffs = sp.Poly(ode_coeffs[-1].subs(q, q_tau), tau).terms()
    rest = sum(weight*(j-k)*U[M, j-k]
               for (k,), weight in leading_coeffs if k != 0)

    for i, coeff in enumerate(ode_coeffs[:-1]):
        lower_coeffs = sp.Poly(dq_tau*coeff.subs(q, q_tau), tau).terms()
        rest += sum(weight*U[i, j-1-k] for (k,), weight in lower_coeffs)

    # i = M: solve the differentiated ODE in tau for U[M, j].
    umj_update_expr = -rest/(leading_coeff_at_q0*j)

    # i < M: (d/dtau)U_i(tau) = q'(tau)*U_{i+1}(tau).
    # [tau^(j-1)] (d/dtau)U_i(tau): j*U[i, j].
    # [tau^(j-1)] q'(tau)*U_{i+1}(tau): q1*U[i+1, j-1] + 2*q2*U[i+1, j-2].
    uij_update_exprs = [
        (q1*U[i+1, j-1] + 2*q2*U[i+1, j-2])/j for i in range(M)]
    uij_update_exprs.append(umj_update_expr)
    return tuple(init_conds), tuple(uij_update_exprs)


_ValueT = TypeVar("_ValueT")


class _LineRecurrenceTable(dict[tuple[int, int], _ValueT | Literal[0]]):  # ruff:ignore[subclass-builtin]
    def __missing__(self, key: tuple[int, int]) -> Literal[0]:
        """
        Return zero for negative Taylor degrees and raise KeyError otherwise.
        """
        _, j = key
        if j < 0:
            return 0
        raise KeyError(key)


def line_recurrence_table(
        init_conds: Sequence[_ValueT],
        uij_update: Callable[[int, _LineRecurrenceTable[_ValueT]], Iterable[_ValueT]],
        expansion_order: int,
) -> _LineRecurrenceTable[_ValueT]:
    r"""
    Return the state table :math:`U_{i,j} = [\tau^j]g^{(i)}(q(\tau))`
    through Taylor degree ``expansion_order``.
    """
    states = _LineRecurrenceTable()
    for i, value in enumerate(init_conds):
        states[i, 0] = value

    for j in range(1, expansion_order+1):
        for i, value in enumerate(uij_update(j, states)):
            states[i, j] = value
    return states
