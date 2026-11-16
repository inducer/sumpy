from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import sumpy.symbolic as sym
from sumpy.tools import mi_increment_axis


if TYPE_CHECKING:
    from collections.abc import Sequence

__doc__ = r"""
Derivative Recurrence
=====================

Derive symbolic one-axis recurrence formulas for the scaled
multivariate Taylor coefficients
:math:`U[i, \mathrm{mi}]` of :math:`g^{(i)}(|z|^2)`.

Notation:

- :math:`q = |z|^2` and :math:`q(h) = |z + \mathrm{rscale} \cdot h|^2`.
- :math:`U[i, \mathrm{mi}] = [h^{\mathrm{mi}}]g^{(i)}(q(h))
  = \mathrm{rscale}^{|\mathrm{mi}|}
  \partial_z^{\mathrm{mi}}[g^{(i)}(|z|^2)]/\mathrm{mi}!`,
  where :math:`i` is the :math:`q`-derivative order and :math:`\mathrm{mi}` is
  the Taylor multi-index for :math:`h`.

Assumptions:

- The kernel has the form :math:`G(z) = g(q)`.
- :math:`g` satisfies :math:`\sum_{i=0}^{n} a_i(q) g^{(i)}(q) = 0`, an ODE of
  order :math:`n \ge 1` with polynomial coefficients :math:`a_i(q)`.
- The ODE's leading coefficient is nonzero at :math:`q = |z|^2`.

.. autofunction:: derivative_recurrence
"""


def _polynomial_coeffs(
        expression: sym.Expr, variables: sym.Matrix,
) -> dict[tuple[int, ...], sym.Expr]:
    """
    Return the nonzero coefficients of a multivariate polynomial.
    """
    coeffs = defaultdict(lambda: 0)
    for term, coefficient in expression.expand().as_coefficients_dict().items():
        powers = term.as_powers_dict()
        mi = tuple(int(powers.get(variable, 0)) for variable in variables)
        weight = coefficient*term
        for variable, power in zip(variables, mi, strict=True):
            weight /= variable**power
        coeffs[mi] += weight
    return {mi: weight.expand() for mi, weight in coeffs.items() if weight != 0}


def derivative_recurrence(
        ode_coeffs: Sequence[sym.Expr | int | float | complex],
        dim: int, axis: int = 0, rscale: sym.Expr | int = 1,
) -> tuple[dict[tuple[int, tuple[int, ...]], sym.Expr], ...]:
    r"""
    Return the update formulas for :math:`U[i, \mathrm{mi}]` with
    :math:`\mathrm{mi}_k \ge 1`, along :math:`k` = ``axis``.

    For :math:`0 \le i \le \mathrm{ode\_order}-2`, the chain rule gives

    .. math::

        U[i, \mathrm{mi}] = \frac{2 \mathrm{rscale} \cdot z_k U[i+1, \mathrm{mi}-e_k]
            + 2 \mathrm{rscale}^2 U[i+1, \mathrm{mi}-2e_k]}{\mathrm{mi}_k}.

    For :math:`i = \mathrm{ode\_order}-1`, the ODE gives

    .. math::

        U[i, \mathrm{mi}] = \sum_{(i',\mathrm{shift})}
            c_{i',\mathrm{shift}}(\mathrm{mi}, k, z, \mathrm{rscale})
            U[i', \mathrm{mi}+\mathrm{shift}],

    where :math:`\mathrm{shift} \le 0` and :math:`\mathrm{shift} \ne 0`.
    """
    q = sym.Symbol("q")
    z = sym.Matrix(sym.symbols(f"z0:{dim}"))
    h = sym.Matrix(sym.symbols(f"h0:{dim}"))
    mi = sym.symbols(f"mi0:{dim}")
    ode_coeffs = tuple(sym.sympify(coeff).expand() for coeff in ode_coeffs)
    ode_order = len(ode_coeffs)-1

    q_h = (z+rscale*h).dot(z+rscale*h)
    dq_h = q_h.diff(h[axis])

    # ===== coefficient matching for the top state i = ode_order-1 =====
    # a_{i+1}(q(h)) * \partial_h^{e_k} g^{(i)}(q(h))
    #   = -\partial_h^{e_k} q(h) * \sum_{j=0}^{i} a_j(q(h)) * g^{(j)}(q(h)).
    #
    # LHS:
    # a_{i+1}(q(h)) * \partial_h^{e_k} g^{(i)}(q(h))
    #   = a_{i+1}(q(h)) * \partial_h^{e_k} (\sum_{mi >= 0} U[i, mi] * h^{mi})
    #   = a_{i+1}(q(h)) * \sum_{mi >= 0} mi_k * U[i, mi] * h^{mi-e_k}.
    #
    # RHS:
    # -2 * rscale * (z_k + rscale * h_k)
    #   * \sum_{j=0}^{i} a_j(q(h))
    #   * \sum_{mi >= 0} U[j, mi] * h^{mi}.
    #
    # Match the coefficients of h^{mi-e_k}, mi_k >= 1.

    leading_coeffs = _polynomial_coeffs(ode_coeffs[-1].subs(q, q_h), h)
    rest = defaultdict(lambda: 0)
    for mi_shift, weight in leading_coeffs.items():
        if not any(mi_shift):
            continue
        # h^{mi_shift} * \partial_h^{e_k} (U[i, mi-mi_shift] * h^{mi-mi_shift})
        #   = (mi_k - (mi_shift)_k) * U[i, mi-mi_shift] * h^{mi-e_k}
        rest[ode_order-1, tuple(-step for step in mi_shift)] = (
            weight*(mi[axis]-mi_shift[axis]))

    for j, lower_order_coeff in enumerate(ode_coeffs[:-1]):
        lower_coeffs = _polynomial_coeffs(dq_h*lower_order_coeff.subs(q, q_h), h)
        for mi_shift, weight in lower_coeffs.items():
            # h^{mi_shift} * U[j, mi-e_k-mi_shift] * h^{mi-e_k-mi_shift}
            #   = U[j, mi-e_k-mi_shift] * h^{mi-e_k}
            rest[j, mi_increment_axis(
                tuple(-step for step in mi_shift), axis, -1)] += weight

    # a_{i+1}(|z|^2) * mi_k * U[i, mi]
    #   = -\sum_{(j, shift)} rest[j, shift] * U[j, mi+shift]
    leading_weight = ode_coeffs[-1].subs(q, z.dot(z))
    top_state_update = {key: -weight/(leading_weight*mi[axis])
                        for key, weight in rest.items() if weight != 0}

    # ===== coefficient matching for the lower states, i < ode_order-1 =====
    #
    # \partial_h^{e_k} g^{(i)}(q(h))
    #           = g^{(i+1)}(q(h)) * \partial_h^{e_k} q(h)
    #           = 2 * rscale * (z_k + rscale * h_k) * g^{(i+1)}(q(h)).
    #
    # LHS term contributing to h^{mi-e_k}:
    #   \partial_h^{e_k} (U[i, mi] * h^{mi}) = mi_k * U[i, mi] * h^{mi-e_k}
    #
    # RHS terms contributing to h^{mi-e_k}:
    #    2 * rscale * z_k * U[i+1, mi-e_k] * h^{mi-e_k}
    #  + 2 * rscale^2 * U[i+1, mi-2e_k] * h^{mi-e_k}
    #
    # Hence,
    #   U[i, mi] = (2 * rscale * z_k * U[i+1, mi-e_k]
    #               + 2 * rscale^2 * U[i+1, mi-2e_k]) / mi_k.

    lower_state_updates = [
        {(i+1, mi_increment_axis((0,)*dim, axis, -1)):
            2*rscale*z[axis]/mi[axis],
         (i+1, mi_increment_axis((0,)*dim, axis, -2)):
            2*rscale**2/mi[axis]}
        for i in range(ode_order-1)]
    return (*lower_state_updates, top_state_update)
