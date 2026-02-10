r"""
With the functionality in this module, we aim to compute a recurrence for
one-dimensional derivatives of functions :math:`f:\mathbb R^n \to \mathbb R`
for functions satisfying two assumptions:

- :math:`f` satisfies a PDE that is linear and has coefficients polynomial
  in the coordinates.
- :math:`f` only depends on the radius :math:`r`,
  i.e. :math:`f(\boldsymbol x)=f(|\boldsymbol x|_2)`.

This process proceeds in multiple steps:

- Convert from the PDE to an ODE in :math:`r`, using :func:`pde_to_ode_in_r`.
- Convert from an ODE in :math:`r` to one in :math:`x`,
  using :func:`ode_in_r_to_x`.
- Sort general-form ODE in :math:`x` into a coefficient array, using
  :func:`ode_in_x_to_coeff_array`.
- Finally, get an expression for the recurrence, using
  :func:`recurrence_from_coeff_array`.

The whole process can be automated using :func:`recurrence_from_pde`.

Once the recurrence is obtained, it is reindexed via
:func:`reindex_recurrence_relation`, so that :math:`s(n)` is expressed in
terms of :math:`s(n-1), s(n-2), \dots`

Computing derivatives
^^^^^^^^^^^^^^^^^^^^^

Given a PDE and its Green's function, we want to compute the :math:`n`-th
derivative :math:`\partial^n G / \partial x_1^n` at a point
:math:`(x_1, x_2, \dots)`. Here :math:`x_1` is the first coordinate
(called ``x0`` in the 0-indexed code variables).

There are two regimes, selected based on the relative magnitude of
:math:`|x_1|`:

- **Large-** :math:`|x_1|` **regime** (:math:`|x_1|/\bar x > 1`):
  Use :func:`get_large_x1_recurrence` directly. The recurrence involves
  all coordinates :math:`(x_1, x_2, \dots)`.

- **Small-** :math:`|x_1|` **regime** (:math:`|x_1|/\bar x \le 1`):
  Use :func:`get_small_x1_expansion`, which returns a Taylor expansion
  in :math:`x_1` whose coefficients are computed via
  :func:`get_small_x1_recurrence` (a recurrence evaluated at
  :math:`x_1 = 0`). The truncation order of the Taylor expansion is
  user-selectable.

::

    Want: d^n G / d x_1^n  at point (x_1, x_2, ...)

                        |x_1| / x_bar > 1?
                       /                   \
                     Yes                    No
                     /                       \
    +---------------------+    +-------------------------+
    | get_large_x1_       |    | get_small_x1_recurrence |
    | recurrence          |    | (recurrence at x_1 = 0  |
    |                     |    |  for Taylor coefficients)|
    | s(n) depends on     |    +-------------------------+
    | s(n-1), ... and     |                 |
    | x_1, x_2, ...      |                 | coefficients
    +---------------------+                 v
             |             +-------------------------+
             |             | get_small_x1_expansion  |
             |             | (Taylor expansion in    |
             |             |  x_1 with user-chosen   |
             |             |  truncation order)       |
             |             +-------------------------+
             |                          |
             v                          v
    +------------------------------------------+
    |         d^n G / d x_1^n                  |
    +------------------------------------------+

Example: large-:math:`|x_1|` recurrence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import sympy as sp
    from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
    from sumpy.recurrence import get_large_x1_recurrence, _make_sympy_vec

    # 1. Define PDE (2D Laplace)
    w = make_identity_diff_op(2)
    pde = laplacian(w)

    # 2. Get recurrence
    n_initial, order, recurrence = get_large_x1_recurrence(pde)
    # n_initial: number of initial derivatives to seed directly
    # order:     recurrence order (how many prior values s(n) depends on)
    # recurrence: sympy expression giving s(n) in terms of s(n-1), ...

    # 3. Compute derivatives at point (x0, x1) = (0.5, 0.3)
    n = sp.symbols("n")
    s = sp.Function("s")
    var = _make_sympy_vec("x", 2)
    x_vals = [(var[0], sp.Rational(1, 2)), (var[1], sp.Rational(3, 10))]

    # Seed initial conditions by direct differentiation of G
    import numpy as np
    var_t = _make_sympy_vec("t", 2)
    g = (-1/(2*np.pi)) * sp.log(sp.sqrt((var[0]-var_t[0])**2
                                         + (var[1]-var_t[1])**2))
    derivs = {}
    for i in range(-order, 0):
        derivs[i] = 0j
    for i in range(n_initial):
        d = sp.diff(g, var[0], i)
        for j in range(2):
            d = d.subs(var_t[j], 0)
        derivs[i] = complex(d.subs(x_vals))

    # Apply recurrence up to order p
    p = 8
    for i in range(n_initial, p + 1):
        expr = recurrence.subs(n, i)
        for j in range(order, 0, -1):
            expr = expr.subs(s(i - j), derivs[i - j])
        derivs[i] = complex(expr.subs(x_vals))

Example: small-:math:`|x_1|` expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sumpy.recurrence import get_small_x1_recurrence, get_small_x1_expansion

    # 1. Get the small-|x_1| recurrence (for Taylor coefficients at x_1=0)
    start_order, recur_order, recur = get_small_x1_recurrence(pde)

    # 2. Get the Taylor expansion with chosen truncation order
    taylor_order = 8
    expansion, n_coeffs, start_order = get_small_x1_expansion(
        pde, taylor_order)
    # expansion: sympy expression in s(n), s(n-1), ..., and x0
    # n_coeffs:  number of prior recurrence values needed
    # start_order: minimum n at which the expansion is valid

    # 3. Compute Taylor coefficients via the small-|x_1| recurrence
    #    (these are derivatives evaluated at x_1=0)
    coeffs = {}
    for i in range(-recur_order, 0):
        coeffs[i] = 0j
    for i in range(start_order):
        # Seed by direct differentiation of G at x0=0
        d = sp.diff(g, var[0], i)
        for j in range(2):
            d = d.subs(var_t[j], 0)
        coeffs[i] = complex(d.subs(var[0], 0).subs(x_vals))
    for i in range(start_order, p + 1):
        expr = recur.subs(n, i)
        for j in range(recur_order, 0, -1):
            expr = expr.subs(s(i - j), coeffs[i - j])
        coeffs[i] = complex(expr.subs(x_vals))

    # 4. Evaluate the expansion at a point with small x_1
    for i in range(start_order, p + 1):
        expr = expansion.subs(n, i)
        for j in range(n_coeffs, -1, -1):
            expr = expr.subs(s(i - j), coeffs[i - j])
        deriv_i = complex(expr.subs(x_vals))

.. autofunction:: pde_to_ode_in_r
.. autofunction:: ode_in_r_to_x
.. autofunction:: ode_in_x_to_coeff_array
.. autofunction:: recurrence_from_coeff_array
.. autofunction:: recurrence_from_pde
.. autofunction:: reindex_recurrence_relation
.. autofunction:: get_large_x1_recurrence
.. autofunction:: get_small_x1_recurrence
.. autofunction:: get_small_x1_expansion
"""

from __future__ import annotations


__copyright__ = """
Copyright (C) 2024 Hirish Chandrasekaran
Copyright (C) 2024 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import math
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import sympy as sp
from sympy import Expr, Symbol

from pytools import obj_array


if TYPE_CHECKING:
    from sumpy.expansion.diff_op import (
        DerivativeIdentifier,
        LinearPDESystemOperator,
    )


# similar to make_sym_vector in sumpy.symbolic, but returns an object array
# instead of a sympy.Matrix.
def _make_sympy_vec(name: str, n: int) -> np.ndarray:
    return obj_array.make_obj_array(  # pyright: ignore[reportReturnType]
        [sp.Symbol(f"{name}{i}") for i in range(n)])


def pde_to_ode_in_r(pde: LinearPDESystemOperator) -> tuple[
        Expr, np.ndarray, int
]:
    r"""
    Returns an ODE satisfied by the radial derivatives of a function
    :math:`f:\mathbb R^n \to \mathbb R` satisfying
    :math:`f(\boldsymbol x)=f(|\boldsymbol x|_2)` and *pde*.

    :arg pde: must satisfy ``pde.eqs == 1`` and have polynomial coefficients.

    :returns: a tuple ``(ode_in_r, var, ode_order)``, where

        - *ode_in_r* with derivatives given as ``sympy.Derivative``
        - *var* is an object array of ``sympy.Symbol``, with successive
          variables representing the Cartesian coordinate directions.
        - *ode_order* the order of ODE that is returned
    """
    if len(pde.eqs) != 1:
        raise ValueError("PDE must be scalar")

    dim = pde.dim
    ode_order = pde.order
    pde_eqn, = pde.eqs

    var = _make_sympy_vec("x", dim)
    r = sp.sqrt(sum(var**2))
    eps = sp.symbols("epsilon")
    rval = r + eps
    f = sp.Function("f")

    def apply_deriv_id(expr: Expr,
                       deriv_id: DerivativeIdentifier) -> Expr:
        for i, nderivs in enumerate(deriv_id.mi):
            expr = expr.diff(var[i], nderivs)
        return expr

    ode_in_r: Expr = sum(  # pyright: ignore[reportAssignmentType]
        # pylint: disable-next=not-callable
        coeff * apply_deriv_id(f(rval), deriv_id)
        for deriv_id, coeff in pde_eqn.items()
    )

    f_r_derivs = _make_sympy_vec("f_r", ode_order+1)
    # pylint: disable-next=not-callable
    f_derivs = [sp.diff(f(rval), eps, i) for i in range(ode_order+1)]

    # PDE ORDER = ODE ORDER
    for i in range(ode_order+1):
        ode_in_r = ode_in_r.subs(  # pyright: ignore[reportAssignmentType]
            f_derivs[i], f_r_derivs[i])

    return ode_in_r, var, ode_order


def _generate_nd_derivative_relations(
        var: np.ndarray, ode_order: int
) -> dict[Symbol, Any]:
    r"""
    Using the chain rule outputs a vector that gives in each component
    respectively
    :math:`[f(r), f'(r), \dots, f^{(ode_order)}(r)]` as a linear combination of
    :math:`[f(x), f'(x), \dots, f^{(ode_order)}(x)]`

    :arg var: array of sympy variables math:`[x_0, x_1, \dots]`
    :arg ode_order: the order of the ODE that we will be translating
    """
    f_r_derivs = _make_sympy_vec("f_r", ode_order+1)
    f_x_derivs = _make_sympy_vec("f_x", ode_order+1)
    f = sp.Function("f")
    eps = sp.symbols("epsilon")
    rval = sp.sqrt(sum(var**2)) + eps
    # pylint: disable=not-callable
    f_derivs_x = [sp.diff(f(rval), var[0], i) for i in range(ode_order+1)]
    f_derivs = [sp.diff(f(rval), eps, i) for i in range(ode_order+1)]
    # pylint: disable=not-callable
    for i in range(len(f_derivs_x)):
        for j in range(len(f_derivs)):
            f_derivs_x[i] = f_derivs_x[i].subs(f_derivs[j], f_r_derivs[j])
    system = [f_x_derivs[i] - f_derivs_x[i] for i in range(ode_order+1)]
    return sp.solve(system, *f_r_derivs, dict=True)[0]


def ode_in_r_to_x(ode_in_r: Expr, var: np.ndarray,
                  ode_order: int) -> Expr:
    r"""
    Translates an ode in the variable r into an ode in the variable x
    by replacing the terms :math:`f, f_r, f_{rr}, \dots` as a linear
    combinations of
    :math:`f, f_x, f_{xx}, \dots` using the chain rule.

    :arg ode_in_r: a linear combination of :math:`f, f_r, f_{rr}, \dots`
        represented by the sympy variables :math:`f_{r0}, f_{r1}, f_{r2}, \dots`
    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    :arg ode_order: the order of the input ODE

    :returns: *ode_in_x* a linear combination of :math:`f, f_x, f_{xx}, \dots`
        represented by the sympy variables :math:`f_{x0}, f_{x1}, f_{x2},
        \dots` with coefficients as rational functions in
        :math:`x_0, x_1, \dots`
    """
    subme = _generate_nd_derivative_relations(var, ode_order+1)
    ode_in_x = ode_in_r
    f_r_derivs = _make_sympy_vec("f_r", ode_order+1)
    for i in range(ode_order+1):
        ode_in_x = ode_in_x.subs(f_r_derivs[i], subme[f_r_derivs[i]])
    return ode_in_x


ODECoefficients = list[list[Expr]]


def ode_in_x_to_coeff_array(poly: sp.Poly, ode_order: int, var:
                            np.ndarray) -> ODECoefficients:
    r"""
    Organizes the coefficients of an ODE in the :math:`x_0` variable into a
    2D array.

    :arg poly: a sympy polynomial in
        :math:`\partial_{x_0}^0 f, \partial_{x_0}^1 f,\cdots` of the form
        :math:`(b_{00} x_0^0 + b_{01} x_0^1 + \cdots) \partial_{x_0}^0 f +
        (b_{10} x_0^0 + b_{11} x_0^1 +\cdots) \partial_x^1 f`

    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    :arg ode_order: the order of the input ODE we return a sequence

    :returns: *coeffs* a sequence of of sequences, with the outer sequence
        iterating over derivative orders, and each inner sequence iterating
        over powers of :math:`x_0`, so that, in terms of the above form,
        coeffs is :math:`[[b_{00}, b_{01}, ...], [b_{10}, b_{11}, ...], ...]`
    """
    return [
        # recast ODE coefficient obtained below as polynomial in x0
        sp.Poly(
            # get coefficient of deriv_ind'th derivative
            poly.coeff_monomial(poly.gens[deriv_ind]),

            var[0])
        # get poly coefficients in /ascending/ order
        .all_coeffs()[::-1]
        for deriv_ind in range(ode_order+1)]


NumberT = TypeVar("NumberT", int, float, complex)


def _falling_factorial(arg: NumberT, num_terms: int) -> NumberT:
    result = 1
    for i in range(num_terms):
        result = result * (arg - i)
    return result


def _auto_product_rule_single_term(p: int, m: int, var: np.ndarray) -> Expr:
    r"""
    We assume that we are given the expression :math:`x_0^p f^(m)(x_0)`. We
    then output the nth order derivative of the expression where :math:`n` is
    a symbolic variable.
    We let :math:`s(i)` represent the ith order derivative of f when
    we output the final result.
    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    """
    n = sp.symbols("n")
    s = sp.Function("s")

    return sum(  # pyright: ignore[reportReturnType]
        # pylint: disable=not-callable
        _falling_factorial(n, i)
        * math.comb(p, i) * s(n-i+m) * var[0]**(p-i)
        for i in range(p+1)
    )


def recurrence_from_coeff_array(
        coeffs: list[list[Any]], var: np.ndarray
) -> Expr:
    r"""
    A function that takes in as input an organized 2D coefficient array (see
    above) and outputs a recurrence relation.

    :arg coeffs: a sequence of of sequences, described in
        :func:`ode_in_x_to_coeff_array`
    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    """
    final_recurrence: Any = 0
    # Outer loop is derivative direction
    # Inner is polynomial order of x_0
    for m, _ in enumerate(coeffs):
        for p, _ in enumerate(coeffs[m]):
            final_recurrence += coeffs[m][p] * _auto_product_rule_single_term(
                p, m, var)
    return final_recurrence


def recurrence_from_pde(pde: LinearPDESystemOperator) -> Expr:
    r"""
    Takes a PDE and outputs a recurrence relation by composing
    :func:`pde_to_ode_in_r`, :func:`ode_in_r_to_x`,
    :func:`ode_in_x_to_coeff_array`, and :func:`recurrence_from_coeff_array`.

    :arg pde: a :class:`sumpy.expansion.diff_op.LinearPDESystemOperator`
        that must satisfy ``pde.eqs == 1`` and have polynomial coefficients
        in the coordinates.

    :returns: a recurrence relation as a sympy expression involving
        :math:`s(n), s(n-1), \dots` that evaluates to zero.
    """
    ode_in_r, var, ode_order = pde_to_ode_in_r(pde)
    ode_in_x = ode_in_r_to_x(ode_in_r, var, ode_order).simplify()
    ode_in_x_cleared = (ode_in_x * var[0]**(pde.order*2-1)).simplify()
    # ode_in_x_cleared shouldn't have rational function coefficients
    assert sp.together(ode_in_x_cleared) == ode_in_x_cleared
    f_x_derivs = _make_sympy_vec("f_x", ode_order+1)
    poly = sp.Poly(ode_in_x_cleared, *f_x_derivs)
    coeffs = ode_in_x_to_coeff_array(poly, ode_order, var)
    return recurrence_from_coeff_array(coeffs, var)


def reindex_recurrence_relation(r: sp.Basic) -> tuple[int, Expr]:
    r"""
    Reindexes a recurrence relation so that :math:`s(n)` is expressed in terms
    of :math:`s(n-1), s(n-2), \dots`. The input recurrence is an expression
    that evaluates to zero, while the output gives :math:`s(n)` directly in
    terms of prior values.

    :arg r: a recurrence relation expression in :math:`s(n)` that evaluates
        to zero.

    :returns: a tuple ``(order, reindexed_recurrence)``, where

        - *order* is the order of the recurrence (the difference between the
          highest and lowest indexed terms).
        - *reindexed_recurrence* is a sympy expression giving :math:`s(n)` in
          terms of :math:`s(n-1), s(n-2), \dots`
    """
    idx_l, terms = _extract_idx_terms_from_recurrence(r)
    # Order is the max difference between highest/lowest in idx_l
    order = max(idx_l) - min(idx_l)

    # How much do we need to shift the recurrence relation
    shift_idx = max(idx_l)

    # Get the respective coefficients in the recurrence relation from r
    n = sp.symbols("n")
    coeffs = sp.poly(r, list(terms)).coeffs()

    # Re-arrange the recurrence relation so we get s(n) = ____
    # in terms of s(n-1), ...
    true_recurrence: Expr = sum(  # pyright: ignore[reportAssignmentType]
        sp.cancel(-coeffs[i]/coeffs[-1]) * terms[i]
        for i in range(0, len(terms)-1))
    true_recurrence1 = true_recurrence.subs(n, n-shift_idx)

    return order, true_recurrence1


def _extract_idx_terms_from_recurrence(r: sp.Basic) -> tuple[np.ndarray,
                                                              np.ndarray]:
    r"""
    Given a recurrence extracts the variables in the recurrence
    as well as the indexes, both in sorted order.

    :arg r: recurrence to extract terms from
    """
    # We're assuming here that s(...) are the only function calls.
    terms = list(r.atoms(sp.Function))
    terms = np.array(terms)

    idx_l = []
    for i in range(len(terms)):
        tms = list(terms[i].atoms(sp.Number))
        if len(tms) == 1:
            idx_l.append(tms[0])
        else:
            idx_l.append(0)
    idx_l = np.array(idx_l, dtype="int")
    idx_sort = idx_l.argsort()
    idx_l = idx_l[idx_sort]
    terms = terms[idx_sort]

    return idx_l, terms


def _check_neg_ind(r_n: sp.Basic) -> bool:
    r"""
    Simply checks if a negative index exists in a recurrence relation.
    """

    idx_l, _ = _extract_idx_terms_from_recurrence(r_n)

    return bool(np.any(idx_l < 0))


def _get_initial_order_large_x1(recurrence: Expr) -> int:
    r"""
    For the large-:math:`|x_1|` recurrence, checks how many initial
    conditions are needed by checking for non-negative indexed terms.
    """
    n = sp.symbols("n")

    i = 0
    r_c = recurrence.subs(n, i)
    while _check_neg_ind(r_c):
        i += 1
        r_c = recurrence.subs(n, i)
    return i


def _get_initial_order_small_x1(recurrence: Expr) -> int:
    r"""
    For the small-:math:`|x_1|` recurrence, checks how many initial
    conditions are needed by checking for non-negative indexed terms.
    """
    n = sp.symbols("n")

    i = 0
    r_c = recurrence.subs(n, i)
    while (_check_neg_ind(r_c) or r_c == 0) or i % 2 != 0:
        i += 1
        r_c = recurrence.subs(n, i)
    return i


def get_large_x1_recurrence(
        pde: LinearPDESystemOperator
) -> tuple[int, int, Expr]:
    r"""
    Computes the large-:math:`|x_1|` recurrence for evaluating
    one-dimensional derivatives of a radially symmetric Green's function
    satisfying *pde*. This recurrence is used when :math:`|x_1|` (the
    on-axis coordinate) is large relative to the off-axis coordinates.
    The recurrence is reindexed so that :math:`s(n)` is given in terms of
    :math:`s(n-1), \dots`

    :arg pde: a :class:`sumpy.expansion.diff_op.LinearPDESystemOperator`
        that must satisfy ``pde.eqs == 1`` and have polynomial coefficients
        in the coordinates.

    :returns: a tuple ``(n_initial, order, recurrence)``, where

        - *n_initial* is the number of initial derivatives that must be
          computed directly (i.e. not via the recurrence).
        - *order* is the order of the recurrence.
        - *recurrence* is the reindexed recurrence giving :math:`s(n)` in
          terms of :math:`s(n-1), \dots`
    """
    r = recurrence_from_pde(pde)
    order, r_p = reindex_recurrence_relation(r)
    n_initial = _get_initial_order_large_x1(r_p)
    return n_initial, order, r_p


# ================ SMALL-|x_1| RECURRENCE AND EXPANSION =================
def get_small_x1_recurrence(
        pde: LinearPDESystemOperator
) -> tuple[int, int, Expr]:
    r"""
    Computes the small-:math:`|x_1|` recurrence for evaluating
    one-dimensional derivatives of a radially symmetric Green's function
    satisfying *pde*, evaluated at :math:`x_1 = 0`. This recurrence produces
    the Taylor coefficients used by :func:`get_small_x1_expansion`.
    The recurrence is reindexed so that :math:`s(n)` is given in terms of
    :math:`s(n-1), \dots`

    :arg pde: a :class:`sumpy.expansion.diff_op.LinearPDESystemOperator`
        that must satisfy ``pde.eqs == 1`` and have polynomial coefficients
        in the coordinates.

    :returns: a tuple ``(start_order, recur_order, recur)``, where

        - *start_order* is the derivative order at which the recurrence
          first becomes valid (lower orders must be computed directly).
        - *recur_order* is the order of the recurrence *recur*.
        - *recur* is the reindexed small-:math:`|x_1|` recurrence giving
          :math:`s(n)` in terms of :math:`s(n-1), \dots`
    """
    var = _make_sympy_vec("x", 1)
    r_exp = recurrence_from_pde(pde).subs(var[0], 0)
    recur_order, recur = reindex_recurrence_relation(r_exp)
    start_order = _get_initial_order_small_x1(recur)
    return start_order, recur_order, recur


def get_small_x1_expansion(
        pde: LinearPDESystemOperator, taylor_order: int = 4
) -> tuple[Expr, int, int]:
    r"""
    Computes the small-:math:`|x_1|` expansion: a truncated Taylor expansion
    in :math:`x_1` that expresses the :math:`n`-th derivative in terms of
    small-:math:`|x_1|` recurrence values :math:`s(n), s(n-1), \dots`
    See :func:`get_small_x1_recurrence`.

    :arg pde: a :class:`sumpy.expansion.diff_op.LinearPDESystemOperator`
        that must satisfy ``pde.eqs == 1`` and have polynomial coefficients
        in the coordinates.
    :arg taylor_order: order of the Taylor expansion in :math:`x_1`.

    :returns: a tuple ``(exp, n_coeffs, start_order)``, where

        - *exp* is the Taylor expansion expression in terms of :math:`s(n),
          s(n-1), \dots` and :math:`x_1`. Must not be evaluated for
          :math:`n` below *start_order*.
        - *n_coeffs* is the number of prior small-:math:`|x_1|` recurrence
          values needed. For example, if *n_coeffs* is 3, then
          :math:`s(n), s(n-1), s(n-2), s(n-3)` are required.
        - *start_order* is the minimum derivative order at which the
          expression is valid.
    """
    s = sp.Function("s")
    n = sp.symbols("n")
    deriv_order = n

    start_order, _, t_recurrence = get_small_x1_recurrence(pde)
    var = _make_sympy_vec("x", 2)
    exp: Any = 0
    for i in range(taylor_order+1):
        exp += s(deriv_order+i)/math.factorial(i) * var[0]**i

    # While derivatives w/order larger than the deriv_order exist in our
    # taylor expression replace them with smaller order derivatives

    idx_l, _ = _extract_idx_terms_from_recurrence(exp)
    max_idx = max(idx_l)

    while max_idx > 0:
        for ind in idx_l:
            if ind > 0:
                exp = exp.subs(s(n+ind), t_recurrence.subs(n, n+ind))

        idx_l, _ = _extract_idx_terms_from_recurrence(exp)
        max_idx = max(idx_l)

    idx_l, _ = _extract_idx_terms_from_recurrence(exp)

    return exp, -min(idx_l), start_order
