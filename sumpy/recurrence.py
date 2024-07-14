"""
.. autofunction:: get_pde_in_recurrence_form
.. autofunction:: generate_nd_derivative_relations
.. autofunction:: ode_in_r_to_x
.. autofunction:: get_recurrence_parametric_from_pde
.. autofunction:: get_recurrence_parametric_from_coeffs
.. autofunction:: auto_product_rule_single_term
.. autofunction:: compute_coefficients_of_poly_parametric
"""

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
from typing import Tuple
import numpy as np
import sympy as sp
from pytools.obj_array import make_obj_array
from sumpy.expansion.diff_op import (
    make_identity_diff_op, laplacian, LinearPDESystemOperator)


# similar to make_sym_vector in sumpy.symbolic, but returns an object array
# instead of a sympy.Matrix.
def _make_sympy_vec(name, n):
    return make_obj_array([sp.Symbol(f"{name}{i}") for i in range(n)])

"""
Overall Narrative:
First we take an elliptic PDE represented as a sumpy 
:class:`sumpy.expansion.diff_op.LinearSystemPDEOperator` type. We then
use get_pde_in_recurrence_form to get an ODE in the radial variable
that the point-potential satisfies assuming radial symmetry of the point-potential.

We then take the ODE in the radial variable that we get and use the chain-rule to
convert it into a ODE in a single spatial variable using ode_in_r_to_x. We then
collect the coefficients of the ODE using compute_coefficients_of_poly_parametric 
and then use these coefficients to finally compute the recurrence relation via
get_recurrence_parametric_from_pde.
"""


def get_pde_in_recurrence_form(pde: LinearPDESystemOperator) -> Tuple[
        sp.Expr, np.ndarray, int
]:
    """
    ## Input
    - *pde*, a :class:`sumpy.expansion.diff_op.LinearSystemPDEOperator` such that
    pde.eqs == 1
    ## Output
    - ode_in_r, an ODE that the point-potential satifies w/respect to radial variable
    - var, an array representing the input coordinates
    - n_derivs, the order of the original PDE + 1, i.e. the number of
      derivatives of f that may be present (the reason this is called n_derivs
      since if we have a second order PDE for example then we might see
      :math:`f, f_{r}, f_{rr}` in our ODE in r, which is technically 3 terms
      since we count the 0th order derivative f as a "derivative." If this
      doesn't make sense just know that n_derivs is the order the of the input
      sumpy PDE + 1)
    ## Description
    Takes as input a scalar pde represented as the type
    :class:`sumpy.expansion.diff_op.LinearSystemPDEOperator`. Assumes that the scalar
    pde has coefficients that are polynomial in the input coordinates. Then assumes
    that the PDE is satisfied by a point-potential with radial symmetry and comes up
    with an ODE in the radial variable that the point-potential satisfies.
    """
    if len(pde.eqs) != 1:
        raise ValueError("PDE must be scalar")

    dim = pde.dim
    n_derivs = pde.order
    assert (len(pde.eqs) == 1)
    ops = len(pde.eqs[0])
    derivs = []
    coeffs = []
    for i in pde.eqs[0]:
        derivs.append(i.mi)
        coeffs.append(pde.eqs[0][i])
    var = _make_sympy_vec("x", dim)
    r = sp.sqrt(sum(var**2))

    eps = sp.symbols("epsilon")
    rval = r + eps
    f = sp.Function("f")
    # pylint: disable=not-callable
    f_derivs = [sp.diff(f(rval), eps, i) for i in range(n_derivs+1)]

    def compute_term(a, t):
        term = a
        for i in range(len(t)):
            term = term.diff(var[i], t[i])
        return term

    ode_in_r = 0
    for i in range(ops):
        ode_in_r += coeffs[i] * compute_term(f(rval), derivs[i])
    n_derivs = len(f_derivs)
    f_r_derivs = _make_sympy_vec("f_r", n_derivs)

    for i in range(n_derivs):
        ode_in_r = ode_in_r.subs(f_derivs[i], f_r_derivs[i])
    return ode_in_r, var, n_derivs


def generate_nd_derivative_relations(var: np.ndarray, n_derivs: int) -> dict:
    """
    ## Input
        - *var*, a sympy vector of variables called [x0, x1, ...]
        - *n_derivs*, the order of the original PDE + 1, i.e. the number of
          derivatives of f that may be present
    ## Output
        - a vector that gives [f, f_r, f_{rr}, ...] in terms of f, f_x, f_{xx}, ...
          using the chain rule
          (f, f_x, f_{xx}, ... in code is represented as f_{x0}, f_{x1}, f_{x2} and
          f, f_r, f_{rr}, ... in code is represented as f_{r0}, f_{r1}, f_{r2})
    ## Description
    Using the chain rule outputs a vector that tells us how to
    write f, f_r, f_{rr}, ... as a linear combination of f, f_x, f_{xx}, ...
    """
    f_r_derivs = _make_sympy_vec("f_r", n_derivs)
    f_x_derivs = _make_sympy_vec("f_x", n_derivs)
    f = sp.Function("f")
    eps = sp.symbols("epsilon")
    rval = sp.sqrt(sum(var**2)) + eps
    # pylint: disable=not-callable
    f_derivs_x = [sp.diff(f(rval), var[0], i) for i in range(n_derivs)]
    f_derivs = [sp.diff(f(rval), eps, i) for i in range(n_derivs)]
    # pylint: disable=not-callable
    for i in range(len(f_derivs_x)):
        for j in range(len(f_derivs)):
            f_derivs_x[i] = f_derivs_x[i].subs(f_derivs[j], f_r_derivs[j])
    system = [f_x_derivs[i] - f_derivs_x[i] for i in range(n_derivs)]
    return sp.solve(system, *f_r_derivs, dict=True)[0]


def ode_in_r_to_x(ode_in_r: sp.Expr, var: np.ndarray, n_derivs: int) -> sp.Expr:
    """
    ## Input
        - *ode_in_r*, a linear combination of f, f_r, f_{rr}, ...
          (in code represented as f_{r0}, f_{r1}, f_{r2})
          with coefficients that are polynomials in var[0], var[1], ...
          divided by some power of var[0]
        - *var*, array of sympy variables [x_0, x_1, ...]
        - *n_derivs*, the order of the original PDE + 1, i.e. the number of
          derivatives of f that may be present
    ## Output
        - ode_in_x, a linear combination of f, f_x, f_{xx}, ... with coefficients as
          rational functions in var[0], var[1], ...
    ## Description
    Translates an ode in the variable r into an ode in the variable x
    by substituting f, f_r, f_{rr}, ... as a linear combination of
    f, f_x, f_{xx}, ... using the chain rule.
    """
    subme = generate_nd_derivative_relations(var, n_derivs)
    ode_in_x = ode_in_r
    f_r_derivs = _make_sympy_vec("f_r", n_derivs)
    for i in range(n_derivs):
        ode_in_x = ode_in_x.subs(f_r_derivs[i], subme[f_r_derivs[i]])
    return ode_in_x


def compute_coefficients_of_poly_parametric(poly: sp.Poly, n_derivs: int,
                                            var: np.ndarray) -> list:
    """
    ## Input
        - *poly*, the original ODE for our point-potential as a polynomial
          in f_{x0}, f_{x1}, f_{x2}, etc. with polynomial coefficients
          in var[0], var[1], ...
        - *n_derivs*, the order of the original PDE + 1, i.e. the number of
          derivatives of f that may be present
        - *var*, array of sympy variables [x_0, x_1, ...]
    ## Output
        - ode_in_x, a linear combination of f, f_x, f_{xx}, ... with coefficients as
          rational functions in var[0], var[1], ...
    ## Description
    Translates an ode in the variable r into an ode in the variable x
    by substituting f, f_r, f_{rr}, ... as a linear combination of
    f, f_x, f_{xx}, ... using the chain rule.
    """
    def tup(i, n=n_derivs):
        a = []
        for j in range(n):
            if j != i:
                a.append(0)
            else:
                a.append(1)
        return tuple(a)

    coeffs = []
    for deriv_ind in range(n_derivs):
        coeffs.append(sp.Poly(poly.coeff_monomial(tup(deriv_ind)),
                              var[0]).all_coeffs()[::-1])

    return coeffs


def auto_product_rule_single_term(p: int, m: int, var: np.ndarray) -> sp.Expr:
    """
    ## Description
    We assume that we are given the expression :math:`x_0^p f^(m)(x_0)`. We then
    output the nth order derivative of the expression where n is a symbolic variable.
    We let :math:`s(i)` represent the ith order derivative of f when
    we output the final result.
    ## Input
    - *p*, see description
    - *m*, see description
    - *var*, array of sympy variables [x_0, x_1, ...]
    ## Output
    - A sympy expression is output corresponding to the nth order derivative of the
    input expression.
    We let :math:`s(i)` represent the ith order derivative of f when
    we output the final result. We let n represent a symbolic variable
    corresponding to how many derivatives of the original expression were
    taken.
    """
    n = sp.symbols("n")
    s = sp.Function("s")
    result = 0
    for i in range(p+1):
        temp = 1
        for j in range(i):
            temp *= (n - j)
        # pylint: disable=not-callable
        temp *= math.comb(p, i) * s(n-i+m) * var[0]**(p-i)
        result += temp
    return result


def get_recurrence_parametric_from_coeffs(coeffs: list, var: np.ndarray) -> sp.Expr:
    """
    ## Input
    - *coeffs*,
      Consider an ODE obeyed by a function f that can be expressed in the following
      form: :math:`(b_{00} x_0^0 + b_{01} x_0^1 + \\cdots) \\partial_{x_0}^0 f +
      (b_{10} x_0^0 + b_{11} x_0^1 +\\cdots) \\partial_x^1 f`. coeffs is a sequence
      of sequences, with the outer sequence iterating over derivative orders, and
      each inner sequence iterating over powers of :math:`x_0`, so that, in terms of
      the above form, coeffs is [[b_00, b_01, ...], [b_10, b_11, ...], ...]
    - *var*, array of sympy variables [x_0, x_1, ...]
    ## Output
    - final_recurrence, the recurrence relation for derivatives of our
      point-potential.
    """
    final_recurrence = 0
    #Outer loop is derivative direction
    #Inner is polynomial order of x_0
    for m, _ in enumerate(coeffs):
        for p, _ in enumerate(coeffs[m]):
            final_recurrence += coeffs[m][p] * auto_product_rule_single_term(p,
                                                                             m, var)
    return final_recurrence


def get_recurrence_parametric_from_pde(pde: LinearPDESystemOperator) -> sp.Expr:
    """
    ## Input
    - *pde*, a :class:`sumpy.expansion.diff_op.LinearSystemPDEOperator` such that
      pde.eqs == 1
    ## Output
    - final_recurrence, the recurrence relation for derivatives of our
      point-potential.
    """
    ode_in_r, var, n_derivs = get_pde_in_recurrence_form(pde)
    ode_in_x = ode_in_r_to_x(ode_in_r, var, n_derivs).simplify()
    ode_in_x_cleared = (ode_in_x * var[0]**n_derivs).simplify()
    f_x_derivs = _make_sympy_vec("f_x", n_derivs)
    poly = sp.Poly(ode_in_x_cleared, *f_x_derivs)
    coeffs = compute_coefficients_of_poly_parametric(poly, n_derivs, var)
    return get_recurrence_parametric_from_coeffs(coeffs, var)


def test_recurrence_finder_laplace():
    """
    ## Description
    Tests our recurrence relation generator for Lapalace 2D.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    r = get_recurrence_parametric_from_pde(laplace2d)
    n = sp.symbols("n")
    s = sp.Function("s")

    def deriv_laplace(i):
        x, y = sp.symbols("x,y")
        var = _make_sympy_vec("x", 2)
        true_f = sp.log(sp.sqrt(x**2 + y**2))
        return sp.diff(true_f, x, i).subs(x, var[0]).subs(
            y, var[1])
    d = 6
    # pylint: disable=not-callable

    r_sub = r.subs(n, d)
    for i in range(d-1, d+3):
        r_sub = r_sub.subs(s(i), deriv_laplace(i))
    r_sub = r_sub.simplify()

    assert r_sub == 0


def test_recurrence_finder_laplace_three_d():
    """
    ## Description
    Tests our recurrence relation generator for Laplace 3D.
    """
    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)
    r = get_recurrence_parametric_from_pde(laplace3d)
    n = sp.symbols("n")
    s = sp.Function("s")

    def deriv_laplace_three_d(i):
        x, y, z = sp.symbols("x,y,z")
        var = _make_sympy_vec("x", 3)
        true_f = 1/(sp.sqrt(x**2 + y**2 + z**2))
        return sp.diff(true_f, x, i).subs(x, var[0]).subs(
            y, var[1]).subs(z, var[2])

    d = 6
    # pylint: disable=not-callable
    r_sub = r.subs(n, d)
    for i in range(d-1, d+3):
        r_sub = r_sub.subs(s(i), deriv_laplace_three_d(i))
    r_sub = r_sub.simplify()
    assert r_sub == 0
