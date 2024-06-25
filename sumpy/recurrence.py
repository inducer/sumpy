"""
.. autofunction:: get_pde_in_recurrence_form
.. autofunction:: generate_nd_derivative_relations
.. autofunction:: ode_in_r_to_x
.. autofunction:: compute_poly_in_deriv
.. autofunction:: compute_coefficients_of_poly
.. autofunction:: compute_recurrence_relation
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

import numpy as np
import math
import sympy as sp
from typing import Tuple
from pytools.obj_array import make_obj_array
from sumpy.expansion.diff_op import (
    make_identity_diff_op, laplacian,LinearPDESystemOperator)


# similar to make_sym_vector in sumpy.symbolic, but returns an object array
# instead of a sympy.Matrix.
def _make_sympy_vec(name, n):
    return make_obj_array([sp.Symbol(f"{name}{i}") for i in range(n)])


def get_pde_in_recurrence_form(pde: LinearPDESystemOperator) -> Tuple[
        sp.Expr, np.ndarray, int
    ]:
    """
    Input:
        - *pde*, representing a scalar PDE.
    Output:
        - ode_in_r, an ode in r which the POINT-POTENTIAL (has radial symmetry)
          satisfies away from the origin.
          Note: to represent f, f_r, f_{rr}, we use the sympy variables
          :math:`f_{r0}`, f_{r1}, .... So ode_in_r is a linear combination of the sympy
          variables f_{r0}, f_{r1}, ....
        - var, represents the variables for the input space: [x0, x1, ...]
        - n_derivs, the order of the original PDE + 1, i.e. the number of
          derivatives of f that may be present (the reason this is called n_derivs
          since if we have a second order PDE for example then we might see f, f_{r},
          f_{rr} in our ODE in r, which is technically 3 terms since we count
          the 0th order derivative f as a "derivative." If this doesn't make sense
          just know that n_derivs is the order the of the input sumpy PDE + 1)

    Description: We assume we are handed a system of 1 sumpy PDE (pde) and output
    the pde in a way that allows us to easily replace derivatives with respect to r.
    In other words we output a linear combination of sympy variables
    f_{r0}, f_{r1}, ... (which represents f, f_r, f_{rr} respectively)
    to represent our ODE in r for the point potential.
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


def generate_nd_derivative_relations(var, n_derivs):
    """
    generate_nd_derivative_relations
    Input:
    - var, a sympy vector of variables called [x0, x1, ...]
    - n_derivs, the order of the original PDE + 1, i.e. the number of derivatives of
    f that may be present

    Output:
    - a vector that gives [f, f_r, f_{rr}, ...] in terms of f, f_x, f_{xx}, ...
    using the chain rule
    (f, f_x, f_{xx}, ... in code is represented as f_{x0}, f_{x1}, f_{x2} and
    f, f_r, f_{rr}, ... in code is represented as f_{r0}, f_{r1}, f_{r2})

    Description: Using the chain rule outputs a vector that tells us how to
    write f, f_r, f_{rr}, ... as a linear
    combination of f, f_x, f_{xx}, ...
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


def ode_in_r_to_x(ode_in_r, var, n_derivs):
    """
    ode_in_r_to_x
    Input:
    - ode_in_r, a linear combination of f, f_r, f_{rr}, ...
    (in code represented as f_{r0}, f_{r1}, f_{r2})
    with coefficients as RATIONAL functions in var[0], var[1], ...
    - var, array of sympy variables [x_0, x_1, ...]
    - n_derivs, the order of the original PDE + 1, i.e. the number of derivatives of
    f that may be present
    Output:
    - ode_in_x, a linear combination of f, f_x, f_{xx}, ... with coefficients as
    rational functions in var[0], var[1], ...

    Description: Translates an ode in the variable r into an ode in the variable x
    by substituting f, f_r, f_{rr}, ... as a linear combination of
    f, f_x, f_{xx}, ... using the chain rule
    """
    subme = generate_nd_derivative_relations(var, n_derivs)
    ode_in_x = ode_in_r
    f_r_derivs = _make_sympy_vec("f_r", n_derivs)
    for i in range(n_derivs):
        ode_in_x = ode_in_x.subs(f_r_derivs[i], subme[f_r_derivs[i]])
    return ode_in_x


def compute_poly_in_deriv(ode_in_x, n_derivs, var):
    """
    compute_poly_in_deriv
    Input:
    - ode_in_x, a linear combination of f, f_x, f_{xx}, ... with coefficients as
    rational functions in var[0], var[1], ...
    - n_derivs, the order of the original PDE + 1, i.e. the number of derivatives
    of f that may be present
    Output:
    - a polynomial in f, f_x, f_{xx}, ... (in code represented as f_{x0}, f_{x1},
    f_{x2}) with coefficients as polynomials in delta_x where delta_x = x_0 - c_0
    that represents the ''shifted ODE'' - i.e. the ODE where we substitute all
    occurences of delta_x with x_0 - c_0

    Description: Converts an ode in x, to a polynomial in f, f_x, f_{xx}, ...,
    with coefficients as polynomials in delta_x = x_0 - c_0.
    """
    #Note that generate_nd_derivative_relations will at worst put some power of
    #$x_0^order$ in the denominator. To clear
    #the denominator we can probably? just multiply by x_0^order.
    delta_x = sp.symbols("delta_x")
    c_vec = _make_sympy_vec("c", len(var))
    ode_in_x_cleared = (ode_in_x * var[0]**n_derivs).simplify()
    ode_in_x_shifted = ode_in_x_cleared.subs(var[0], delta_x + c_vec[0]).simplify()
    f_x_derivs = _make_sympy_vec("f_x", n_derivs)
    poly = sp.Poly(ode_in_x_shifted, *f_x_derivs)
    return poly


def compute_coefficients_of_poly(poly, n_derivs):
    """
    compute_coefficients_of_poly
    Input:
    - poly, a polynomial in sympy variables f_{x0}, f_{x1}, ...,
        (recall that this corresponds to f_0, f_x, f_{xx}, ...) with coefficients
        that are polynomials in delta_x where poly represents the ''shifted ODE''
        - i.e. we substitute all occurences of delta_x with x_0 - c_0
    Output:
    - a 2d array, each row giving the coefficient of f_0, f_x, f_{xx}, ...,
        each entry in the row giving the coefficients of the polynomial in delta_x
    Description: Takes in a polynomial in f_{x0}, f_{x1}, ..., w/coeffs that are
    polynomials in delta_x and outputs a 2d array for easy access to the
    coefficients based on their degree as a polynomial in delta_x.
    """
    delta_x = sp.symbols("delta_x")

    #Returns coefficients in lexographic order. So lowest order first
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
        coeffs.append(
            sp.Poly(poly.coeff_monomial(tup(deriv_ind)), delta_x).all_coeffs())

    return coeffs


def compute_recurrence_relation(coeffs, n_derivs, var):
    """
    compute_recurrence_relation
    Input:
    - coeffs a 2d array that gives access to the coefficients of poly, where poly
    represents the coefficients of the ''shifted ODE''
    (''shifted ODE'' = we substitute all occurences of delta_x with x_0 - c_0)
    based on their degree as a polynomial in delta_x
    - n_derivs, the order of the original PDE + 1, i.e. the number of derivatives
    of f that may be present
    Output:
    - a recurrence statement that equals 0 where s(i) is the ith coefficient of
    the Taylor polynomial for our point potential.

    Description: Takes in coeffs which represents our ``shifted ode in x"
    (i.e. ode_in_x with coefficients in delta_x) and outputs a recurrence relation
    for the point potential.
    """
    i = sp.symbols("i")
    s = sp.Function("s")
    c_vec = _make_sympy_vec("c", len(var))

    #Compute symbolic derivative
    def hc_diff(i, n):
        retme = 1
        for j in range(n):
            retme *= (i-j)
        return retme

    #We are differentiating deriv_ind, which shifts down deriv_ind.
    #Do this for one deriv_ind
    r = 0
    for deriv_ind in range(n_derivs):
        part_of_r = 0
        pow_delta = 0
        for j in range(len(coeffs[deriv_ind])-1, -1, -1):
            shift = pow_delta - deriv_ind + 1
            pow_delta += 1
            # pylint: disable=not-callable
            temp = coeffs[deriv_ind][j] * s(i) * hc_diff(i, deriv_ind)
            part_of_r += temp.subs(i, i-shift)
        r += part_of_r

    for j in range(1, len(var)):
        r = r.subs(var[j], c_vec[j])

    return r.simplify()


def get_recurrence_from_pde(pde):
    """
    Input:
        - *pde*, representing a scalar PDE.

    Output:
        - r, a recurrence relation for a Line-Taylor expansion.

    Description: Takes in a pde, outputs a recurrence.
    """
    ode_in_r, var, n_derivs = get_pde_in_recurrence_form(pde)
    ode_in_x = ode_in_r_to_x(ode_in_r, var, n_derivs).simplify()
    poly = compute_poly_in_deriv(ode_in_x, n_derivs, var)
    coeffs = compute_coefficients_of_poly(poly, n_derivs)
    r = compute_recurrence_relation(coeffs, n_derivs, var)
    return r


def test_recurrence_finder_laplace():
    """
    test_recurrence_finder_laplace
    Description: Checks that the recurrence finder works correctly for the Laplace
    2D point potential.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    r = get_recurrence_from_pde(laplace2d)
    i = sp.symbols("i")
    s = sp.Function("s")

    def coeff_laplace(i):
        x, y = sp.symbols("x,y")
        c_vec = _make_sympy_vec("c", 2)
        true_f = sp.log(sp.sqrt(x**2 + y**2))
        return sp.diff(true_f, x, i).subs(x, c_vec[0]).subs(
            y, c_vec[1])/math.factorial(i)
    d = 6
    # pylint: disable=not-callable
    val = r.subs(i, d).subs(s(d+1), coeff_laplace(d+1)).subs(
        s(d), coeff_laplace(d)).subs(s(d-1), coeff_laplace(d-1)).subs(
            s(d-2), coeff_laplace(d-2)).simplify()
    assert val == 0


def test_recurrence_finder_laplace_three_d():
    """
    test_recurrence_finder_laplace_three_d
    Description: Checks that the recurrence finder works correctly for the Laplace
    3D point potential.
    """
    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)
    print(laplace3d)
    r = get_recurrence_from_pde(laplace3d)
    i = sp.symbols("i")
    s = sp.Function("s")

    def coeff_laplace_three_d(i):
        x, y, z = sp.symbols("x,y,z")
        c_vec = _make_sympy_vec("c", 3)
        true_f = 1/(sp.sqrt(x**2 + y**2 + z**2))
        return sp.diff(true_f, x, i).subs(x, c_vec[0]).subs(
            y, c_vec[1]).subs(z, c_vec[2])/math.factorial(i)

    d = 6
    # pylint: disable=not-callable
    val = r.subs(i, d).subs(s(d+1), coeff_laplace_three_d(d+1)).subs(
        s(d), coeff_laplace_three_d(d)).subs(s(d-1),
            coeff_laplace_three_d(d-1)).subs(
            s(d-2), coeff_laplace_three_d(d-2)).simplify()

    assert val == 0
