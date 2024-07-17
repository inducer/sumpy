r"""
With the functionality in this module, we aim to compute a recurrence for
one-dimensional derivatives of functions :math:`f:\mathbb R^n \to \mathbb R`
for functions :math:`f` satisfying two assumptions:

- :math:`f` satisfies a PDE is linear and has coefficients polynomial
  in the coordinates.
- :math:`f` only depends on the radius :math:`r`,
  i.e. :math:`f(\boldsymbol x)=f(|\boldsymbol x|_2)`.

This process proceeds in multiple steps:

- Convert from the PDE to an ODE in :math:`r`, using :func:`pde_to_ode_in_r`.
- Convert from an ODE in :math:`r` to one in :math:`x`, using :func:`ode_in_r_to_x`.
- Sort general-form ODE in :math:`x` into a coefficient array, using
  :func:`ode_in_x_to_coeff_array`.
- Finally, get an expression for the recurrence, using
  :func:`recurrence_from_coeff_array`.

The whole process can be automated using :func:`recurrence_from_pde`.

.. autofunction:: pde_to_ode_in_r
.. autofunction:: ode_in_r_to_x
.. autofunction:: ode_in_x_to_coeff_array
.. autofunction:: recurrence_from_coeff_array
.. autofunction:: recurrence_from_pde
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
import numpy as np
import sympy as sp
from pytools.obj_array import make_obj_array
from sumpy.expansion.diff_op import (
    DerivativeIdentifier, make_identity_diff_op, laplacian, LinearPDESystemOperator)


# similar to make_sym_vector in sumpy.symbolic, but returns an object array
# instead of a sympy.Matrix.
def _make_sympy_vec(name, n):
    return make_obj_array([sp.Symbol(f"{name}{i}") for i in range(n)])


def pde_to_ode_in_r(pde: LinearPDESystemOperator) -> tuple[
        sp.Expr, np.ndarray, int
]:
    r"""
    Returns an ODE satisfied by the radial derivatives of a function
    :math:`f:\mathbb R^n \to \mathbb R` satisfying
    :math:`f(\boldsymbol x)=f(|\boldsymbol x|_2)` and *pde*.

    :arg pde: must satisfy ``pde.eqs == 1`` and have polynomial coefficients.

    :returns: a tuple ``(ode_in_r, var, ode_order)``, where
    - *ode_in_r* with derivatives given as :class:`sympy.Derivative`.
    - *var* is an object array of :class:`sympy.Symbol`, with successive variables
      representing the Cartesian coordinate directions.
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

    def apply_deriv_id(expr: sp.Expr, deriv_id: DerivativeIdentifier) -> sp.Expr:
        for i, nderivs in enumerate(deriv_id.mi):
            expr = expr.diff(var[i], nderivs)
        return expr
    # pylint: disable-next=not-callable
    ode_in_r = sum(
        coeff * apply_deriv_id(f(rval), deriv_id)
        for deriv_id, coeff in pde_eqn.items()
    )

    f_r_derivs = _make_sympy_vec("f_r", ode_order+1)
    # pylint: disable-next=not-callable
    f_derivs = [sp.diff(f(rval), eps, i) for i in range(ode_order+1)]

    # PDE ORDER = ODE ORDER
    for i in range(ode_order+1):
        ode_in_r = ode_in_r.subs(f_derivs[i], f_r_derivs[i])

    return ode_in_r, var, ode_order


def _generate_nd_derivative_relations(var: np.ndarray, ode_order: int) -> dict:
    r"""
    Using the chain rule outputs a vector that gives in each component respectively
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


def ode_in_r_to_x(ode_in_r: sp.Expr, var: np.ndarray, ode_order: int) -> sp.Expr:
    r"""
    Translates an ode in the variable r into an ode in the variable x
    by replcaing the terms :math:`f, f_r, f_{rr}, \dots` as a linear combinations of
    :math:`f, f_x, f_{xx}, \dots` using the chain rule.

    :arg ode_in_r: a linear combination of :math:`f, f_r, f_{rr}, \dots` represented
        by the sympy variables :math:`f_{r0}, f_{r1}, f_{r1}, f_{r2}, \dots`
    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    :arg ode_order: the order of the input ODE

    :returns: *ode_in_x* a linear combination of :math:`f, f_x, f_{xx}, \dots` with 
        coefficients as rational functions in :math:`x_0, x_1, \dots`
    """
    subme = _generate_nd_derivative_relations(var, ode_order+1)
    ode_in_x = ode_in_r
    f_r_derivs = _make_sympy_vec("f_r", ode_order+1)
    for i in range(ode_order+1):
        ode_in_x = ode_in_x.subs(f_r_derivs[i], subme[f_r_derivs[i]])
    return ode_in_x


def ode_in_x_to_coeff_array(poly: sp.Poly, ode_order: int,
                                            var: np.ndarray) -> list:
    """
    ## Input
        - *poly*, the original ODE for our point-potential as a polynomial
          in f_{x0}, f_{x1}, f_{x2}, etc. with polynomial coefficients
          in var[0], var[1], ...
        - *ode_order*, the order of input ODE
        - *var*, array of sympy variables [x_0, x_1, ...]
    ## Output
        - ode_in_x, a linear combination of f, f_x, f_{xx}, ... with coefficients as
          rational functions in var[0], var[1], ...
    ## Description
    Translates an ode in the variable r into an ode in the variable x
    by substituting f, f_r, f_{rr}, ... as a linear combination of
    f, f_x, f_{xx}, ... using the chain rule.
    """
    def tup(i, n=ode_order+1):
        a = []
        for j in range(n):
            if j != i:
                a.append(0)
            else:
                a.append(1)
        return tuple(a)

    coeffs = []
    for deriv_ind in range(ode_order+1):
        coeffs.append(sp.Poly(poly.coeff_monomial(tup(deriv_ind)),
                              var[0]).all_coeffs()[::-1])

    return coeffs


def _auto_product_rule_single_term(p: int, m: int, var: np.ndarray) -> sp.Expr:
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


def recurrence_from_coeff_array(coeffs: list, var: np.ndarray) -> sp.Expr:
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
            final_recurrence += coeffs[m][p] * _auto_product_rule_single_term(p,
                                                                             m, var)
    return final_recurrence


def recurrence_from_pde(pde: LinearPDESystemOperator) -> sp.Expr:
    """
    ## Input
    - *pde*, a :class:`sumpy.expansion.diff_op.LinearSystemPDEOperator` such that
      pde.eqs == 1
    ## Output
    - final_recurrence, the recurrence relation for derivatives of our
      point-potential.
    """
    ode_in_r, var, ode_order = pde_to_ode_in_r(pde)
    ode_in_x = ode_in_r_to_x(ode_in_r, var, ode_order).simplify()
    ode_in_x_cleared = (ode_in_x * var[0]**(ode_order+1)).simplify()
    f_x_derivs = _make_sympy_vec("f_x", ode_order+1)
    poly = sp.Poly(ode_in_x_cleared, *f_x_derivs)
    coeffs = ode_in_x_to_coeff_array(poly, ode_order, var)
    return recurrence_from_coeff_array(coeffs, var)


def test_recurrence_finder_laplace():
    """
    ## Description
    Tests our recurrence relation generator for Lapalace 2D.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    r = recurrence_from_pde(laplace2d)
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
    r = recurrence_from_pde(laplace3d)
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


def test_recurrence_finder_helmholtz_three_d():
    """
    ## Description
    Tests our recurrence relation generator for Helmhotlz 3D.
    """
    #We are creating the recurrence relation for helmholtz3d which 
    #seems to be an order 5 recurrence relation
    w = make_identity_diff_op(3)
    helmholtz3d = laplacian(w) + w
    r = recurrence_from_pde(helmholtz3d)

    #We create that function that gives the derivatives of the point 
    # potential for helmholtz
    #Remember! Our point-source was placed at the origin and we 
    # were performing a LT expansion at x_0
    def deriv_helmholtz_three_d(i, s_loc):
        s_x = s_loc[0]
        s_y = s_loc[1]
        s_z = s_loc[2]
        x, y, z = sp.symbols("x,y,z")
        true_f = sp.exp(1j * sp.sqrt(x**2 + y**2 + z**2)
                        ) / (sp.sqrt(x**2 + y**2 + z**2))
        return sp.diff(true_f, x, i).subs(x, s_x).subs(
            y, s_y).subs(z, s_z)
    
    #Create relevant symbols
    var = _make_sympy_vec("x", 3)
    n = sp.symbols("n")
    s = sp.Function("s")

    #Create random source location
    s_loc = np.random.rand(3)

    #Create random order to check
    d = np.random.randint(0, 5)

    #Substitute random location into recurrence relation and value of n = d
    r_loc = r.subs(var[0], s_loc[0])
    r_loc = r_loc.subs(var[1], s_loc[1])
    r_loc = r_loc.subs(var[2], s_loc[2])
    r_sub = r_loc.subs(n, d)

    #Checking that the recurrence holds to some machine epsilon
    for i in range(max(d-3, 0), d+3):
        # pylint: disable=not-callable
        r_sub = r_sub.subs(s(i), deriv_helmholtz_three_d(i, s_loc))
    err = abs(abs(r_sub).evalf())
    print(err)
    assert err <= 1e-10

test_recurrence_finder_laplace()
test_recurrence_finder_laplace_three_d()
test_recurrence_finder_helmholtz_three_d()

