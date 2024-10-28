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

.. autofunction:: pde_to_ode_in_r
.. autofunction:: ode_in_r_to_x
.. autofunction:: ode_in_x_to_coeff_array
.. autofunction:: recurrence_from_coeff_array
.. autofunction:: recurrence_from_pde
"""

from __future__ import annotations

from typing import TypeVar


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

from random import randrange
import numpy as np
import sympy as sp

from pytools.obj_array import make_obj_array


from sumpy.expansion.diff_op import (
    DerivativeIdentifier,
    LinearPDESystemOperator
)


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
    - *ode_in_r* with derivatives given as :class:`sympy.Derivative`
    - *var* is an object array of :class:`sympy.Symbol`, with successive
      variables
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

    def apply_deriv_id(expr: sp.Expr,
                       deriv_id: DerivativeIdentifier) -> sp.Expr:
        for i, nderivs in enumerate(deriv_id.mi):
            expr = expr.diff(var[i], nderivs)
        return expr

    ode_in_r = sum(
        # pylint: disable-next=not-callable
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


def ode_in_r_to_x(ode_in_r: sp.Expr, var: np.ndarray,
                  ode_order: int) -> sp.Expr:
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


ODECoefficients = list[list[sp.Expr]]


def ode_in_x_to_coeff_array(poly: sp.Poly, ode_order: int,  var:
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


def _auto_product_rule_single_term(p: int, m: int, var: np.ndarray) -> sp.Expr:
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

    return sum(
        # pylint: disable=not-callable
        _falling_factorial(n, i)
        * math.comb(p, i) * s(n-i+m) * var[0]**(p-i)
        for i in range(p+1)
    )


def recurrence_from_coeff_array(coeffs: list, var: np.ndarray) -> sp.Expr:
    r"""
    A function that takes in as input an organized 2D coefficient array (see
    above) and outputs a recurrence relation.

    :arg coeffs: a sequence of of sequences, described in
        :func:`ode_in_x_to_coeff_array`
    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    """
    final_recurrence = 0
    # Outer loop is derivative direction
    # Inner is polynomial order of x_0
    for m, _ in enumerate(coeffs):
        for p, _ in enumerate(coeffs[m]):
            final_recurrence += coeffs[m][p] * _auto_product_rule_single_term(
                p, m, var)
    return final_recurrence


def recurrence_from_pde(pde: LinearPDESystemOperator) -> sp.Expr:
    r"""
    A function that takes in as input a sympy PDE and outputs a recurrence
    relation.

    :arg pde: a :class:`sumpy.expansion.diff_op.LinearSystemPDEOperator`
        that must satisfy ``pde.eqs == 1`` and have polynomial coefficients
        in the coordinates.
    :arg var: array of sympy variables :math:`[x_0, x_1, \dots]`
    """
    ode_in_r, var, ode_order = pde_to_ode_in_r(pde)
    ode_in_x = ode_in_r_to_x(ode_in_r, var, ode_order).simplify()
    ode_in_x_cleared = (ode_in_x * var[0]**(ode_order+1)).simplify()
    # ode_in_x_cleared shouldn't have rational function coefficients
    assert sp.together(ode_in_x_cleared) == ode_in_x_cleared
    f_x_derivs = _make_sympy_vec("f_x", ode_order+1)
    poly = sp.Poly(ode_in_x_cleared, *f_x_derivs)
    coeffs = ode_in_x_to_coeff_array(poly, ode_order, var)
    return recurrence_from_coeff_array(coeffs, var)


def process_recurrence_relation(r: sp.Expr,
                                replace=True) -> tuple[int, sp.Expr]:
    r"""
    A function that takes in as input a recurrence and outputs a recurrence
    relation that has the nth term in terms of the n-1th, n-2th etc.
    Also returns the order of the recurrence relation.

    If replace=True then the recurrence is output in a form that is ideal
    for pymbolic processing. If replace=False then a standard recurrence 
    is output.

    :arg recurrence: a recurrence relation in :math:`s(n)`
    """
    terms = list(r.atoms(sp.Function))
    terms = np.array(terms)

    # Sort terms and create idx_l
    idx_l = []
    for i in range(len(terms)):
        tms = list(terms[i].atoms(sp.Number))
        if len(tms) == 1:
            idx_l.append(tms[0])
        else:
            idx_l.append(0)
    idx_l = np.array(idx_l, dtype='int')
    idx_sort = idx_l.argsort()
    idx_l = idx_l[idx_sort]
    terms = terms[idx_sort]

    # Order is the max difference between highest/lowest in idx_l
    order = max(idx_l) - min(idx_l) + 1

    # How much do we need to shift the recurrence relation
    shift_idx = max(idx_l)

    # Get the respective coefficients in the recurrence relation from r
    n = sp.symbols("n")
    s = sp.Function("s")
    coeffs = sp.poly(r, list(terms)).coeffs()

    # Re-arrange the recurrence relation so we get s(n) = ____
    # in terms of s(n-1), ...
    true_recurrence = sum([coeffs[i]/coeffs[-1] * terms[i]
                           for i in range(0, len(terms)-1)])
    true_recurrence1 = true_recurrence.subs(n, n-shift_idx)

    if replace:
        # Replace s(n-1) with snm_1, s(n-2) with snm_2 etc.
        # because pymbolic.substitute won't recognize it
        last_syms = [sp.Symbol(f"anm{i+1}") for i in range(order-1)]
        # pylint: disable=not-callable
        # Assumes order > 1
        true_recurrence2 = true_recurrence1.subs(s(n-1), last_syms[0])
        for i in range(2, order):
            true_recurrence2 = true_recurrence2.subs(s(n-i), last_syms[i-1])
        return order, true_recurrence2

    return order, true_recurrence1


def extract_idx_terms_from_recurrence(r: sp.Expr) -> tuple[np.ndarray,
                                                           np.ndarray]:
    r"""
    Given a recurrence extracts the variables in the recurrence
    as well as the indexes in sorted order.

    :arg r: recurrence to extract terms from
    """
    terms = list(r.atoms(sp.Function))
    terms = np.array(terms)


    idx_l = []
    for i in range(len(terms)):
        tms = list(terms[i].atoms(sp.Number))
        if len(tms) == 1:
            idx_l.append(tms[0])
        else:
            idx_l.append(0)
    idx_l = np.array(idx_l, dtype='int')
    idx_sort = idx_l.argsort()
    idx_l = idx_l[idx_sort]
    terms = terms[idx_sort]

    return idx_l, terms


def __check_neg_ind(r_n):
    r"""
    Simply checks if a negative index exists in a recurrence relation.
    """

    idx_l, _ = extract_idx_terms_from_recurrence(r_n)

    return np.any(idx_l < 0)


def __get_initial_c(recurrence):
    r"""
    For a given recurrence checks how many initial conditions by
    checking for non-negative indexed terms.
    """
    n = sp.symbols("n")

    i = 0
    r_c = recurrence.subs(n, i)
    while __check_neg_ind(r_c):
        i += 1
        r_c = recurrence.subs(n, i)
    return i


def shift_recurrence(r: sp.Expr, var: np.ndarray) -> sp.Expr:
    r"""
    A function that "shifts" the recurrence so it's center is placed
    at the origin and source is the input for the recurrence generated.

    :arg recurrence: a recurrence relation in :math:`s(n)`
    """
    idx_l, terms = extract_idx_terms_from_recurrence(r)

    r_ret = r

    n = sp.symbols('n')
    for i in range(len(idx_l)):
        r_ret = r_ret.subs(terms[i], (-1)**(n+idx_l[i])*terms[i])

    return r_ret*((-1)**(n+1))


def get_processed_recurrence_from_pde_shift(pde, ndim) -> tuple[int, int,
                                                                sp.Expr]:
    r"""
    A function that "shifts" the recurrence so the expansion center is placed
    at the origin and source is the input for the recurrence generated.

    :arg recurrence: a recurrence relation in :math:`s(n)`
    """
    r = recurrence_from_pde(pde)
    var = _make_sympy_vec("x", ndim)
    order, r_p = process_recurrence_relation(r, False)
    n_initial = __get_initial_c(r_p)
    r_s = shift_recurrence(r_p, var)
    return n_initial, order, r_s


# ================ Transform/Rotate =================
def __produce_orthogonal_basis(normals):
    ndim, ncenters = normals.shape
    orth_coordsys = [normals]
    for i in range(1, ndim):
        v = np.random.rand(ndim, ncenters)
        v = v/np.linalg.norm(v, 2, axis=0)
        for j in range(i):
            v = v - np.einsum("dc,dc->c", v, orth_coordsys[j]) * orth_coordsys[j]
        v = v/np.linalg.norm(v, 2, axis=0)
        orth_coordsys.append(v)

    return orth_coordsys


def __compute_rotated_shifted_coordinates(sources, centers, normals):

    cts = sources[:, None] - centers[:, :, None]
    orth_coordsys = __produce_orthogonal_basis(normals)
    cts_rotated_shifted = np.einsum("idc,dcs->ics", orth_coordsys, cts)

    return cts_rotated_shifted


# ================ Recurrence LP Eval =================
def recurrence_qbx_lp(sources, centers, normals, strengths, radius, pde, g_x_y,
                       p) -> np.ndarray:
    r"""
    A function that computes a single-layer potential using a recurrence.

    :arg sources: a (ndim, nsources) array of source locations
    :arg centers: a (ndim, ncenters) array of center locations
    :arg normals: a (ndim, ncenters) array of normals
    :arg strengths: array corresponding to quadrature weight multiplied by 
    density
    :arg radius: expansion radius
    :arg pde: pde that we are computing layer potential for
    :arg g_x_y: a green's function in (x0, x1, ...) source and 
    (t0, t1, ...) target
    :arg p: order of expansion computed
    """

    #------------- 2. Compute rotated/shifted coordinates
    cts_r_s = __compute_rotated_shifted_coordinates(sources, centers, normals)


    #------------- 4. Compute green's function expression
    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)

    #------------ 5. Compute recurrence
    n_initial, order, recurrence = get_processed_recurrence_from_pde_shift(pde, ndim=2)

    #------------ 6. Set order p = 5
    n_p = sources.shape[1]
    storage = [np.zeros((n_p,n_p))] * order 

    s = sp.Function("s")
    r,n = sp.symbols("r,n")

    def generate_lamb_expr(i, n_initial):
        arg_list = []
        for j in range(order,0,-1):
            arg_list.append(s(i-j))
        arg_list.append(var[0])
        arg_list.append(var[1])
        arg_list.append(r)
        
        if i < n_initial:
            lamb_expr = sp.diff(g_x_y, var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
        else:
            lamb_expr = recurrence.subs(n, i)
        return sp.lambdify(arg_list, lamb_expr) 

    interactions_2d = 0
    for i in range(p+1):
        lamb_expr = generate_lamb_expr(i, n_initial)
        a = storage[-4:] + [cts_r_s[0],cts_r_s[1],radius]
        s_new = lamb_expr(*a)
        interactions_2d += s_new * radius**i/math.factorial(i)

        storage.pop(0)
        storage.append(s_new)

    exp_res = (interactions_2d * strengths[None, :]).sum(axis=1)

    return exp_res


# TEST CODE
from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)

import numpy as np
from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf  # noqa: F401
from sumpy.expansion.local import LineTaylorLocalExpansion, VolumeTaylorLocalExpansion


actx_factory = _acf
expn_class = LineTaylorLocalExpansion

actx = actx_factory()

from sumpy.kernel import LaplaceKernel
lknl = LaplaceKernel(2)

from sumpy.qbx import LayerPotential

def qbx_lp_laplace_general(sources,targets,centers,radius,strengths,order):
        lpot = LayerPotential(actx.context,
        expansion=expn_class(lknl, order),
        target_kernels=(lknl,),
        source_kernels=(lknl,))

        #print(lpot.get_kernel())
        expansion_radii = actx.from_numpy(radius * np.ones(sources.shape[1]))
        sources = actx.from_numpy(sources)
        targets = actx.from_numpy(targets)
        centers = actx.from_numpy(centers)

        strengths = (strengths,)

        _evt, (result_qbx,) = lpot(
                actx.queue,
                targets, sources, centers, strengths,
                expansion_radii=expansion_radii)
        result_qbx = actx.to_numpy(result_qbx)

        return result_qbx

def create_ellipse(n_p):
    h = 9.688 / n_p
    radius = 7*h
    t = np.linspace(0, 2 * np.pi, n_p, endpoint=False)

    unit_circle_param = np.exp(1j * t)
    unit_circle = np.array([2 * unit_circle_param.real, unit_circle_param.imag])

    sources = unit_circle
    normals = np.array([unit_circle_param.real, 2*unit_circle_param.imag])
    normals = normals / np.linalg.norm(normals, axis=0)
    centers = sources - normals * radius

    mode_nr = 25
    density = np.cos(mode_nr * t)

    return sources, centers, normals, density, h, radius

def test_recurrence_laplace_2d_ellipse():

    #------------- 1. Define PDE, Green's Function
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = (-1/(2*np.pi)) * sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))

    p = 4
    err = []
    for n_p in range(200, 1001, 200):
        sources, centers, normals, density, h, radius = create_ellipse(n_p)
        strengths = h * density
        exp_res = recurrence_qbx_lp(sources, centers, normals, strengths, radius, laplace2d, g_x_y, p)
        qbx_res = qbx_lp_laplace_general(sources, sources, centers, radius, strengths, p)
        #qbx_res,_ = lpot_eval_circle(sources.shape[1], p)
        err.append(np.max(exp_res - qbx_res))

    print(err)

test_recurrence_laplace_2d_ellipse()