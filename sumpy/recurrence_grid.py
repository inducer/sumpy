r"""
With the functionality in this module, we aim to compute a recurrence for
one-dimensional derivatives of functions :math:`f:\mathbb R^n \to \mathbb R`
for functions satisfying two assumptions:

- :math:`f` satisfies a PDE that is linear and has coefficients polynomial
  in the coordinates.
- :math:`f` only depends on the radius :math:`r`,
  i.e. :math:`f(\boldsymbol x)=f(|\boldsymbol x|_2)`.

  However, unlike recurrence.py, the recurrences produced here are numerically
  stable in a different source-location space.

.. autofunction:: get_grid
.. autofunction:: convert
.. autofunction:: grid_recur_to_column_recur
.. autofunction:: get_taylor_recurrence
.. autofunction:: create_subs_grid
.. autofunction:: extend_grid
.. autofunction:: compute_taylor_lp
.. autofunction:: compute_lp_orders


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


from sumpy.recurrence import _make_sympy_vec, get_processed_and_shifted_recurrence

from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)

from sumpy.recurrence import get_recurrence, recurrence_from_pde, shift_recurrence, get_shifted_recurrence_exp_from_pde, _extract_idx_terms_from_recurrence

import sympy as sp
from sympy import hankel1

import numpy as np

import math

import matplotlib.pyplot as plt
from matplotlib import cm, ticker


def get_grid(recur_exp, order):
    r"""
    Organizes the coefficients of recur into a 2D array, called a grid.

    :arg recur_exp: A recurrence expression for derivatives s(n), etc. where s(n)
    represents the nth derivative of the Green's function w/respect to the target at
    the origin.
    recur_exp looks like :math:`(b_{00} x_0^0 + b_{01} x_0^1 + \cdots) s(n) +
    (b_{10} x_0^0 + b_{11} x_0^1 +\cdots) s(n-1) + \cdots`

    :arg order: The order of the input recurrence expression

    :returns: *table* a sequence of of sequences, with the outer sequence
        iterating over s(n), s(n-1),.. and each inner sequence iterating
        over powers of :math:`x_0`, so that, in terms of the above form,
        coeffs is :math:`[[b_{00}, b_{01}, ...], [b_{10}, b_{11}, ...], ...]`
    """
    var = _make_sympy_vec("x", 2)
    s = sp.Function("s")
    n = sp.symbols("n")
    i = sp.symbols("i")

    poly_in_s_n = sp.poly(recur_exp, [s(n-i) for i in range(order)])
    coeff_s_n = [poly_in_s_n.coeff_monomial(poly_in_s_n.gens[i]) for i in range(order)]

    table = []
    for i in range(len(coeff_s_n)):
        table.append(sp.poly(coeff_s_n[i], var[0]).all_coeffs()[::-1])

    return table


def convert(grid):
    r"""
    Given a grid of coefficients, produce a grid recurrence. Suppose that
    :math:`s(n) = \sum_i s(n,i) x_0^i`. A grid recurrence is an expression
    involving s(n,i) instead of s(n).

    :arg grid: The coefficients of a recurrence expression organized into a grid
        see :func:`get_grid`

    :returns: a tuple ``(recur_exp, s_terms)``, where
        - *grid_recur_exp* a grid recurrence for terms s(n,i)
        - *s_terms* are the terms s(n,i) that exist in recur_exp
    """
    s = sp.Function("s")
    n = sp.symbols("n")
    i = sp.symbols("i")

    grid_recur_exp = 0
    i = sp.symbols("i")
    s_terms = []
    for j in range(len(grid)):
        for k in range(len(grid[j])):
            grid_recur_exp += grid[j][k] * s(n-j,i-k)/sp.factorial(i-k)
            if grid[j][k] != 0:
                s_terms.append((j,k))
    return grid_recur_exp, s_terms


def grid_recur_to_column_recur(grid_recur, s_terms):
    r"""
    Given a grid recurrence, produce a recurrence that only involves
    terms of the form s(n,i), s(n-1,i), ..., s(n-k,i).

    :arg grid_recur: A grid recurrence see :func:`get_grid`
    :arg s_terms: The s(i,j) terms in grid_recur
    """
    s = sp.Function("s")
    n = sp.symbols("n")
    i = sp.symbols("i")

    grid_recur_simp = grid_recur
    bag = set()
    for s_t in s_terms:
        bag.add(-((0-s_t[0])-s_t[1]))
        grid_recur_simp = grid_recur_simp.subs(s(n-s_t[0],i-s_t[1]), (-1)**(s_t[1])*s((n-s_t[0])-s_t[1],(i-s_t[1])+s_t[1]))
    shift = min(bag)
    return sp.solve(sp.simplify(grid_recur_simp * sp.factorial(i)).subs(n, n+shift), s(n,i))[0]


def get_taylor_recurrence(pde):
    recur, order = get_shifted_recurrence_exp_from_pde(pde)
    grid = get_grid(recur, order)
    grid_recur, s_terms = convert(grid)
    column_recur = grid_recur_to_column_recur(grid_recur, s_terms)
    return column_recur


def create_subs_grid(width, length, derivs, coord_dict):
    var = _make_sympy_vec("x", 2)
    initial_grid = [[sp.diff(derivs[i], var[0], j).subs(var[0], 0) for j in range(width)] for i in range(length)]

    # assume len(initial_grid) >= 1
    initial_grid_subs = []
    initial_grid_width = len(initial_grid[0])
    initial_grid_length = len(initial_grid)

    for i_x in range(initial_grid_length):
        tmp = []
        for j_x in range(initial_grid_width):
            tmp.append((initial_grid[i_x][j_x].subs(var[1],coord_dict[var[1]])).evalf())
        initial_grid_subs.append(tmp)
    
    return initial_grid_subs


def extend_grid(initial_grid_in, grid_recur, coord_dict, n_derivs_compute, order_grid_recur, derivs):
    initial_grid_subs = [row[:] for row in initial_grid_in] #deep copy

    initial_grid_width = len(initial_grid_subs[0])
    initial_grid_length = len(initial_grid_subs)

    var = _make_sympy_vec("x", 2)
    s = sp.Function("s")
    n = sp.symbols("n")
    i = sp.symbols("i")

    for n_x in range(initial_grid_length, n_derivs_compute):
        appMe = []
        for i_x in range(initial_grid_width):
            exp_i_n = grid_recur.subs(n, n_x).subs(i, i_x)
            if exp_i_n == 0:
                exp_i_n = sp.diff(derivs[n_x], var[0], i_x).subs(var[0], 0)
            assert n_x-order_grid_recur >= 0
            kys = [s(n_x-k,i_x) for k in range(1,order_grid_recur+1)]
            vals = [initial_grid_subs[n_x-k][i_x] for k in range(1, order_grid_recur+1)]
            my_dict = dict(zip(kys, vals))
            res = exp_i_n.subs(my_dict).subs(coord_dict)
            appMe.append(res)

        initial_grid_subs.append(appMe)

    return initial_grid_subs


def compute_taylor_lp(inp_grid, coord_dict):
    var = _make_sympy_vec("x", 2)
    inp_grid = np.array(inp_grid)
    _, c = inp_grid.shape
    return np.sum(inp_grid * np.reshape(np.array([coord_dict[var[0]]**i/math.factorial(i) for i in range(c)]), (1, c)), axis = 1)


def compute_lp_orders(pde, loc, num_of_derivs, derivs_list, recur_order, taylor_order):
    var = _make_sympy_vec("x", 2)
    coord_dict_t = {var[0]: loc[0], var[1]: loc[1]}

    initial_grid_subs = create_subs_grid(taylor_order, recur_order, derivs_list, coord_dict_t)

    extended_grid = extend_grid(initial_grid_subs, get_taylor_recurrence(pde), coord_dict_t, num_of_derivs, recur_order, derivs_list)

    return compute_taylor_lp(extended_grid, coord_dict_t)