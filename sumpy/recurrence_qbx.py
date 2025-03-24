r"""
With the functionality in this module, we compute layer potentials
using a recurrence for one-dimensional derivatives of the corresponding
Green's function. See recurrence.py.

.. autofunction:: recurrence_qbx_lp
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
from typing import Sequence

import numpy as np
import sympy as sp

from sumpy.recurrence import (
    _make_sympy_vec,
    get_off_axis_expression,
    get_reindexed_and_center_origin_off_axis_recurrence,
    get_reindexed_and_center_origin_on_axis_recurrence,
)


# ================ Transform/Rotate =================
def _produce_orthogonal_basis(normals: np.ndarray) -> Sequence[np.ndarray]:
    ndim, ncenters = normals.shape
    orth_coordsys = [normals]
    for i in range(1, ndim):
        v = np.random.rand(ndim, ncenters)  # noqa: NPY002
        v = v/np.linalg.norm(v, 2, axis=0)
        for j in range(i):
            v = v - np.einsum("dc,dc->c", v,
                              orth_coordsys[j]) * orth_coordsys[j]
        v = v/np.linalg.norm(v, 2, axis=0)
        orth_coordsys.append(v)

    return orth_coordsys


def _compute_rotated_shifted_coordinates(
            sources: np.ndarray,
            centers: np.ndarray,
            normals: np.ndarray
        ) -> np.ndarray:
    cts = sources[:, None] - centers[:, :, None]
    orth_coordsys = _produce_orthogonal_basis(normals)
    cts_rotated_shifted = np.einsum("idc,dcs->ics", orth_coordsys, cts)

    return cts_rotated_shifted


# ================ Recurrence LP Eval =================
def recurrence_qbx_lp(sources, centers, normals, strengths, radius, pde, g_x_y,
                      ndim, p) -> np.ndarray:
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
    :arg ndim: number of spatial variables
    :arg p: order of expansion computed
    """

    # ------------- 2. Compute rotated/shifted coordinates
    cts_r_s = _compute_rotated_shifted_coordinates(sources, centers, normals)

    # ------------- 4. Define input variables for green's function expression
    var = _make_sympy_vec("x", ndim)
    var_t = _make_sympy_vec("t", ndim)

    # ------------ 5. Compute recurrence
    n_initial, order, recurrence = get_reindexed_and_center_origin_on_axis_recurrence(pde)

    # ------------ 6. Set order p = 5
    n_p = sources.shape[1]
    storage = [np.zeros((n_p, n_p))] * order

    s = sp.Function("s")
    n = sp.symbols("n")

    def generate_lamb_expr(i, n_initial):
        arg_list = []
        for j in range(order, 0, -1):
            # pylint: disable-next=not-callable
            arg_list.append(s(i-j))
        for j in range(ndim):
            arg_list.append(var[j])

        if i < n_initial:
            lamb_expr_symb_deriv = sp.diff(g_x_y, var_t[0], i)
            for j in range(ndim):
                lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
            lamb_expr_symb = lamb_expr_symb_deriv
        else:
            lamb_expr_symb = recurrence.subs(n, i)
        #print("=============== ORDER = " + str(i))
        #print(lamb_expr_symb)
        return sp.lambdify(arg_list, lamb_expr_symb)#, sp.lambdify(arg_list, lamb_expr_symb_deriv)

    interactions_on_axis = 0
    coord = [cts_r_s[j] for j in range(ndim)]
    for i in range(p+1):
        lamb_expr = generate_lamb_expr(i, n_initial)
        a = [*storage, *coord]
        s_new = lamb_expr(*a)

        """
        s_new_true = true_lamb_expr(*a)
        arg_max = np.argmax(abs(s_new-s_new_true)/abs(s_new_true))
        print((s_new-s_new_true).reshape(-1)[arg_max]/s_new_true.reshape(-1)[arg_max])
        print("x:", coord[0].reshape(-1)[arg_max], "y:", coord[1].reshape(-1)[arg_max],
              "s_recur:", s_new.reshape(-1)[arg_max], "s_true:", s_new_true.reshape(-1)[arg_max], "order: ", i) 
        """

        interactions_on_axis += s_new * radius**i/math.factorial(i)

        storage.pop(0)
        storage.append(s_new)


    ### NEW CODE - COMPUTE OFF AXIS INTERACTIONS
    start_order, t_recur_order, t_recur = get_reindexed_and_center_origin_off_axis_recurrence(pde)
    t_exp, t_exp_order = get_off_axis_expression(pde)
    storage_taylor_order = max(t_recur_order, t_exp_order+1)

    storage_taylor = [np.zeros((n_p, n_p))] * storage_taylor_order

    def gen_lamb_expr_t_recur(i, start_order):
        arg_list = []
        for j in range(t_recur_order, 0, -1):
            # pylint: disable-next=not-callable
            arg_list.append(s(i-j))
        for j in range(ndim):
            arg_list.append(var[j])

        if i < start_order:
            lamb_expr_symb_deriv = sp.diff(g_x_y, var_t[0], i)
            for j in range(ndim):
                lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
            lamb_expr_symb = lamb_expr_symb_deriv.subs(var[0], 0)
        else:
            lamb_expr_symb = t_recur.subs(n, i)

        return sp.lambdify(arg_list, lamb_expr_symb)


    def gen_lamb_expr_t_exp(i, t_exp_order):
        arg_list = []
        for j in range(t_exp_order, -1, -1):
            # pylint: disable-next=not-callable
            arg_list.append(s(i-j))
        for j in range(ndim):
            arg_list.append(var[j])

        if i < t_exp_order:
            lamb_expr_symb_deriv = sp.diff(g_x_y, var_t[0], i)
            for j in range(ndim):
                lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
            lamb_expr_symb = lamb_expr_symb_deriv
        else:
            lamb_expr_symb = t_exp.subs(n, i)

        return sp.lambdify(arg_list, lamb_expr_symb)

    interactions_off_axis = 0
    for i in range(p+1):
        lamb_expr_t_recur = gen_lamb_expr_t_recur(i, start_order)
        a1 = [*storage_taylor[(-t_recur_order):], *coord]

        storage.pop(0)
        storage.append(lamb_expr_t_recur(*a1))

        lamb_expr_t_exp = gen_lamb_expr_t_exp(i, t_exp_order)
        a2 = [*storage_taylor[-(t_exp_order+1):], *coord]

        interactions_off_axis += lamb_expr_t_exp(*a2) * radius**i/math.factorial(i)

    ################
    # Compute True Interactions
    def generate_true(i):
        arg_list = []
        for j in range(ndim):
            arg_list.append(var[j])

        lamb_expr_symb_deriv = sp.diff(g_x_y, var_t[0], i)
        for j in range(ndim):
            lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
        lamb_expr_symb = lamb_expr_symb_deriv

        #print("=============== ORDER = " + str(i))
        #print(lamb_expr_symb)
        return sp.lambdify(arg_list, lamb_expr_symb)#, sp.lambdify(arg_list, lamb_expr_symb_deriv)

    interactions_true = 0
    for i in range(p+1):
        lamb_expr_true = generate_true(i)
        a4 = [*coord]
        s_new_true = lamb_expr_true(*a4)
        interactions_true += s_new_true * radius**i/math.factorial(i)
    ###############

    #slope of line y = mx
    m = 1e5/2 
    mask_on_axis = m*np.abs(coord[0]) >= np.abs(coord[1])
    mask_off_axis = m*np.abs(coord[0]) < np.abs(coord[1])

    print("-------------------------")

    percent_on = np.sum(mask_on_axis)/(mask_on_axis.shape[0]*mask_on_axis.shape[1])
    percent_off = 1-percent_on

    relerr_on = np.abs(interactions_on_axis[mask_on_axis]-interactions_true[mask_on_axis])/np.abs(interactions_on_axis[mask_on_axis])
    print("MAX ON AXIS ERROR(", percent_on, "):", np.max(relerr_on))
    print(np.mean(relerr_on))
    print("X:", coord[0].reshape(-1)[np.argmax(relerr_on)])
    print("Y:", coord[1].reshape(-1)[np.argmax(relerr_on)])

    print("-------------------------")

    if np.any(mask_off_axis):
        relerr_off = np.abs(interactions_off_axis[mask_off_axis]-interactions_true[mask_off_axis])/np.abs(interactions_off_axis[mask_off_axis])
        print("MAX OFF AXIS ERROR(", percent_off, "):", np.max(relerr_off))
        print(np.mean(relerr_off))
        print("X:", coord[0].reshape(-1)[np.argmax(relerr_off)])
        print("Y:", coord[1].reshape(-1)[np.argmax(relerr_off)])

    interactions_total = np.zeros(coord[0].shape)
    interactions_total[mask_on_axis] = interactions_on_axis[mask_on_axis]
    interactions_total[mask_off_axis] = interactions_off_axis[mask_off_axis]

    exp_res = (interactions_total * strengths[None, :]).sum(axis=1)
    exp_res_true = (interactions_true * strengths[None, :]).sum(axis=1)

    relerr_total = np.max(np.abs(exp_res-exp_res_true)/np.abs(exp_res_true))
    print("OVERALL ERROR:", relerr_total)

    return exp_res