r"""
With the functionality in this module, we aim to compute layer potentials
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

from sumpy.recurrence import _make_sympy_vec, get_processed_and_shifted_recurrence


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
    n_initial, order, recurrence = get_processed_and_shifted_recurrence(pde)

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

        lamb_expr_symb_deriv = sp.diff(g_x_y, var_t[0], i)
        for j in range(ndim):
            lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
        
        if i < n_initial:
            lamb_expr_symb = lamb_expr_symb_deriv
        else:
            lamb_expr_symb = recurrence.subs(n, i)
        #print("=============== ORDER = " + str(i))
        #print(lamb_expr_symb)
        return sp.lambdify(arg_list, lamb_expr_symb), sp.lambdify(arg_list, lamb_expr_symb_deriv)

    interactions = 0
    coord = [cts_r_s[j] for j in range(ndim)]
    for i in range(p+1):
        lamb_expr, true_lamb_expr = generate_lamb_expr(i, n_initial)
        a = [*storage, *coord]
        s_new = lamb_expr(*a)
        s_new_true = true_lamb_expr(*a)
        arg_max = np.argmax(abs(s_new-s_new_true)/abs(s_new_true))
        print((s_new-s_new_true).reshape(-1)[arg_max]/s_new_true.reshape(-1)[arg_max])
        print("x:", coord[0].reshape(-1)[arg_max], "y:", coord[1].reshape(-1)[arg_max],
              "s_recur:", s_new.reshape(-1)[arg_max], "s_true:", s_new_true.reshape(-1)[arg_max], "order: ", i)
        interactions += s_new * radius**i/math.factorial(i)

        storage.pop(0)
        storage.append(s_new)

    exp_res = (interactions * strengths[None, :]).sum(axis=1)

    return exp_res