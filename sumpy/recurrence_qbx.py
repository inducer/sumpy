r"""
Evaluates QBX layer potentials using recurrence-based computation of
Green's function derivatives. Sources are first rotated into a coordinate
system aligned with the expansion center normal, and then the
large-:math:`|x_1|` and small-:math:`|x_1|` recurrences from
:mod:`sumpy.recurrence` are used to build the local expansion.

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
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from sumpy.recurrence import (
    _make_sympy_vec,
    get_large_x1_recurrence,
    get_small_x1_expansion,
    get_small_x1_recurrence,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


# ================ Transform/Rotate =================
def _produce_orthogonal_basis(normals: np.ndarray) -> Sequence[np.ndarray]:
    r"""
    Produces an orthonormal basis for each center, with the first basis
    vector equal to the given normal. The remaining basis vectors are
    generated via Gram-Schmidt orthogonalization of random vectors.

    :arg normals: a ``(ndim, ncenters)`` array of unit normal vectors.

    :returns: a list of *ndim* arrays, each of shape ``(ndim, ncenters)``,
        forming a per-center orthonormal basis.
    """
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
    r"""
    Computes source coordinates shifted by the center locations and rotated
    into a coordinate system where the first axis is aligned with the
    center normal.

    :arg sources: a ``(ndim, nsources)`` array of source locations.
    :arg centers: a ``(ndim, ncenters)`` array of expansion center locations.
    :arg normals: a ``(ndim, ncenters)`` array of unit normals at each center.

    :returns: a ``(ndim, ncenters, nsources)`` array of rotated, shifted
        coordinates.
    """
    cts = centers[:, :, None] - sources[:, None]
    orth_coordsys = _produce_orthogonal_basis(normals)
    cts_rotated_shifted = np.einsum("idc,dcs->ics", orth_coordsys, cts)

    return cts_rotated_shifted


# ================ Recurrence LP Eval =================
def recurrence_qbx_lp(sources, centers, normals, strengths, radius, pde, g_x_y,
                      ndim, p, off_axis_start=0) -> np.ndarray:
    r"""
    Computes a single-layer potential using recurrence-based QBX. Sources
    are rotated into a per-center coordinate system aligned with the normal,
    and derivatives of the Green's function are computed via the
    large-:math:`|x_1|` and small-:math:`|x_1|` recurrences. The two
    regimes are blended based on the relative magnitude of the coordinates.

    :arg sources: a ``(ndim, nsources)`` array of source locations.
    :arg centers: a ``(ndim, ncenters)`` array of expansion center locations.
    :arg normals: a ``(ndim, ncenters)`` array of unit normals at each center.
    :arg strengths: a ``(nsources,)`` array of quadrature weights multiplied
        by the density.
    :arg radius: the QBX expansion radius.
    :arg pde: a :class:`sumpy.expansion.diff_op.LinearPDESystemOperator`
        describing the PDE whose Green's function is used.
    :arg g_x_y: a sympy expression for the Green's function in source
        variables :math:`(x_0, x_1, \dots)` and target variables
        :math:`(t_0, t_1, \dots)`.
    :arg ndim: the number of spatial dimensions.
    :arg p: the order of the QBX expansion.

    :returns: a ``(ncenters,)`` array of layer potential values at the
        expansion centers.
    """

    # ------------- 2. Compute rotated/shifted coordinates
    cts_r_s = _compute_rotated_shifted_coordinates(sources, centers, normals)

    # ------------- 4. Define input variables for green's function expression
    var = _make_sympy_vec("x", ndim)
    var_t = _make_sympy_vec("t", ndim)

    # ------------ 5. Compute large-|x_1| recurrence
    n_initial, order, recurrence = get_large_x1_recurrence(pde)

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
            lamb_expr_symb_deriv = sp.diff(g_x_y, var[0], i)
            for j in range(ndim):
                lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
            lamb_expr_symb = lamb_expr_symb_deriv
        else:
            lamb_expr_symb = recurrence.subs(n, i)
        return sp.lambdify(arg_list, lamb_expr_symb)

    coord = [cts_r_s[j] for j in range(ndim)]
    interactions_on_axis = coord[0] * 0
    for i in range(p+1):
        lamb_expr = generate_lamb_expr(i, n_initial)
        a = [*storage, *coord]
        s_new = lamb_expr(*a)
        interactions_on_axis += s_new * radius**i/math.factorial(i)

        storage.pop(0)
        storage.append(s_new)

    # Compute small-|x_1| interactions
    start_order, t_recur_order, t_recur = get_small_x1_recurrence(pde)
    t_exp, t_exp_order, _ = get_small_x1_expansion(pde, 8)
    storage_taylor_order = max(t_recur_order, t_exp_order+1)

    start_order = max(start_order, order)

    storage_taylor = [np.zeros((n_p, n_p))] * storage_taylor_order

    def gen_lamb_expr_t_recur(i, start_order):
        arg_list = []
        for j in range(t_recur_order, 0, -1):
            # pylint: disable-next=not-callable
            arg_list.append(s(i-j))
        for j in range(ndim):
            arg_list.append(var[j])

        if i < start_order:
            lamb_expr_symb_deriv = sp.diff(g_x_y, var[0], i)
            for j in range(ndim):
                lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
            lamb_expr_symb = lamb_expr_symb_deriv.subs(var[0], 0)
        else:
            lamb_expr_symb = t_recur.subs(n, i)

        return sp.lambdify(arg_list, lamb_expr_symb)

    def gen_lamb_expr_t_exp(i, t_exp_order, start_order):
        arg_list = []
        for j in range(t_exp_order, -1, -1):
            # pylint: disable-next=not-callable
            arg_list.append(s(i-j))
        for j in range(ndim):
            arg_list.append(var[j])

        if i < start_order:
            lamb_expr_symb_deriv = sp.diff(g_x_y, var[0], i)
            for j in range(ndim):
                lamb_expr_symb_deriv = lamb_expr_symb_deriv.subs(var_t[j], 0)
            lamb_expr_symb = lamb_expr_symb_deriv
        else:
            lamb_expr_symb = t_exp.subs(n, i)

        return sp.lambdify(arg_list, lamb_expr_symb)

    interactions_off_axis: np.ndarray | int = 0
    for i in range(p+1):
        lamb_expr_t_recur = gen_lamb_expr_t_recur(i, start_order)
        a1 = [*storage_taylor[(-t_recur_order):], *coord]

        storage_taylor.pop(0)
        storage_taylor.append(lamb_expr_t_recur(*a1) + np.zeros((n_p, n_p)))

        lamb_expr_t_exp = gen_lamb_expr_t_exp(i, t_exp_order, start_order)
        a2 = [*storage_taylor[-(t_exp_order+1):], *coord]

        interactions_off_axis += lamb_expr_t_exp(*a2) * radius**i/math.factorial(i)

    # Blend large-|x_1| and small-|x_1| regimes based on relative coordinates
    m = 100
    mask_on_axis = m*np.abs(coord[0]) >= np.abs(coord[1])
    mask_off_axis = m*np.abs(coord[0]) < np.abs(coord[1])

    interactions_total = np.zeros(coord[0].shape)
    interactions_total[mask_on_axis] = interactions_on_axis[mask_on_axis]
    interactions_total[mask_off_axis] = interactions_off_axis[mask_off_axis]  # pyright: ignore[reportIndexIssue]

    return (interactions_total * strengths[None, :]).sum(axis=1)
