r"""
With the functionality in this module, we aim to compute layer potentials
using a recurrence for one-dimensional derivatives of the corresponding 
Green's function. See recurrence.py.

.. autofunction:: recurrence_qbx_lp
"""
import numpy as np
import sympy as sp
from sumpy.recurrence import (
    _make_sympy_vec, 
    get_processed_and_shifted_recurrence)

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


    #------------- 4. Define input variables for green's function expression
    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)

    #------------ 5. Compute recurrence
    n_initial, order, recurrence = get_processed_and_shifted_recurrence(pde)

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