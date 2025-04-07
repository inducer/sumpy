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

def show_scalar_in_matplotlib(self, fld, max_val=None, func_name="imshow", **kwargs):
    squeezed_points = self.points.squeeze()

    if len(squeezed_points.shape) != 2:
        raise RuntimeError(
                "matplotlib plotting requires 2D geometry")

    if len(fld.shape) == 1:
        fld = fld.reshape(self.nd_points.shape[1:])

    squeezed_fld = fld.squeeze()

    if max_val is not None:
        squeezed_fld[squeezed_fld > max_val] = max_val
        squeezed_fld[squeezed_fld < -max_val] = -max_val

    squeezed_fld = squeezed_fld[..., ::-1]

    a, b = self._get_squeezed_bounds()

    kwargs["extent"] = (
            # (left, right, bottom, top)
            a[0], b[0],
            a[1], b[1])

    import matplotlib.pyplot as pt
    return getattr(pt, func_name)(squeezed_fld.T, **kwargs)

def produce_error_for_recurrences(coords, pde, g_x_y, deriv_order, m=100):

    #Possibly reshape coords?
    p = deriv_order-1
    cts_r_s = coords
    ndim = cts_r_s.shape[0]
    var = _make_sympy_vec("x", ndim)
    var_t = _make_sympy_vec("t", ndim)

    # ------------ 5. Compute recurrence
    n_initial, order, recurrence = get_reindexed_and_center_origin_on_axis_recurrence(pde)

    # ------------ 6. Set order p = 5
    n_p = cts_r_s.shape[1]
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
        if i == p+1:
            interactions_on_axis += s_new

        storage.pop(0)
        storage.append(s_new)


    ### NEW CODE - COMPUTE OFF AXIS INTERACTIONS
    start_order, t_recur_order, t_recur = get_reindexed_and_center_origin_off_axis_recurrence(pde)
    t_exp, t_exp_order, _ = get_off_axis_expression(pde, 8)
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


    def gen_lamb_expr_t_exp(i, t_exp_order, start_order):
        arg_list = []
        for j in range(t_exp_order, -1, -1):
            # pylint: disable-next=not-callable
            arg_list.append(s(i-j))
        for j in range(ndim):
            arg_list.append(var[j])

        if i < start_order:
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

        storage_taylor.pop(0)
        storage_taylor.append(lamb_expr_t_recur(*a1) + np.zeros((n_p, n_p)))

        lamb_expr_t_exp = gen_lamb_expr_t_exp(i, t_exp_order, start_order)
        a2 = [*storage_taylor[-(t_exp_order+1):], *coord]

        if i == p+1:
            interactions_off_axis += lamb_expr_t_exp(*a2)

    ################
    # Compute True Interactions
    storage_taylor_true = [np.zeros((n_p, n_p))] * storage_taylor_order
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
    for i in range(p, p+1):
        lamb_expr_true = generate_true(i)
        a4 = [*coord]
        s_new_true = lamb_expr_true(*a4)
        if i == p+1:
            interactions_true += s_new_true
    ###############

    #slope of line y = mx
    mask_on_axis = m*np.abs(coord[0]) >= np.abs(coord[1])
    mask_off_axis = m*np.abs(coord[0]) < np.abs(coord[1])

    interactions_total = np.zeros(coord[0].shape)
    interactions_total[mask_on_axis] = interactions_on_axis[mask_on_axis]
    interactions_total[mask_off_axis] = interactions_off_axis[mask_off_axis]

    return interactions_on_axis, interactions_off_axis, interactions_true, interactions_total


import matplotlib.pyplot as plt
from sumpy.visualization import FieldPlotter
center = np.asarray([0, 0], dtype=np.float64)
fp = FieldPlotter(center, npoints=1000, extent=6)

plt.clf()
vol_pot = np.outer(0.3**np.arange(1, 100), 1.1**np.arange(1, 100))
plotval = np.log10(1e-20+np.abs(vol_pot))
im = fp.show_scalar_in_matplotlib(plotval.real)
from matplotlib.colors import Normalize
im.set_norm(Normalize(vmin=-8, vmax=5))

cb = plt.colorbar(shrink=0.9)
cb.set_label(r"$\log_{10}(\mathdefault{Error})$")
fp.set_matplotlib_limits()

plt.show()