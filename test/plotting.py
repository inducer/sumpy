import numpy as np
import sympy as sp

from sumpy.recurrence import (
    _make_sympy_vec,
    get_off_axis_expression,
    get_reindexed_and_center_origin_off_axis_recurrence,
    get_reindexed_and_center_origin_on_axis_recurrence,
)

from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from sympy import hankel1

from immutabledict import immutabledict
from sumpy.expansion.diff_op import LinearPDESystemOperator

def produce_error_for_recurrences(coords, pde, g_x_y, deriv_order, m=100):

    #Possibly reshape coords?
    cts_r_s = coords.reshape(2,coords.shape[1],1)

    p = deriv_order-1
    cts_r_s = coords
    ndim = cts_r_s.shape[0]
    var = _make_sympy_vec("x", ndim)
    var_t = _make_sympy_vec("t", ndim)

    # ------------ 5. Compute recurrence
    n_initial, order, recurrence = get_reindexed_and_center_origin_on_axis_recurrence(pde)

    # ------------ 6. Set order p = 5
    n_p = cts_r_s.shape[1]
    storage = [np.zeros((1, n_p))] * order

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
        if i == p:
            interactions_on_axis += s_new

        storage.pop(0)
        storage.append(s_new)


    ### NEW CODE - COMPUTE OFF AXIS INTERACTIONS
    start_order, t_recur_order, t_recur = get_reindexed_and_center_origin_off_axis_recurrence(pde)
    t_exp, t_exp_order, _ = get_off_axis_expression(pde, 8)
    storage_taylor_order = max(t_recur_order, t_exp_order+1)

    storage_taylor = [np.zeros((1, n_p))] * storage_taylor_order
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
        storage_taylor.append(lamb_expr_t_recur(*a1) + np.zeros((1, n_p)))

        lamb_expr_t_exp = gen_lamb_expr_t_exp(i, t_exp_order, start_order)
        a2 = [*storage_taylor[-(t_exp_order+1):], *coord]

        if i == p:
            interactions_off_axis += lamb_expr_t_exp(*a2)

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
    for i in range(p, p+1):
        lamb_expr_true = generate_true(i)
        a4 = [*coord]
        s_new_true = lamb_expr_true(*a4)
        if i == p:
            interactions_true += s_new_true
    ###############

    #slope of line y = mx
    mask_on_axis = m*np.abs(coord[0]) >= np.abs(coord[1])
    mask_off_axis = m*np.abs(coord[0]) < np.abs(coord[1])

    interactions_off_axis = interactions_off_axis.reshape(coord[0].shape)

    interactions_total = np.zeros(coord[0].shape)
    interactions_total[mask_on_axis] = interactions_on_axis[mask_on_axis]
    interactions_total[mask_off_axis] = interactions_off_axis[mask_off_axis]

    return interactions_on_axis, interactions_off_axis, interactions_true, interactions_total

def create_logarithmic_mesh(res):

    x_grid = [10**(pw) for pw in np.linspace(-8, 0, res)]
    y_grid = [10**(pw) for pw in np.linspace(-8, 0, res)]

    mesh = np.meshgrid(x_grid, y_grid)
    mesh_points = np.array(mesh).reshape(2, -1)

    return mesh_points, x_grid, y_grid

def create_plot(relerr_on, str_title):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    n_levels = 18
    levels = 10**np.linspace(-n_levels+2, 1, n_levels)
    cs = ax.contourf(x_grid, y_grid, relerr_on.reshape(res, res), locator=ticker.LogLocator(), cmap=cm.coolwarm, levels=levels, extend="both")
    cbar = fig.colorbar(cs)

    cbar.set_ticks(levels)
    cbar.set_ticklabels(["1e"+str(int(i)) for i in np.linspace(-n_levels+2, 1, n_levels)])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("$x_1$-coordinate", fontsize=15)
    ax.set_ylabel("$x_2$-coordinate", fontsize=15)
    plt.title(str_title)

#========================= DEFINE PLOT RESOLUTION ====================================
res = 8
mesh_points, x_grid, y_grid = create_logarithmic_mesh(res)

#========================= DEFINE GREEN'S FUNCTIONS/PDE's ====================================
from collections import namedtuple
DerivativeIdentifier = namedtuple("DerivativeIdentifier", ["mi", "vec_idx"])
var = _make_sympy_vec("x", 2)
var_t = _make_sympy_vec("t", 2)
abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2)
w = make_identity_diff_op(2)

partial_1x = DerivativeIdentifier((4,0), 0)
partial_1y = DerivativeIdentifier((0,4), 0)
biharmonic_op = {partial_1x: 1, partial_1y: 1}
list_pde = immutabledict(biharmonic_op)


biharmonic_pde = LinearPDESystemOperator(2, (list_pde,))
g_x_y_biharmonic = abs_dist**2 * sp.log(abs_dist)

laplace2d = laplacian(w)
g_x_y_laplace = (-1/(2*np.pi)) * sp.log(abs_dist)

k = 1
helmholtz2d = laplacian(w) + w
g_x_y_helmholtz = (1j/4) * hankel1(0, k * abs_dist)
#========================= LAPLACE 2D ====================================
#interactions_on_axis, interactions_off_axis, interactions_true, interactions_total = produce_error_for_recurrences(mesh_points, laplace2d, g_x_y_laplace, 9)

#relerr_on = np.abs((interactions_on_axis-interactions_true)/interactions_true)
#plt.figure(1)
#create_plot(relerr_on, "Laplace (2D): On-Axis Recurrence, 9th Order Derivative Evaluation Error $(u_{recur}-u_{sym})/u_{sym}$")

#========================= HELMOLTZ 2D ====================================
#interactions_on_axis, interactions_off_axis, interactions_true, interactions_total = produce_error_for_recurrences(mesh_points, helmholtz2d, g_x_y_helmholtz, 9)

#relerr_on = np.abs((interactions_on_axis-interactions_true)/interactions_true)
#create_plot(relerr_on, "Helmholtz (2D): On-Axis Recurrence, 8th Order Derivative Evaluation Error $(u_{recur}-u_{sym})/u_{sym}$")


#======================== BIHARMONIC 2D ===================================
interactions_on_axis, interactions_off_axis, interactions_true, interactions_total = produce_error_for_recurrences(mesh_points, biharmonic_pde, g_x_y_biharmonic, 7)

relerr_on = np.abs((interactions_on_axis-interactions_true)/interactions_true)
create_plot(relerr_on, "Biharmonic (2D): On-Axis Recurrence, 8th Order Derivative Evaluation Error $(u_{recur}-u_{sym})/u_{sym}$")

plt.show()