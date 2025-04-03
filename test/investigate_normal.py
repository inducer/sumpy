# %%
from sumpy.recurrence import _make_sympy_vec, get_reindexed_and_center_origin_on_axis_recurrence

from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)


import sympy as sp
from sympy import hankel1

import numpy as np

import math

import matplotlib.pyplot as plt
from matplotlib import cm, ticker

# %%
w = make_identity_diff_op(2)
laplace2d = laplacian(w)
n_init_lap, order_lap, recur_laplace = get_reindexed_and_center_origin_on_axis_recurrence(laplace2d)

w = make_identity_diff_op(2)
helmholtz2d = laplacian(w) + w
n_init_helm, order_helm, recur_helmholtz = get_reindexed_and_center_origin_on_axis_recurrence(helmholtz2d)

# %%
var = _make_sympy_vec("x", 2)
rct = sp.symbols("r_{ct}")
s = sp.Function("s")
n = sp.symbols("n")

# %%
def compute_derivatives(p):
    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))
    derivs = [sp.diff(g_x_y,
                        var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                        for i in range(p)]
    return derivs
l_max = 10
derivs_laplace = compute_derivatives(l_max)

# %%
def compute_derivatives_h2d(p):
    k = 1
    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 +
                        (var[1]-var_t[1])**2)
    g_x_y = (1j/4) * hankel1(0, k * abs_dist)
    derivs_helmholtz = [sp.diff(g_x_y,
                        var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                                                for i in range(p)]
    return derivs_helmholtz
h_max = 8
#derivs_helmholtz = compute_derivatives_h2d(h_max)

# %%
def evaluate_recurrence_lamb(coord_dict, recur, p, derivs_list, n_initial, n_order):
    s = sp.Function("s")
    subs_dict = {}
    for i in range(n_initial-n_order, 0):
        subs_dict[s(i)] = 0
    for i in range(n_initial):
        subs_dict[s(i)] = derivs_list[i].subs(coord_dict)
    var = _make_sympy_vec("x", 2)
    for i in range(n_initial, p):
        exp = recur.subs(n, i)
        f = sp.lambdify([var[0], var[1]] + [s(i-(1+k)) for k in range(n_order-1)], exp)
        subs_dict[s(i)] = f(*([coord_dict[var[0]], coord_dict[var[1]]] + [subs_dict[s(i-(1+k))] for k in range(n_order-1)]))
    for i in range(n_initial-n_order, 0):
        subs_dict.pop(s(i))
    return np.array(list(subs_dict.values()))

# %%
def evaluate_true(coord_dict, p, derivs_list):
    retMe = []
    for i in range(p):
        exp = derivs_list[i]
        f = sp.lambdify(var, exp)
        retMe.append(f(coord_dict[var[0]], coord_dict[var[1]]))
    return np.array(retMe)

# %%
def compute_error_coord(recur, loc, order, derivs_list, n_initial, n_order):
    var = _make_sympy_vec("x", 2)
    coord_dict = {var[0]: loc[0], var[1]: loc[1]}

    exp = evaluate_recurrence_lamb(coord_dict, recur, order+1, derivs_list, n_initial, n_order)[order].evalf()
    
    true = derivs_list[order].subs(coord_dict).evalf()

    return (np.abs(exp-true)/np.abs(true))

# %%
def generate_error_grid(res, order_plot, recur, derivs, n_initial, n_order):
    x_grid = [10**(pw) for pw in np.linspace(-8, 0, res)]
    y_grid = [10**(pw) for pw in np.linspace(-8, 0, res)]
    res=len(x_grid)
    plot_me = np.empty((res, res))
    for i in range(res):
        for j in range(res):
            if abs(x_grid[i]) == abs(y_grid[j]):
                plot_me[i, j] = 1e-16
            else:
                plot_me[i,j] = compute_error_coord(recur, np.array([x_grid[i],y_grid[j]]), order_plot, derivs, n_initial, n_order)
                if plot_me[i,j] == 0:
                    plot_me[i, j] = 1e-16
    return x_grid, y_grid, plot_me

# %%
order_plot = 5
#x_grid, y_grid, plot_me_hem = generate_error_grid(res=5, order_plot=order_plot, recur=recur_helmholtz, derivs=derivs_helmholtz, n_initial=n_init_helm, n_order=order_helm)
x_grid, y_grid, plot_me_lap  = generate_error_grid(res=10, order_plot=order_plot, recur=recur_laplace, derivs=derivs_laplace, n_initial=n_init_lap, n_order=order_lap)
plot_me_hem = plot_me_lap
      
                
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
cs = ax1.contourf(x_grid, y_grid, plot_me_hem.T, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)

cs = ax2.contourf(x_grid, y_grid, plot_me_lap.T, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
cbar = fig.colorbar(cs)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel("source x-coord", fontsize=15)
ax1.set_ylabel("source y-coord", fontsize=15)


ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("source x-coord", fontsize=15)
ax2.set_ylabel("source y-coord", fontsize=15)

ax1.set_title("Helmholtz recurrence relative error for order = "+str(order_plot), fontsize=15)
ax2.set_title("Laplace recurrence relative error for order = "+str(order_plot), fontsize=15)

fig.savefig('order'+str(order_plot))
plt.show()

# %%



