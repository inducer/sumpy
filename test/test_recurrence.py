r"""
With the functionality in this module, we aim to test recurrence
code.

.. autofunction:: test_laplace3d
.. autofunction:: test_helmholtz3d
.. autofunction:: test_laplace2d
.. autofunction:: test_helmholtz2d
.. autofunction:: test_laplace_2d_off_axis
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
import numpy as np
import sympy as sp
from sympy import hankel1

from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)
from sumpy.recurrence import _make_sympy_vec, get_reindexed_and_center_origin_on_axis_recurrence, get_off_axis_expression, get_reindexed_and_center_origin_off_axis_recurrence
import math

def test_laplace3d():
    r"""
    Tests recurrence code for orders up to 6 laplace3d.
    """
    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)
    n_init, _, r = get_reindexed_and_center_origin_on_axis_recurrence(laplace3d)
    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 +
                       (var[1]-var_t[1])**2 + (var[2]-var_t[2])**2)
    g_x_y = 1/abs_dist
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0).subs(var_t[2], 0)
                                               for i in range(6)]

    # pylint: disable-next=not-callable
    subs_dict = {s(0): derivs[0], s(1): derivs[1]}
    check = []

    assert n_init == 2
    max_order_check = 6
    for i in range(n_init, max_order_check):
        check.append(r.subs(n, i).subs(subs_dict) - derivs[i])
        # pylint: disable-next=not-callable
        subs_dict[s(i)] = derivs[i]

    x_coord = np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    z_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord, var[2]: z_coord}

    check = np.array([check[i].subs(coord_dict) for i in range(len(check))])

    assert max(abs(check)) <= 1e-12


def test_helmholtz3d():
    r"""
    Tests recurrence code for orders up to 6 helmholtz3d.
    """
    w = make_identity_diff_op(3)
    helmholtz3d = laplacian(w) + w
    n_init, _, r = get_reindexed_and_center_origin_on_axis_recurrence(helmholtz3d)

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 +
                       (var[1]-var_t[1])**2 + (var[2]-var_t[2])**2)
    g_x_y = sp.exp(1j * abs_dist) / abs_dist
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0).subs(var_t[2], 0)
                                               for i in range(6)]

    # pylint: disable-next=not-callable
    subs_dict = {s(0): derivs[0], s(1): derivs[1]}
    check = []

    assert n_init == 2
    max_order_check = 6
    for i in range(n_init, max_order_check):
        check.append(r.subs(n, i).subs(subs_dict) - derivs[i])
        # pylint: disable-next=not-callable
        subs_dict[s(i)] = derivs[i]

    x_coord = np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    z_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord, var[2]: z_coord}

    check = np.array([check[i].subs(coord_dict) for i in range(len(check))])

    assert max(abs(abs(check))) <= 1e-12


def test_helmholtz2d():
    r"""
    Tests recurrence code for orders up to 6 helmholtz2d.
    """
    w = make_identity_diff_op(2)
    helmholtz2d = laplacian(w) + w
    n_init, _, r = get_reindexed_and_center_origin_on_axis_recurrence(helmholtz2d)

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 +
                       (var[1]-var_t[1])**2)
    k = 1
    g_x_y = (1j/4) * hankel1(0, k * abs_dist)
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                                               for i in range(6)]
    x_coord = np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord}
    derivs = [d.subs(coord_dict) for d in derivs]

    # pylint: disable-next=not-callable
    subs_dict = {s(0): derivs[0], s(1): derivs[1]}
    check = []

    assert n_init == 2
    max_order_check = 6
    for i in range(n_init, max_order_check):
        check.append(r.subs(n, i).subs(subs_dict) - derivs[i])
        # pylint: disable-next=not-callable
        subs_dict[s(i)] = derivs[i]

    f2 = sp.lambdify([var[0], var[1]], check[0])
    assert abs(f2(x_coord, y_coord)) <= 1e-13
    f3 = sp.lambdify([var[0], var[1]], check[1])
    assert abs(f3(x_coord, y_coord)) <= 1e-13
    f4 = sp.lambdify([var[0], var[1]], check[2])
    assert abs(f4(x_coord, y_coord)) <= 1e-13
    f5 = sp.lambdify([var[0], var[1]], check[3])
    assert abs(f5(x_coord, y_coord)) <= 1e-12


def test_laplace2d():
    r"""
    Tests recurrence code for orders up to 6 laplace2d.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    n_init, _, r = get_reindexed_and_center_origin_on_axis_recurrence(laplace2d)

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                      for i in range(6)]

    # pylint: disable-next=not-callable
    subs_dict = {s(0): derivs[0], s(1): derivs[1]}
    check = []

    assert n_init == 2
    max_order_check = 6
    for i in range(n_init, max_order_check):
        check.append(r.subs(n, i).subs(subs_dict) - derivs[i])
        # pylint: disable-next=not-callable
        subs_dict[s(i)] = derivs[i]

    x_coord = np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord}

    check = np.array([check[i].subs(coord_dict) for i in range(len(check))])
    assert max(abs(abs(check))) <= 1e-12


def test_helmholtz_2d_off_axis():
    r"""
    Tests off-axis recurrence code for orders up to 6 laplace2d.
    """
    w = make_identity_diff_op(2)
    helmholtz2d = laplacian(w) + w

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 +
                       (var[1]-var_t[1])**2)
    k = 1
    g_x_y = (1j/4) * hankel1(0, k * abs_dist)
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                                               for i in range(8)]
    
    x_coord = 1e-2 * np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord}
    start_order, recur_order, recur = get_reindexed_and_center_origin_off_axis_recurrence(helmholtz2d)

    ic = []
    #Generate ic

    for i in range(start_order):
        ic.append(derivs[i].subs(var[0], 0).subs(var[1], coord_dict[var[1]]))

    n = sp.symbols("n")
    for i in range(start_order, 15):
        recur_eval = recur.subs(var[0], coord_dict[var[0]]).subs(var[1], coord_dict[var[1]]).subs(n, i)
        for j in range(i-recur_order, i):
            recur_eval = recur_eval.subs(s(j), ic[j])
        ic.append(recur_eval)

    ic = np.array(ic)

    #true_ic = np.array([derivs[i].subs(var[0], 0).subs(var[1], coord_dict[var[1]]) for i in range(15)])
    
    #assert np.max(np.abs(ic[::2]-true_ic[::2])/np.abs(true_ic[::2])) < 1e-8
    #print(np.max(np.abs(ic[::2]-true_ic[::2])/np.abs(true_ic[::2])))

    # CHECK ACCURACY OF EXPRESSION FOR deriv_order
    deriv_order = 7
    exp_order = 4

    exp, exp_range = get_off_axis_expression(helmholtz2d, exp_order)
    approx_deriv = exp.subs(n, deriv_order)
    exp_range = (exp_range[0]+deriv_order, exp_range[1]+deriv_order)
    for i in range(exp_range[0], exp_range[1]+1):
        approx_deriv = approx_deriv.subs(s(i), ic[i])
    
    rat = coord_dict[var[0]]/coord_dict[var[1]]
    if deriv_order + exp_order % 2 == 0:
        prederror = abs((ic[deriv_order+exp_order+2] * coord_dict[var[0]]**(exp_order+2)/math.factorial(exp_order+2)).evalf())
    else:
        prederror = abs((ic[deriv_order+exp_order+1] * coord_dict[var[0]]**(exp_order+1)/math.factorial(exp_order+1)).evalf())
    print("PREDICTED ERROR: ", prederror)
    relerr = abs(((approx_deriv - derivs[deriv_order])/derivs[deriv_order]).subs(var[0], coord_dict[var[0]]).subs(var[1], coord_dict[var[1]]).evalf())
    print("RELATIVE ERROR: ", relerr)
    print("RATIO: ", rat)
    assert relerr <= prederror

test_helmholtz_2d_off_axis()


def test_laplace_2d_off_axis():
    r"""
    Tests off-axis recurrence code for orders up to 6 laplace2d.
    """
    s = sp.Function("s")
    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                      for i in range(15)]
    x_coord = 1e-2 * np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord}

    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    start_order, recur_order, recur = get_reindexed_and_center_origin_off_axis_recurrence(laplace2d)

    ic = []
    #Generate ic

    for i in range(start_order):
        ic.append(derivs[i].subs(var[0], 0).subs(var[1], coord_dict[var[1]]))

    n = sp.symbols("n")
    for i in range(start_order, 15):
        recur_eval = recur.subs(var[0], coord_dict[var[0]]).subs(var[1], coord_dict[var[1]]).subs(n, i)
        for j in range(i-recur_order, i):
            recur_eval = recur_eval.subs(s(j), ic[j])
        ic.append(recur_eval)

    ic = np.array(ic)

    true_ic = np.array([derivs[i].subs(var[0], 0).subs(var[1], coord_dict[var[1]]) for i in range(15)])
    
    assert np.max(np.abs(ic[::2]-true_ic[::2])/np.abs(true_ic[::2])) < 1e-8
    #print(np.max(np.abs(ic[::2]-true_ic[::2])/np.abs(true_ic[::2])))

    # CHECK ACCURACY OF EXPRESSION FOR deriv_order
    deriv_order = 7
    exp_order = 6

    exp, exp_range = get_off_axis_expression(laplace2d, exp_order)
    approx_deriv = exp.subs(n, deriv_order)
    exp_range = (exp_range[0]+deriv_order, exp_range[1]+deriv_order)
    for i in range(exp_range[0], exp_range[1]+1):
        approx_deriv = approx_deriv.subs(s(i), ic[i])
    
    rat = coord_dict[var[0]]/coord_dict[var[1]]
    if deriv_order + exp_order % 2 == 0:
        prederror = abs(ic[deriv_order+exp_order+2] * coord_dict[var[0]]**(exp_order+2)/math.factorial(exp_order+2))
    else:
        prederror = abs(ic[deriv_order+exp_order+1] * coord_dict[var[0]]**(exp_order+1)/math.factorial(exp_order+1))
    print("PREDICTED ERROR: ", prederror)
    relerr = abs((approx_deriv - derivs[deriv_order])/derivs[deriv_order]).subs(var[0], coord_dict[var[0]]).subs(var[1], coord_dict[var[1]])
    print("RELATIVE ERROR: ", relerr)
    print("RATIO: ", rat)
    assert relerr <= prederror

import matplotlib.pyplot as plt
def _plot_laplace_2d(max_order_check, max_abs):
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    n_init, _, r = get_reindexed_and_center_origin_on_axis_recurrence(laplace2d)

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))
    derivs = [sp.diff(g_x_y,
                      var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0)
                      for i in range(max_order_check)]

    # pylint: disable-next=not-callable
    subs_dict = {s(0): derivs[0], s(1): derivs[1]}
    check = []

    assert n_init == 2
    for i in range(n_init, max_order_check):
        check.append(abs(r.subs(n, i).subs(subs_dict) - derivs[i])/abs(derivs[i]))
        # pylint: disable-next=not-callable
        subs_dict[s(i)] = derivs[i]

    x_coord = abs(np.random.rand()*max_abs)  # noqa: NPY002
    y_coord = abs(np.random.rand()*max_abs)  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord}

    return np.array([check[i].subs(coord_dict) for i in range(len(check))])

""" plot_me = _plot_laplace_2d(13, 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = ax.plot([i+2 for i in range(len(plot_me))], plot_me)
ax.set_yscale('log')
plt.ylabel("Error")
plt.xlabel("Order")
plt.show() """

