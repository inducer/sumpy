from sumpy.recurrence import get_processed_and_shifted_recurrence, _make_sympy_vec
import sympy as sp
import numpy as np

from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)

def test_laplace_3D():
    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)
    _, _, r = get_processed_and_shifted_recurrence(laplace3d)
    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2 + (var[2]-var_t[2])**2)
    g_x_y = 1/abs_dist
    derivs = [sp.diff(g_x_y, var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0).subs(var_t[2], 0) for i in range(6)]

    check_2_s = r.subs(n, 2).subs(s(0), derivs[0]).subs(s(1), derivs[1]) - derivs[2]
    check_3_s = r.subs(n, 3).subs(s(0), derivs[0]).subs(s(1), derivs[1]).subs(s(2), derivs[2]) - derivs[3]
    check_4_s = r.subs(n, 4).subs(s(0), derivs[0]).subs(s(1), derivs[1]).subs(s(2), derivs[2]).subs(s(3), derivs[3]) - derivs[4]
    check_5_s = r.subs(n, 5).subs(s(0), derivs[0]).subs(s(1), derivs[1]).subs(s(2), derivs[2]).subs(s(3), derivs[3]).subs(s(4), derivs[4]) - derivs[5]

    assert abs(abs(check_2_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-15
    assert abs(abs(check_3_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-14
    assert abs(abs(check_4_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-12
    #print(abs(abs(check_5_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))))
    assert abs(abs(check_5_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-12


def test_helmholtz_3D():
    w = make_identity_diff_op(3)
    helmholtz3d = laplacian(w) + w
    _, _, r = get_processed_and_shifted_recurrence(helmholtz3d)

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2 + (var[2]-var_t[2])**2)
    g_x_y = sp.exp(1j * abs_dist) / abs_dist
    derivs = [sp.diff(g_x_y, var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0).subs(var_t[2], 0) for i in range(6)]


    check_2_s = r.subs(n, 2).subs(s(0), derivs[0]).subs(s(1), derivs[1]) - derivs[2]
    check_3_s = r.subs(n, 3).subs(s(0), derivs[0]).subs(s(1), derivs[1]).subs(s(2), derivs[2]) - derivs[3]
    check_4_s = r.subs(n, 4).subs(s(0), derivs[0]).subs(s(1), derivs[1]).subs(s(2), derivs[2]).subs(s(3), derivs[3]) - derivs[4]
    check_5_s = r.subs(n, 5).subs(s(0), derivs[0]).subs(s(1), derivs[1]).subs(s(2), derivs[2]).subs(s(3), derivs[3]).subs(s(4), derivs[4]) - derivs[5]

    assert abs(check_2_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand())) <= 1e-15
    assert abs(abs(check_3_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-14
    assert abs(abs(check_4_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-12
    assert abs(abs(check_5_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand()).subs(var[2], np.random.rand()))) <= 1e-12
    








def test_laplace_2D():
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    _,_, r = get_processed_and_shifted_recurrence(laplace2d)

    n = sp.symbols("n")
    s = sp.Function("s")

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))
    derivs = [sp.diff(g_x_y, var_t[0], i).subs(var_t[0], 0).subs(var_t[1], 0) for i in range(6)]

    check_2_s = r.subs(n, 2).subs(s(1), derivs[1]) - derivs[2]
    check_3_s = r.subs(n, 3).subs(s(1), derivs[1]).subs(s(2), derivs[2]) - derivs[3]
    check_4_s = r.subs(n, 4).subs(s(1), derivs[1]).subs(s(2), derivs[2]).subs(s(3), derivs[3]) - derivs[4]
    check_5_s = r.subs(n, 5).subs(s(1), derivs[1]).subs(s(2), derivs[2]).subs(s(3), derivs[3]).subs(s(4), derivs[4]) - derivs[5]

    assert abs(check_2_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand())) <= 1e-15
    assert abs(check_3_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand())) <= 1e-14
    assert abs(check_4_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand())) <= 1e-12
    assert abs(check_5_s.subs(var[0], np.random.rand()).subs(var[1], np.random.rand())) <= 1e-12


