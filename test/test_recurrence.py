r"""
With the functionality in this module, we aim to test recurrence
code.

.. autofunction:: test_laplace3d
.. autofunction:: test_helmholtz3d
.. autofunction:: test_laplace2d
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
from sumpy.recurrence import _make_sympy_vec, get_processed_and_shifted_recurrence


def test_laplace3d():
    r"""
    Tests recurrence code for orders up to 6 laplace3d.
    """
    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)
    n_init, _, r = get_processed_and_shifted_recurrence(laplace3d)
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
    # Check that the lowest order recurrence that works is 2
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
    _, _, r = get_processed_and_shifted_recurrence(helmholtz3d)

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
    check_2_s = r.subs(n, 2).subs(subs_dict) - derivs[2]
    # pylint: disable-next=not-callable
    subs_dict[s(2)] = derivs[2]
    check_3_s = r.subs(n, 3).subs(subs_dict) - derivs[3]
    # pylint: disable-next=not-callable
    subs_dict[s(3)] = derivs[3]
    check_4_s = r.subs(n, 4).subs(subs_dict) - derivs[4]
    # pylint: disable-next=not-callable
    subs_dict[s(4)] = derivs[4]
    check_5_s = r.subs(n, 5).subs(subs_dict) - derivs[5]

    x_coord = np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    z_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord, var[2]: z_coord}

    assert abs(abs(check_2_s.subs(coord_dict))) <= 1e-15
    assert abs(abs(check_3_s.subs(coord_dict))) <= 1e-14
    assert abs(abs(check_4_s.subs(coord_dict))) <= 1e-12
    assert abs(abs(check_5_s.subs(coord_dict))) <= 1e-12


def test_helmholtz2d():
    r"""
    Tests recurrence code for orders up to 6 helmholtz2d.
    """
    w = make_identity_diff_op(2)
    helmholtz2d = laplacian(w) + w
    _, _, r = get_processed_and_shifted_recurrence(helmholtz2d)

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
    derivs = [derivs[i].subs(coord_dict) for i in range(6)]

    # pylint: disable-next=not-callable
    subs_dict = {s(0): derivs[0], s(1): derivs[1]}
    check_2_s = r.subs(n, 2).subs(subs_dict) - derivs[2]
    # pylint: disable-next=not-callable
    subs_dict[s(2)] = derivs[2]
    check_3_s = r.subs(n, 3).subs(subs_dict) - derivs[3]
    # pylint: disable-next=not-callable
    subs_dict[s(3)] = derivs[3]
    check_4_s = r.subs(n, 4).subs(subs_dict) - derivs[4]
    # pylint: disable-next=not-callable
    subs_dict[s(4)] = derivs[4]
    check_5_s = r.subs(n, 5).subs(subs_dict) - derivs[5]

    f2 = sp.lambdify([var[0], var[1]], check_2_s)
    assert abs(f2(x_coord, y_coord)) <= 1e-13
    f3 = sp.lambdify([var[0], var[1]], check_3_s)
    assert abs(f3(x_coord, y_coord)) <= 1e-13
    f4 = sp.lambdify([var[0], var[1]], check_4_s)
    assert abs(f4(x_coord, y_coord)) <= 1e-13
    f5 = sp.lambdify([var[0], var[1]], check_5_s)
    assert abs(f5(x_coord, y_coord)) <= 1e-12


def test_laplace2d():
    r"""
    Tests recurrence code for orders up to 6 laplace2d.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    _, _, r = get_processed_and_shifted_recurrence(laplace2d)

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
    check_2_s = r.subs(n, 2).subs(subs_dict) - derivs[2]
    # pylint: disable-next=not-callable
    subs_dict[s(2)] = derivs[2]
    check_3_s = r.subs(n, 3).subs(subs_dict) - derivs[3]
    # pylint: disable-next=not-callable
    subs_dict[s(3)] = derivs[3]
    check_4_s = r.subs(n, 4).subs(subs_dict) - derivs[4]
    # pylint: disable-next=not-callable
    subs_dict[s(4)] = derivs[4]
    check_5_s = r.subs(n, 5).subs(subs_dict) - derivs[5]

    x_coord = np.random.rand()  # noqa: NPY002
    y_coord = np.random.rand()  # noqa: NPY002
    coord_dict = {var[0]: x_coord, var[1]: y_coord}

    assert abs(abs(check_2_s.subs(coord_dict))) <= 1e-15
    assert abs(abs(check_3_s.subs(coord_dict))) <= 1e-14
    assert abs(abs(check_4_s.subs(coord_dict))) <= 1e-12
    assert abs(abs(check_5_s.subs(coord_dict))) <= 1e-12
