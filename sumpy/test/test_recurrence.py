r"""
Tests for the recurrence computation module :mod:`sumpy.recurrence`.

Verifies that recurrence relations for Green's function derivatives
produce results matching direct symbolic differentiation.
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
from sumpy.recurrence import (
    _extract_idx_terms_from_recurrence,
    _make_sympy_vec,
    get_large_x1_recurrence,
    get_small_x1_expansion,
    get_small_x1_recurrence,
    recurrence_from_pde,
    reindex_recurrence_relation,
)


def _verify_large_x1_recurrence(pde, g_x_y, ndim, p, x_vals):
    r"""
    Verifies that the large-:math:`|x_1|` recurrence produces derivatives
    matching direct symbolic differentiation at a given evaluation point.

    Computes the first *p* + 1 derivatives of *g_x_y* with respect to
    the first coordinate at the target origin, both via direct
    differentiation and via the recurrence, and compares.
    """
    n_initial, order, recurrence = get_large_x1_recurrence(pde)

    var = _make_sympy_vec("x", ndim)
    var_t = _make_sympy_vec("t", ndim)
    n = sp.symbols("n")
    s = sp.Function("s")

    # Compute true derivatives of G w.r.t. x_0 at t=0
    true_derivs = []
    for i in range(p + 1):
        d = sp.diff(g_x_y, var[0], i)
        for j in range(ndim):
            d = d.subs(var_t[j], 0)
        true_derivs.append(complex(sp.N(d.subs(x_vals), 30)))

    # Compute via recurrence, seeding with true initial conditions.
    # Negative indices are assumed to be zero (matching the zero-initialized
    # storage in recurrence_qbx_lp).
    recur_vals: dict[int, complex] = {}
    for idx in range(-order, 0):
        recur_vals[idx] = 0j

    for i in range(n_initial):
        recur_vals[i] = true_derivs[i]

    for i in range(n_initial, p + 1):
        expr = recurrence.subs(n, i)
        for j in range(order, 0, -1):
            # pylint: disable-next=not-callable
            expr = expr.subs(s(i - j), recur_vals.get(i - j, 0))
        recur_vals[i] = complex(sp.N(expr.subs(x_vals), 30))

    for i in range(p + 1):
        if abs(true_derivs[i]) > 1e-30:
            rel_err = abs(recur_vals[i] - true_derivs[i]) / abs(true_derivs[i])
            assert rel_err < 1e-10


def test_recurrence_from_pde_nonzero():
    r"""
    Verifies that :func:`recurrence_from_pde` produces a nonzero recurrence
    expression for the 2D Laplace equation.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    r = recurrence_from_pde(laplace2d)
    assert r != 0


def test_reindex_recurrence_relation_structure():
    r"""
    Verifies that :func:`reindex_recurrence_relation` produces a recurrence
    with positive order and all :math:`s()` indices non-positive (i.e.,
    :math:`s(n)` depends only on :math:`s(n-1), s(n-2), \dots`).
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)
    r = recurrence_from_pde(laplace2d)
    order, reindexed = reindex_recurrence_relation(r)

    assert order > 0

    idx_l, _ = _extract_idx_terms_from_recurrence(reindexed)
    assert all(idx <= 0 for idx in idx_l)


def test_large_x1_recurrence_laplace_2d():
    r"""
    Verifies the large-:math:`|x_1|` recurrence for the 2D Laplace Green's
    function :math:`G = -\frac{1}{2\pi} \log|x - t|` by comparing
    recurrence-computed derivatives against direct symbolic differentiation
    at a test point.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = (-1/(2*np.pi)) * sp.log(sp.sqrt((var[0]-var_t[0])**2
                                             + (var[1]-var_t[1])**2))

    x_vals = [(var[0], sp.Rational(1, 2)), (var[1], sp.Rational(3, 10))]
    _verify_large_x1_recurrence(laplace2d, g_x_y, 2, 8, x_vals)


def test_large_x1_recurrence_laplace_3d():
    r"""
    Verifies the large-:math:`|x_1|` recurrence for the 3D Laplace Green's
    function :math:`G = \frac{1}{4\pi |x - t|}` by comparing
    recurrence-computed derivatives against direct symbolic differentiation
    at a test point.
    """
    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)

    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2
                       + (var[2]-var_t[2])**2)
    g_x_y = 1/(4*np.pi) * 1/abs_dist

    x_vals = [(var[0], sp.Rational(1, 2)), (var[1], sp.Rational(3, 10)),
              (var[2], sp.Rational(1, 5))]
    _verify_large_x1_recurrence(laplace3d, g_x_y, 3, 6, x_vals)


def test_large_x1_recurrence_helmholtz_2d():
    r"""
    Verifies the large-:math:`|x_1|` recurrence for the 2D Helmholtz Green's
    function :math:`G = \frac{i}{4} H_0^{(1)}(k|x - t|)` by comparing
    recurrence-computed derivatives against direct symbolic differentiation
    at a test point.
    """
    w = make_identity_diff_op(2)
    helmholtz2d = laplacian(w) + w

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    k = 1
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2)
    g_x_y = (1j/4) * hankel1(0, k * abs_dist)

    x_vals = [(var[0], sp.Rational(1, 2)), (var[1], sp.Rational(3, 10))]
    _verify_large_x1_recurrence(helmholtz2d, g_x_y, 2, 5, x_vals)


def test_small_x1_recurrence_valid_structure():
    r"""
    Verifies that the small-:math:`|x_1|` recurrence for 2D Laplace returns
    a recurrence with positive order and non-negative start order.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    start_order, recur_order, recur = (
        get_small_x1_recurrence(laplace2d)
    )

    assert start_order >= 0
    assert recur_order > 0
    assert recur != 0


def test_small_x1_expansion_valid_structure():
    r"""
    Verifies that the small-:math:`|x_1|` expansion for 2D Laplace returns
    a nonzero expression with non-negative parameters.
    """
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    exp, n_coeffs, start_order = get_small_x1_expansion(laplace2d, 4)

    assert exp != 0
    assert n_coeffs >= 0
    assert start_order >= 0
