r"""
With the functionality in this module, we aim to test recurrence
+ qbx code.
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

from sumpy.array_context import _acf
from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import HelmholtzKernel, LaplaceKernel
from sumpy.qbx import LayerPotential
from sumpy.recurrenceqbx import _make_sympy_vec, recurrence_qbx_lp


actx_factory = _acf
ExpnClass = LineTaylorLocalExpansion

actx = actx_factory()
lknl = LaplaceKernel(2)
hlknl = HelmholtzKernel(2, "k")


def _qbx_lp_helmholtz_general(sources, targets, centers, radius, strengths, order):
    lpot = LayerPotential(actx.context,
    expansion=ExpnClass(hlknl, order),
    target_kernels=(hlknl,),
    source_kernels=(hlknl,))

    # print(lpot.get_kernel())
    expansion_radii = actx.from_numpy(radius * np.ones(sources.shape[1]))
    sources = actx.from_numpy(sources)
    targets = actx.from_numpy(targets)
    centers = actx.from_numpy(centers)

    strengths = (strengths,)
    _evt, (result_qbx,) = lpot(
            actx.queue,
            targets, sources, centers, strengths,
            expansion_radii=expansion_radii,
            k=1)
    result_qbx = actx.to_numpy(result_qbx)

    return result_qbx


def _qbx_lp_laplace_general(sources, targets, centers, radius, strengths, order):
    lpot = LayerPotential(actx.context,
    expansion=ExpnClass(lknl, order),
    target_kernels=(lknl,),
    source_kernels=(lknl,))

    # print(lpot.get_kernel())
    expansion_radii = actx.from_numpy(radius * np.ones(sources.shape[1]))
    sources = actx.from_numpy(sources)
    targets = actx.from_numpy(targets)
    centers = actx.from_numpy(centers)

    strengths = (strengths,)

    _evt, (result_qbx,) = lpot(
            actx.queue,
            targets, sources, centers, strengths,
            expansion_radii=expansion_radii)
    result_qbx = actx.to_numpy(result_qbx)

    return result_qbx


def _create_ellipse(n_p):
    h = 9.688 / n_p
    radius = 7*h
    t = np.linspace(0, 2 * np.pi, n_p, endpoint=False)

    unit_circle_param = np.exp(1j * t)
    unit_circle = np.array([2 * unit_circle_param.real, unit_circle_param.imag])

    sources = unit_circle
    normals = np.array([unit_circle_param.real, 2*unit_circle_param.imag])
    normals = normals / np.linalg.norm(normals, axis=0)
    centers = sources - normals * radius

    mode_nr = 25
    density = np.cos(mode_nr * t)

    return sources, centers, normals, density, h, radius


def test_recurrence_laplace_2d_ellipse():
    r"""
    Tests recurrence + qbx code.
    """

    # ------------- 1. Define PDE, Green's Function
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = (-1/(2*np.pi)) * sp.log(sp.sqrt((var[0]-var_t[0])**2 +
                                            (var[1]-var_t[1])**2))

    p = 4
    err = []
    for n_p in range(200, 1001, 200):
        sources, centers, normals, density, h, radius = _create_ellipse(n_p)
        strengths = h * density
        exp_res = recurrence_qbx_lp(sources, centers, normals,
                                    strengths, radius, laplace2d,
                                    g_x_y, 2, p)
        qbx_res = _qbx_lp_laplace_general(sources, sources, centers,
                                          radius, strengths, p)
        # qbx_res,_ = lpot_eval_circle(sources.shape[1], p)
        err.append(np.max(np.abs(exp_res - qbx_res)))
    assert np.max(err) <= 1e-13


def test_recurrence_helmholtz_2d_ellipse():
    r"""
    Tests recurrence + qbx code.
    """
    # ------------- 1. Define PDE, Green's Function
    w = make_identity_diff_op(2)
    helmholtz2d = laplacian(w) + w

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    k = 1
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2)
    g_x_y = (1j/4) * hankel1(0, k * abs_dist)

    p = 4
    err = []
    for n_p in range(200, 1001, 200):
        sources, centers, normals, density, h, radius = _create_ellipse(n_p)
        strengths = h * density
        exp_res = recurrence_qbx_lp(sources, centers, normals, strengths,
        radius, helmholtz2d, g_x_y, 2, p)
        qbx_res = _qbx_lp_helmholtz_general(sources, sources,
                                            centers, radius, strengths, p)
        err.append(np.max(np.abs(exp_res - qbx_res)))
    assert np.max(err) <= 1e-13
