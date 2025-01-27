r"""
With the functionality in this module, we test recurrence
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
import meshmode.mesh.generation as mgen  # type: ignore
import numpy as np
import sympy as sp
from meshmode import _acf as _acf_meshmode  # type: ignore
from meshmode.discretization import Discretization  # type: ignore
from meshmode.discretization.poly_element import (  # type: ignore
    default_simplex_group_factory,
)
from pytential import bind, sym  # type: ignore
from sympy import hankel1

from sumpy.array_context import _acf
from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import HelmholtzKernel, LaplaceKernel
from sumpy.qbx import LayerPotential
from sumpy.recurrence_qbx import (
    _compute_rotated_shifted_coordinates,
    _make_sympy_vec,
    recurrence_qbx_lp,
)


actx_factory = _acf
ExpnClass = LineTaylorLocalExpansion

actx = actx_factory()
lknl2d = LaplaceKernel(2)
hknl2d = HelmholtzKernel(2)
lknl3d = LaplaceKernel(3)
hknl3d = HelmholtzKernel(3)


def _qbx_lp_general(knl, sources, targets, centers, radius,
                                strengths, order, k=0):
    lpot = LayerPotential(actx.context,
    expansion=ExpnClass(knl, order),
    target_kernels=(knl,),
    source_kernels=(knl,))

    # print(lpot.get_kernel())
    expansion_radii = actx.from_numpy(radius * np.ones(sources.shape[1]))
    sources = actx.from_numpy(sources)
    targets = actx.from_numpy(targets)
    centers = actx.from_numpy(centers)

    strengths = (strengths,)
    if k == 0:
        _evt, (result_qbx,) = lpot(
                actx.queue,
                targets, sources, centers, strengths,
                expansion_radii=expansion_radii)
    else:
        _evt, (result_qbx,) = lpot(
                actx.queue,
                targets, sources, centers, strengths,
                expansion_radii=expansion_radii,
                k=1)

    result_qbx = actx.to_numpy(result_qbx)

    return result_qbx


def _create_ellipse(n_p):
    h = 9.688 / n_p
    radius = 7*h * 1/40
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


def _create_sphere(refinment_rounds, exp_radius):
    target_order = 4

    actx_m = _acf_meshmode()
    mesh = mgen.generate_sphere(1.0, target_order,
                                uniform_refinement_rounds=refinment_rounds)
    grp_factory = default_simplex_group_factory(3, target_order)
    discr = Discretization(actx_m, mesh, grp_factory)
    nodes = actx_m.to_numpy(discr.nodes())
    sources = np.array([nodes[0][0].reshape(-1),
                        nodes[1][0].reshape(-1), nodes[2][0].reshape(-1)])

    area_weight_a = bind(discr, sym.QWeight()*sym.area_element(3))(actx_m)
    area_weight = actx_m.to_numpy(area_weight_a)[0]
    area_weight = area_weight.reshape(-1)

    normals_a = bind(discr, sym.normal(3))(actx_m).as_vector(dtype=object)
    normals_a = actx_m.to_numpy(normals_a)
    normals = np.array([normals_a[0][0].reshape(-1), normals_a[1][0].reshape(-1),
                        normals_a[2][0].reshape(-1)])

    radius = exp_radius
    centers = sources - radius * normals

    return sources, centers, normals, area_weight, radius


def test_compute_rotated_shifted_coordinates():
    r"""
    Tests rotated shifted code.
    """
    sources = np.array([[1], [2], [2]])
    centers = np.array([[0], [0], [0]])
    normals = np.array([[1], [0], [0]])
    cts = _compute_rotated_shifted_coordinates(sources, centers, normals)
    assert np.sqrt(cts[1]**2 + cts[2]**2) - np.sqrt(8) <= 1e-12


def test_recurrence_laplace_3d_sphere():
    r"""
    Tests reccurrence + qbx laplace 3d on sphere
    """
    radius = 0.0001
    sources, centers, normals, area_weight, radius = _create_sphere(1, radius)

    out = _qbx_lp_general(lknl3d, sources, sources, centers, radius,
                                   area_weight, 4)

    w = make_identity_diff_op(3)
    laplace3d = laplacian(w)
    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2
                       + (var[2]-var_t[2])**2)
    g_x_y = 1/(4*np.pi) * 1/abs_dist

    exp_res = recurrence_qbx_lp(sources, centers, normals, area_weight,
                                radius, laplace3d, g_x_y, 3, 4)

    assert (np.max(exp_res-out)/np.max(abs(exp_res))) <= 1e-12


def test_recurrence_helmholtz_3d_sphere():
    r"""
    Tests reccurrence + qbx helmholtz 3d on sphere
    """
    # import time
    radius = 0.0001
    sources, centers, normals, area_weight, radius = _create_sphere(2, radius)

    # start = time.time()
    out = _qbx_lp_general(hknl3d, sources, sources, centers, radius,
                                   np.ones(area_weight.shape), 1, 1)
    # end = time.time()
    # length1 = end - start

    w = make_identity_diff_op(3)
    helmholtz3d = laplacian(w) + w
    var = _make_sympy_vec("x", 3)
    var_t = _make_sympy_vec("t", 3)
    abs_dist = sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2
                       + (var[2]-var_t[2])**2)
    g_x_y = (1/(4*np.pi)) * sp.exp(1j * abs_dist) / abs_dist

    # start = time.time()
    exp_res = recurrence_qbx_lp(sources, centers, normals, np.ones(area_weight.shape),
                                radius, helmholtz3d, g_x_y, 3, 1)
    # end = time.time()
    # length2 = end - start
    # print(sources.shape[1], length1, length2)

    assert np.max(abs(out - exp_res)) <= 1e-8


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
        qbx_res = _qbx_lp_general(lknl2d, sources, sources, centers,
                                          radius, strengths, p)
        # qbx_res,_ = lpot_eval_circle(sources.shape[1], p)
        err.append(np.max(np.abs(exp_res - qbx_res))/np.max(np.abs(qbx_res)))
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

    p = 5
    err = []
    for n_p in range(200, 1001, 200):
        sources, centers, normals, density, h, radius = _create_ellipse(n_p)
        strengths = h * density
        exp_res = recurrence_qbx_lp(sources, centers, normals, strengths,
        radius, helmholtz2d, g_x_y, 2, p)
        qbx_res = _qbx_lp_general(hknl2d, sources, sources,
                                  centers, radius, strengths, p, 1)
        err.append(np.max(np.abs(exp_res - qbx_res)))
    assert np.max(err) <= 1e-13


# ============ Plotting Functionality
def _construct_laplace_axis_2d(orders, resolutions):
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = (-1/(2*np.pi)) * sp.log(sp.sqrt((var[0]-var_t[0])**2 +
                                            (var[1]-var_t[1])**2))

    err = []
    for p in orders:
        err_per_order = []
        for n_p in resolutions:
            sources, centers, normals, density, h, radius = _create_ellipse(n_p)
            strengths = h * density
            exp_res = recurrence_qbx_lp(sources, centers, normals,
                                        strengths, radius, laplace2d,
                                        g_x_y, 2, p)
            qbx_res = _qbx_lp_general(lknl2d, sources, sources, centers,
                                            radius, strengths, p)
            # qbx_res,_ = lpot_eval_circle(sources.shape[1], p)
            err_per_order.append(np.max(np.abs(exp_res - qbx_res)/
                                        np.max(np.abs(qbx_res))))
        err.append(err_per_order)

    return err

import matplotlib.pyplot as plt
orders = [8]
resolutions = range(200, 800, 200)
err_mat = _construct_laplace_axis_2d(orders, resolutions)
for i in range(len(orders)):
    plt.plot(resolutions, err_mat[i], label="order ="+str(orders[i]))
plt.xlabel("Number of Nodes")
plt.ylabel("Error")
plt.title("2D Ellipse LP Eval Error")
plt.legend()
plt.show()



