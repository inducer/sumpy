from sumpy.expansion.diff_op import (
    laplacian,
    make_identity_diff_op,
)

import numpy as np
from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf  # noqa: F401
from sumpy.expansion.local import LineTaylorLocalExpansion, VolumeTaylorLocalExpansion


actx_factory = _acf
expn_class = LineTaylorLocalExpansion

actx = actx_factory()

from sumpy.kernel import LaplaceKernel
lknl = LaplaceKernel(2)

from sumpy.qbx import LayerPotential
from sumpy.recurrence import recurrence_qbx_lp, _make_sympy_vec

def qbx_lp_laplace_general(sources,targets,centers,radius,strengths,order):
        lpot = LayerPotential(actx.context,
        expansion=expn_class(lknl, order),
        target_kernels=(lknl,),
        source_kernels=(lknl,))

        #print(lpot.get_kernel())
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

def create_ellipse(n_p):
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

    #------------- 1. Define PDE, Green's Function
    w = make_identity_diff_op(2)
    laplace2d = laplacian(w)

    var = _make_sympy_vec("x", 2)
    var_t = _make_sympy_vec("t", 2)
    g_x_y = (-1/(2*np.pi)) * sp.log(sp.sqrt((var[0]-var_t[0])**2 + (var[1]-var_t[1])**2))

    p = 4
    err = []
    for n_p in range(200, 1001, 200):
        sources, centers, normals, density, h, radius = create_ellipse(n_p)
        strengths = h * density
        exp_res = recurrence_qbx_lp(sources, centers, normals, strengths, radius, laplace2d, g_x_y, p)
        qbx_res = qbx_lp_laplace_general(sources, sources, centers, radius, strengths, p)
        #qbx_res,_ = lpot_eval_circle(sources.shape[1], p)
        err.append(np.max(exp_res - qbx_res))

    print(err)

