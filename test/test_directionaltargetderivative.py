from __future__ import annotations

import numpy as np
import pytest
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)
from meshmode.mesh.generation import make_curve_mesh, starfish
from pytential import GeometryCollection, bind, sym
from pytential.qbx import QBXLayerPotentialSource

import pyopencl as cl
from arraycontext import flatten
from pytools.convergence import EOCRecorder

from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import AxisTargetDerivative, LaplaceKernel


def starfish_parametrization(t, n_arms=5, amplitude=0.25):
    """
    Parametrization:
            (x(θ), y(θ)) = r(θ)(cos(θ), sin(θ)), r(θ) = 1 + amplitude * sin(n_arms * θ).
    It is used to compute normal vectors and expansion radius at different
    refinement levels for arbitrary boundary targets.
    """

    theta = 2 * np.pi * t

    r = 1 + amplitude * np.sin(n_arms * theta)
    dr_dt = amplitude * n_arms * 2 * np.pi * np.cos(n_arms * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    dx_dt = dr_dt * np.cos(theta) - r * np.sin(theta) * 2 * np.pi
    dy_dt = dr_dt * np.sin(theta) + r * np.cos(theta) * 2 * np.pi

    jacobian_norm = np.sqrt(dx_dt**2 + dy_dt**2)

    tangent_x = dx_dt / jacobian_norm
    tangent_y = dy_dt / jacobian_norm

    normal_x = tangent_y
    normal_y = -tangent_x

    coords = np.vstack([x, y])
    tangents = np.vstack([tangent_x, tangent_y])
    normals = np.vstack([normal_x, normal_y])

    return coords, tangents, normals, jacobian_norm


@pytest.mark.parametrize("kernel_type", ["laplace"])
def test_lpot_dx_jump_relation_convergence(kernel_type):
    """Test convergence of jump relations for single layer potential derivatives."""

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    if kernel_type == "laplace":
        knl = LaplaceKernel(2)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    qbx_order = 5
    nelements = [100, 150, 200]
    target_order = 5
    upsampling_factor = 5

    ntargets = 20
    rng = np.random.default_rng(42)
    t = rng.uniform(0, 1, ntargets)
    targets_h, _, targets_normals_h, jac = starfish_parametrization(
        t, n_arms=5, amplitude=0.25
    )
    targets = actx.from_numpy(targets_h)

    from sumpy.qbx import LayerPotential
    expansion = LineTaylorLocalExpansion(knl, qbx_order)
    lplot_dx = LayerPotential(
        actx.context,
        expansion=expansion,
        target_kernels=(AxisTargetDerivative(0, knl),),
        source_kernels=(knl,)
    )
    lplot_dy = LayerPotential(
        actx.context,
        expansion=expansion,
        target_kernels=(AxisTargetDerivative(1, knl),),
        source_kernels=(knl,)
    )
    eocrec = EOCRecorder()

    for nelement in nelements:
        mesh = make_curve_mesh(starfish, np.linspace(0, 1, nelement + 1), target_order)
        pre_density_discr = Discretization(
            actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
        )

        qbx = QBXLayerPotentialSource(
            pre_density_discr,
            upsampling_factor * target_order,
            qbx_order,
            fmm_order=False
        )
        places = GeometryCollection({"qbx": qbx}, auto_where=("qbx"))

        source_discr = places.get_discretization("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
        sources_h = actx.to_numpy(flatten(source_discr.nodes(), actx)).reshape(2, -1)
        sources = actx.from_numpy(sources_h)

        dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
        weights_nodes = bind(
            places,
            sym.weights_and_area_elements(ambient_dim=2, dim=1, dofdesc=dofdesc)
        )(actx)
        weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))
        strengths = (actx.from_numpy(weights_nodes_h),)

        expansion_radii_h = jac / (2 * nelement)
        centers_in = actx.from_numpy(targets_h - targets_normals_h * expansion_radii_h)
        centers_out = actx.from_numpy(targets_h + targets_normals_h * expansion_radii_h)

        _, (eval_in_dx,) = lplot_dx(
            actx.queue,
            targets, sources, centers_in, strengths,
            expansion_radii=expansion_radii_h
        )

        _, (eval_in_dy,) = lplot_dy(
            actx.queue,
            targets, sources, centers_in, strengths,
            expansion_radii=expansion_radii_h
        )

        _, (eval_out_dx,) = lplot_dx(
            actx.queue,
            targets, sources, centers_out, strengths,
            expansion_radii=expansion_radii_h
        )

        _, (eval_out_dy,) = lplot_dy(
            actx.queue,
            targets, sources, centers_out, strengths,
            expansion_radii=expansion_radii_h
        )

        eval_in_dx = actx.to_numpy(eval_in_dx)
        eval_in_dy = actx.to_numpy(eval_in_dy)
        eval_out_dx = actx.to_numpy(eval_out_dx)
        eval_out_dy = actx.to_numpy(eval_out_dy)

        eval_in = eval_in_dx * targets_normals_h[0] + \
                   eval_in_dy * targets_normals_h[1]
        eval_out = eval_out_dx * targets_normals_h[0] + \
                   eval_out_dy * targets_normals_h[1]

        # check jump relation: S'_int - S'_ext = sigma (=1 for constant density)
        jump_error = np.abs(eval_in - eval_out - 1)

        h_max = actx.to_numpy(bind(places, sym.h_max(places.ambient_dim))(actx))
        eocrec.add_data_point(h_max, np.max(jump_error))

    assert eocrec.order_estimate() > qbx_order - 1


if __name__ == "__main__":
    pytest.main([__file__])
