import pytest
import numpy as np

import pyopencl as cl
from arraycontext import flatten
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import InterpolatoryQuadratureSimplexGroupFactory
from meshmode.mesh.generation import make_curve_mesh, starfish

from pytential.qbx import QBXLayerPotentialSource
from pytential import GeometryCollection, bind, sym

from sumpy.kernel import AxisTargetDerivative, LaplaceKernel
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.qbx import LayerPotentialMatrixGenerator
from pytools.convergence import EOCRecorder


def starfish_parametrization(t, n_arms=5, amplitude=0.25):
    """Parametrization (x(θ), y(θ)) = r(θ)(cos(θ), sin(θ)), where r(θ) = 1 + amplitude * sin(n_arms * θ)."""
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
    np.random.seed(42)  
    t = np.random.uniform(0, 1, ntargets)
    targets_h, _, targets_normals_h, jac = starfish_parametrization(
        t, n_arms=5, amplitude=0.25
    )
    targets = actx.from_numpy(targets_h)

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
        places = GeometryCollection({"qbx": qbx}, auto_where=('qbx'))
        
        source_discr = places.get_discretization('qbx', sym.QBX_SOURCE_QUAD_STAGE2)
        sources_h = actx.to_numpy(flatten(source_discr.nodes(), actx)).reshape(2, -1)
        sources = actx.from_numpy(sources_h)
        
        dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
        weights_nodes = bind(
            places, 
            sym.weights_and_area_elements(ambient_dim=2, dim=1, dofdesc=dofdesc)
        )(actx)
        weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))
        
        expansion_radii_h = jac / (2 * nelement)
        centers_in = actx.from_numpy(targets_h - targets_normals_h * expansion_radii_h)
        centers_out = actx.from_numpy(targets_h + targets_normals_h * expansion_radii_h)
        
        mat_gen_x = LayerPotentialMatrixGenerator(
            actx.context,
            expansion=LineTaylorLocalExpansion(knl, qbx_order),
            source_kernels=(knl,),
            target_kernels=(AxisTargetDerivative(0, knl),)
        )
        
        mat_gen_y = LayerPotentialMatrixGenerator(
            actx.context,
            expansion=LineTaylorLocalExpansion(knl, qbx_order),
            source_kernels=(knl,),
            target_kernels=(AxisTargetDerivative(1, knl),)
        )
        
        _, (mat_in_x,) = mat_gen_x(
            actx.queue,
            targets=targets,
            sources=sources,
            expansion_radii=expansion_radii_h,
            centers=centers_in,
        )
        mat_in_x = actx.to_numpy(mat_in_x)
        weighted_mat_in_x = mat_in_x * weights_nodes_h[None, :]
        
        _, (mat_in_y,) = mat_gen_y(
            actx.queue,
            targets=targets,
            sources=sources,
            expansion_radii=expansion_radii_h,
            centers=centers_in,
        )
        mat_in_y = actx.to_numpy(mat_in_y)
        weighted_mat_in_y = mat_in_y * weights_nodes_h[None, :]
        
        _, (mat_out_x,) = mat_gen_x(
            actx.queue,
            targets=targets,
            sources=sources,
            expansion_radii=expansion_radii_h,
            centers=centers_out,
        )
        mat_out_x = actx.to_numpy(mat_out_x)
        weighted_mat_out_x = mat_out_x * weights_nodes_h[None, :]
        
        _, (mat_out_y,) = mat_gen_y(
            actx.queue,
            targets=targets,
            sources=sources,
            expansion_radii=expansion_radii_h,
            centers=centers_out,
        )
        mat_out_y = actx.to_numpy(mat_out_y)
        weighted_mat_out_y = mat_out_y * weights_nodes_h[None, :]
        
        eval_in = (weighted_mat_in_x.sum(axis=1) * targets_normals_h[0] + 
                   weighted_mat_in_y.sum(axis=1) * targets_normals_h[1])
        eval_out = (weighted_mat_out_x.sum(axis=1) * targets_normals_h[0] + 
                    weighted_mat_out_y.sum(axis=1) * targets_normals_h[1])
        
        # check jump relation: S'_int - S'_ext = sigma (=1 for constant density)
        jump_error = np.abs(eval_in - eval_out - 1)
        
        h_max = actx.to_numpy(bind(places, sym.h_max(places.ambient_dim))(actx))
        eocrec.add_data_point(h_max, np.max(jump_error))
        
    assert eocrec.order_estimate() > qbx_order - 1
  
    
if __name__ == "__main__":
    pytest.main([__file__])