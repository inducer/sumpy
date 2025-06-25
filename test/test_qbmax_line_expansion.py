import numpy as np
import pytest
import pyopencl as cl

import meshmode.mesh.generation as mgen
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import InterpolatoryQuadratureSimplexGroupFactory

from pytential import GeometryCollection, bind, sym
from pytential.qbx import QBXLayerPotentialSource

from sumpy.kernel import YukawaKernel
from sumpy.expansion.local import AsymptoticDividingLineTaylorExpansion
from sumpy.qbx import LayerPotentialMatrixGenerator

from arraycontext import flatten
from pytools.convergence import EOCRecorder

import mpmath


def asym_yukawa(dim, lam=None):
    """Asymptotic function of the Yukawa kernel."""
    from pymbolic import primitives, var
    from sumpy.symbolic import pymbolic_real_norm_2, SpatialConstant
    
    b = pymbolic_real_norm_2(primitives.make_sym_vector("b", dim))
    
    if lam:
        expr = var("exp")(-lam * b * (1 - var('tau')))
    else:
        lam = SpatialConstant("lam")
        expr = var("exp")(-lam * b * (1 - var('tau')))
    return expr


def utrue(lam, r, tau, targets_h, f_mode, side):
    """Test convergence of QBMAX (asymptotic Yukawa expansion) on a unit circle 
    with density φ(y) = exp(imθ_y)"""
    mpmath.mp.dps = 25
    
    angles = np.arctan2(targets_h[1, :], targets_h[0, :])
    n_points = len(angles)
    result = np.zeros(n_points, dtype=np.complex128)
    
    for i in range(n_points):
        r_i = float(r[i])
        
        if side == -1:
            coeff = float(mpmath.besselk(f_mode, lam) * 
                         mpmath.besseli(f_mode, lam * (1 - (1 - tau) * r_i)))
        else:
            coeff = float(mpmath.besseli(f_mode, lam) * 
                         mpmath.besselk(f_mode, lam * (1 + (1 - tau) * r_i)))
        
        result[i] = coeff * np.exp(1j * f_mode * angles[i])
    
    return result


def test_qbmax_yukawa_convergence():
    """Test convergence of QBMAX (asymptotic Yukawa expansion) for various τ values."""
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    
    lam = 15
    f_mode = 7
    nelements = [20, 40, 60]
    qbx_order = 4
    target_order = 5
    upsampling_factor = 5
    extra_kwargs = {'lam': lam}

    knl = YukawaKernel(2)
    asym_knl = asym_yukawa(2)

    np.random.seed(42) 
    t = np.random.uniform(0, 1, 10)
    targets_h = np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    targets = actx.from_numpy(targets_h)

    for tau in [0, 0.5, 1]:
        eoc_in = EOCRecorder()
        eoc_out = EOCRecorder()
        
        asym_expn = AsymptoticDividingLineTaylorExpansion(
            knl, asym_knl, qbx_order, tau=tau)
        
        for nelement in nelements:
            mesh = mgen.make_curve_mesh(
                mgen.circle, np.linspace(0, 1, nelement+1), target_order)
            pre_density_discr = Discretization(
                actx, mesh, 
                InterpolatoryQuadratureSimplexGroupFactory(target_order))
            
            qbx = QBXLayerPotentialSource(
                pre_density_discr, 
                upsampling_factor * target_order, 
                qbx_order, 
                fmm_order=False)
            
            places = GeometryCollection({"qbx": qbx}, auto_where=('qbx'))
            
            source_discr = places.get_discretization(
                'qbx', sym.QBX_SOURCE_QUAD_STAGE2)
            sources = source_discr.nodes()
            sources_h = actx.to_numpy(flatten(sources, actx)).reshape(2, -1)
            
            dofdesc = sym.DOFDescriptor("qbx", sym.QBX_SOURCE_QUAD_STAGE2)
            weights_nodes = bind(
                places, 
                sym.weights_and_area_elements(
                    ambient_dim=2, dim=1, dofdesc=dofdesc))(actx)
            weights_nodes_h = actx.to_numpy(flatten(weights_nodes, actx))
            
            angle = np.arctan2(sources_h[1, :], sources_h[0, :])
            sigma = np.exp(1j * f_mode * angle) * weights_nodes_h
            
            expansion_radii_h = np.ones(targets_h.shape[1]) * np.pi / nelement
            centers_in = actx.from_numpy((1 - expansion_radii_h) * targets_h)
            centers_out = actx.from_numpy((1 + expansion_radii_h) * targets_h)
            
            mat_asym_gen = LayerPotentialMatrixGenerator(
                actx.context,
                expansion=asym_expn,
                source_kernels=(knl,),
                target_kernels=(knl,))
            
            _, (mat_asym_in,) = mat_asym_gen(
                actx.queue,
                targets=targets,
                sources=actx.from_numpy(sources_h),
                expansion_radii=expansion_radii_h,
                centers=centers_in,
                **extra_kwargs)
            
            mat_asym_in = actx.to_numpy(mat_asym_in)
            weighted_mat_asym_in = mat_asym_in * sigma[None, :]
            asym_eval_in = (np.sum(weighted_mat_asym_in, axis=1) * 
                           np.exp(-lam * expansion_radii_h * (1 - tau)))
            
            _, (mat_asym_out,) = mat_asym_gen(
                actx.queue,
                targets=targets,
                sources=actx.from_numpy(sources_h),
                expansion_radii=expansion_radii_h,
                centers=centers_out,
                **extra_kwargs)
            
            mat_asym_out = actx.to_numpy(mat_asym_out)
            weighted_mat_asym_out = mat_asym_out * sigma[None, :]
            asym_eval_out = (np.sum(weighted_mat_asym_out, axis=1) * 
                            np.exp(-lam * expansion_radii_h * (1 - tau)))
            
            utrue_in = utrue(lam, expansion_radii_h, tau, targets_h, f_mode, -1)
            utrue_out = utrue(lam, expansion_radii_h, tau, targets_h, f_mode, 1)
            
            err_in = (np.max(np.abs(asym_eval_in - utrue_in)) / 
                     np.max(np.abs(utrue_in)))
            err_out = (np.max(np.abs(asym_eval_out - utrue_out)) / 
                      np.max(np.abs(utrue_out)))
            
            h_max = actx.to_numpy(
                bind(places, sym.h_max(places.ambient_dim))(actx))
            
            eoc_in.add_data_point(h_max, err_in)
            eoc_out.add_data_point(h_max, err_out)
            
        assert eoc_in.order_estimate() > qbx_order, \
                f"Interior convergence too slow: {eoc_in.order_estimate()}"

        assert eoc_out.order_estimate() > qbx_order, \
                f"Exterior convergence too slow: {eoc_out.order_estimate()}"


if __name__ == "__main__":
    test_qbmax_yukawa_convergence()
