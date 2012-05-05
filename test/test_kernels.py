from __future__ import division

import numpy as np
import numpy.linalg as la
import sys
import pytools.test




def have_cl():
    try:
        import pyopencl
        return True
    except:
        return False

if have_cl():
    import pyopencl.array as cl_array
    import pyopencl as cl
    from pyopencl.tools import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests




@pytools.test.mark_test.opencl
def test_p2p(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dimensions = 3
    n = 5000

    from sumpy.kernel import LaplaceKernel, TargetDerivative
    from sumpy.p2p import P2P
    lknl = LaplaceKernel(dimensions)
    knl = P2P(ctx, [lknl, TargetDerivative(0, lknl)], exclude_self=False)

    targets = np.random.rand(dimensions, n)
    sources = np.random.rand(dimensions, n)

    #from sumpy.tools import vector_to_device
    #targets_dev = vector_to_device(queue, targets)
    #sources_dev = vector_to_device(queue, sources)
    targets_dev = cl_array.to_device(queue, targets.T.copy())
    sources_dev = cl_array.to_device(queue, sources.T.copy())
    strengths_dev = cl_array.empty(queue, n, dtype=np.float64)
    strengths_dev.fill(1)

    evt, (potential_dev, x_derivative) = knl(
            queue, targets_dev, sources_dev, [strengths_dev])

    potential = potential_dev.get()
    potential_host = np.empty_like(potential)
    strengths = strengths_dev.get()

    targets = targets.T
    sources = sources.T
    for itarg in xrange(n):
        potential_host[itarg] = np.sum(
                strengths
                /
                np.sum((targets[itarg] - sources)**2, axis=-1)**0.5)

    potential_host *= 1/(4*np.pi)
    rel_err = la.norm(potential - potential_host)/la.norm(potential_host)
    print rel_err
    assert rel_err < 1e-3




@pytools.test.mark_test.opencl
def test_p2m2p(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    res = 100

    dimensions = 3
    sources = np.random.rand(dimensions, 5).astype(np.float32)
    #targets = np.random.rand(dimensions, 10).astype(np.float32)
    targets = np.mgrid[-2:2:res*1j, -2:2:res*1j, 2:2:1j].reshape(3, -1).astype(np.float32)
    centers = np.array([[0.5]]*dimensions).astype(np.float32)

    from sumpy.tools import vector_to_device
    targets_dev = vector_to_device(queue, targets)
    sources_dev = vector_to_device(queue, sources)
    centers_dev = vector_to_device(queue, centers)
    strengths_dev = cl_array.empty(queue, sources.shape[1], dtype=np.float32)
    strengths_dev.fill(1)

    cell_idx_to_particle_offset = np.array([0], dtype=np.uint32)
    cell_idx_to_particle_cnt_src = np.array([sources.shape[1]], dtype=np.uint32)
    cell_idx_to_particle_cnt_tgt = np.array([targets.shape[1]], dtype=np.uint32)

    from pyopencl.array import to_device
    cell_idx_to_particle_offset_dev = to_device(queue, cell_idx_to_particle_offset)
    cell_idx_to_particle_cnt_src_dev = to_device(queue, cell_idx_to_particle_cnt_src)
    cell_idx_to_particle_cnt_tgt_dev = to_device(queue, cell_idx_to_particle_cnt_tgt)

    from sumpy.symbolic import make_coulomb_kernel_in
    from sumpy.expansion import TaylorMultipoleExpansion
    texp = TaylorMultipoleExpansion(
            make_coulomb_kernel_in("b", dimensions),
            order=3, dimensions=dimensions)

    coeff_dtype = np.float32

    # {{{ apply P2M

    from sumpy.p2m import P2MKernel
    p2m = P2MKernel(ctx, texp)
    mpole_coeff = p2m(cell_idx_to_particle_offset_dev,
            cell_idx_to_particle_cnt_src_dev,
            sources_dev, strengths_dev, centers_dev, coeff_dtype)

    # }}}

    # {{{ apply M2P

    output_maps = [
            lambda expr: expr,
            lambda expr: sp.diff(expr, sp.Symbol("t0"))
            ]

    m2p_ilist_starts = np.array([0, 1], dtype=np.uint32)
    m2p_ilist_mpole_offsets = np.array([0], dtype=np.uint32)

    m2p_ilist_starts_dev = to_device(queue, m2p_ilist_starts)
    m2p_ilist_mpole_offsets_dev = to_device(queue, m2p_ilist_mpole_offsets)

    from sumpy.m2p import M2PKernel
    m2p = M2PKernel(ctx, texp, output_maps=output_maps)
    potential_dev, x_derivative = m2p(targets_dev, m2p_ilist_starts_dev, m2p_ilist_mpole_offsets_dev, mpole_coeff, 
            cell_idx_to_particle_offset_dev,
            cell_idx_to_particle_cnt_tgt_dev)

    # }}}

    # {{{ compute (direct) reference solution

    from sumpy.p2p import P2PKernel
    from sumpy.symbolic import make_coulomb_kernel_ts
    coulomb_knl = make_coulomb_kernel_ts(dimensions)

    knl = P2PKernel(ctx, dimensions, 
            exprs=[f(coulomb_knl) for f in output_maps], exclude_self=False)

    potential_dev_direct, x_derivative_dir = knl(targets_dev, sources_dev, strengths_dev)

    if 0:
        import matplotlib.pyplot as pt
        pt.imshow((potential_dev-potential_dev_direct).get().reshape(res, res))
        pt.colorbar()
        pt.show()

    assert la.norm((potential_dev-potential_dev_direct).get())/res**2 < 1e-3

    # }}}






@pytools.test.mark_test.opencl
def test_p2m2m2p(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    res = 100

    dimensions = 3
    sources = np.random.rand(dimensions, 5).astype(np.float32)
    #targets = np.random.rand(dimensions, 10).astype(np.float32)
    targets = np.mgrid[-2:2:res*1j, -2:2:res*1j, 2:2:1j].reshape(3, -1).astype(np.float32)
    centers = np.array([[0.5]]*dimensions).astype(np.float32)

    from sumpy.tools import vector_to_device
    targets_dev = vector_to_device(queue, targets)
    sources_dev = vector_to_device(queue, sources)
    centers_dev = vector_to_device(queue, centers)
    strengths_dev = cl_array.empty(queue, sources.shape[1], dtype=np.float32)
    strengths_dev.fill(1)

    cell_idx_to_particle_offset = np.array([0], dtype=np.uint32)
    cell_idx_to_particle_cnt_src = np.array([sources.shape[1]], dtype=np.uint32)
    cell_idx_to_particle_cnt_tgt = np.array([targets.shape[1]], dtype=np.uint32)

    from pyopencl.array import to_device
    cell_idx_to_particle_offset_dev = to_device(queue, cell_idx_to_particle_offset)
    cell_idx_to_particle_cnt_src_dev = to_device(queue, cell_idx_to_particle_cnt_src)
    cell_idx_to_particle_cnt_tgt_dev = to_device(queue, cell_idx_to_particle_cnt_tgt)

    from sumpy.symbolic import make_coulomb_kernel_in
    from sumpy.expansion import TaylorMultipoleExpansion
    texp = TaylorMultipoleExpansion(
            make_coulomb_kernel_in("b", dimensions),
            order=3, dimensions=dimensions)

    def get_coefficient_expr(idx):
        return sp.Symbol("coeff%d" % idx)

    for i in texp.m2m_exprs(get_coefficient_expr):
        print i

    print "-------------------------------------"
    from sumpy.expansion import TaylorLocalExpansion
    locexp = TaylorLocalExpansion(order=2, dimensions=3)

    for i in texp.m2l_exprs(locexp, get_coefficient_expr):
        print i

    1/0

    coeff_dtype = np.float32

    # {{{ apply P2M

    from sumpy.p2m import P2MKernel
    p2m = P2MKernel(ctx, texp)
    mpole_coeff = p2m(cell_idx_to_particle_offset_dev,
            cell_idx_to_particle_cnt_src_dev,
            sources_dev, strengths_dev, centers_dev, coeff_dtype)

    # }}}

    # {{{ apply M2P

    output_maps = [
            lambda expr: expr,
            lambda expr: sp.diff(expr, sp.Symbol("t0"))
            ]

    m2p_ilist_starts = np.array([0, 1], dtype=np.uint32)
    m2p_ilist_mpole_offsets = np.array([0], dtype=np.uint32)

    m2p_ilist_starts_dev = to_device(queue, m2p_ilist_starts)
    m2p_ilist_mpole_offsets_dev = to_device(queue, m2p_ilist_mpole_offsets)

    from sumpy.m2p import M2PKernel
    m2p = M2PKernel(ctx, texp, output_maps=output_maps)
    potential_dev, x_derivative = m2p(targets_dev, m2p_ilist_starts_dev, m2p_ilist_mpole_offsets_dev, mpole_coeff, 
            cell_idx_to_particle_offset_dev,
            cell_idx_to_particle_cnt_tgt_dev)

    # }}}

    # {{{ compute (direct) reference solution

    from sumpy.p2p import P2PKernel
    from sumpy.symbolic import make_coulomb_kernel_ts
    coulomb_knl = make_coulomb_kernel_ts(dimensions)

    knl = P2PKernel(ctx, dimensions, 
            exprs=[f(coulomb_knl) for f in output_maps], exclude_self=False)

    potential_dev_direct, x_derivative_dir = knl(targets_dev, sources_dev, strengths_dev)

    if 0:
        import matplotlib.pyplot as pt
        pt.imshow((potential_dev-potential_dev_direct).get().reshape(res, res))
        pt.colorbar()
        pt.show()

    assert la.norm((potential_dev-potential_dev_direct).get())/res**2 < 1e-3

    # }}}




# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pyopencl as cl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
