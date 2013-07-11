from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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
import numpy.linalg as la
import sys
import pytools.test

import pytest
import pyopencl.array as cl_array
import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion

import logging
logger = logging.getLogger(__name__)


@pytest.mark.opencl
def test_p2p(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dimensions = 3
    n = 5000

    from sumpy.kernel import LaplaceKernel, AxisTargetDerivative
    from sumpy.p2p import P2P
    lknl = LaplaceKernel(dimensions)
    knl = P2P(ctx,
            [lknl, AxisTargetDerivative(0, lknl)],
            exclude_self=False)

    targets = np.random.rand(dimensions, n)
    sources = np.random.rand(dimensions, n)

    strengths = np.ones(n, dtype=np.float64)

    evt, (potential, x_derivative) = knl(
            queue, targets, sources, [strengths],
            out_host=True)

    potential_ref = np.empty_like(potential)

    targets = targets.T
    sources = sources.T
    for itarg in xrange(n):
        potential_ref[itarg] = np.sum(
                strengths
                /
                np.sum((targets[itarg] - sources)**2, axis=-1)**0.5)

    potential_ref *= 1/(4*np.pi)

    rel_err = la.norm(potential - potential_ref)/la.norm(potential_ref)
    print rel_err
    assert rel_err < 1e-3


@pytools.test.mark_test.opencl
@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("expn_class", [
    VolumeTaylorLocalExpansion,
    VolumeTaylorMultipoleExpansion,
    ])
def test_p2e2p(ctx_getter, order, expn_class):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 100
    nsources = 500

    dim = 2

    from sumpy.kernel import LaplaceKernel, AxisTargetDerivative
    knl = LaplaceKernel(dim)
    out_kernels = [
            knl, AxisTargetDerivative(0, knl)
            ]
    texp = expn_class(knl, order=order)

    from sumpy.p2e import P2E
    p2m = P2E(ctx, texp, out_kernels)

    from sumpy.e2p import E2P
    m2p = E2P(ctx, texp, out_kernels)

    from sumpy.p2p import P2P
    p2p = P2P(ctx, out_kernels, exclude_self=False)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for dist in [3, 5, 7]:
        from sumpy.expansion.local import LocalExpansionBase
        if issubclass(expn_class, LocalExpansionBase):
            centers = np.array([0.5+dist, 0.5], dtype=np.float64).reshape(dim, 1)
        else:
            centers = np.array([0.5, 0.5], dtype=np.float64).reshape(dim, 1)

        sources = np.random.rand(dim, nsources).astype(np.float64)
        strengths = np.ones(nsources, dtype=np.float64)
        targets = np.mgrid[dist:dist + 1:res*1j, 0:1:res*1j] \
                .reshape(dim, -1).astype(np.float64)

        source_boxes = np.array([0], dtype=np.int32)
        box_source_starts = np.array([0], dtype=np.int32)
        box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

        # {{{ apply P2M

        evt, (mpoles,) = p2m(queue,
                source_boxes=source_boxes,
                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=strengths,
                #iflags="print_hl_wrapper",
                out_host=True)

        # }}}

        # {{{ apply M2P

        ntargets = targets.shape[-1]

        box_target_starts = np.array([0], dtype=np.int32)
        box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

        evt, (pot, grad_x) = m2p(
                queue,
                expansions=mpoles,
                target_boxes=source_boxes,
                box_target_starts=box_target_starts,
                box_target_counts_nonchild=box_target_counts_nonchild,
                centers=centers,
                targets=targets,
                #iflags="print_hl",
                out_host=True,
                )

        # }}}

        # {{{ compute (direct) reference solution

        evt, (pot_direct, grad_x_direct) = p2p(
                queue,
                targets, sources, (strengths,),
                out_host=True)

        err = la.norm((pot - pot_direct)/res**2)

        # }}}

        eoc_rec.add_data_point(1/dist, err)

    print eoc_rec
    assert eoc_rec.order_estimate() > order + 0.5


@pytools.test.mark_test.opencl
def no_test_p2m2m2p(ctx_getter):
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
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
