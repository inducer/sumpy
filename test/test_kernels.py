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
import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion
from sumpy.kernel import LaplaceKernel, HelmholtzKernel, AxisTargetDerivative

import logging
logger = logging.getLogger(__name__)


@pytest.mark.opencl
def test_p2p(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dimensions = 3
    n = 5000

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
@pytest.mark.parametrize("knl", [
    LaplaceKernel(2),
    #HelmholtzKernel(2)
    ])
def test_p2e2p(ctx_getter, knl, expn_class, order):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 100
    nsources = 500

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 2

    out_kernels = [
            knl,
            AxisTargetDerivative(0, knl)
            ]
    expn = expn_class(knl, order=order)

    from sumpy import P2E, E2P, P2P
    p2e = P2E(ctx, expn, out_kernels)
    e2p = E2P(ctx, expn, out_kernels)
    p2p = P2P(ctx, out_kernels, exclude_self=False)

    from pytools.convergence import EOCRecorder
    eoc_rec_pot = EOCRecorder()
    eoc_rec_grad_x = EOCRecorder()

    for dist in [3, 5, 7]:
        from sumpy.expansion.local import LocalExpansionBase
        if issubclass(expn_class, LocalExpansionBase):
            centers = np.array([0.5+dist, 0.5], dtype=np.float64).reshape(knl.dim, 1)
        else:
            centers = np.array([0.5, 0.5], dtype=np.float64).reshape(knl.dim, 1)

        sources = np.random.rand(knl.dim, nsources).astype(np.float64)
        strengths = np.ones(nsources, dtype=np.float64)
        targets = np.mgrid[dist:dist + 1:res*1j, 0:1:res*1j] \
                .reshape(knl.dim, -1).astype(np.float64)

        source_boxes = np.array([0], dtype=np.int32)
        box_source_starts = np.array([0], dtype=np.int32)
        box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

        # {{{ apply p2e

        evt, (mpoles,) = p2e(queue,
                source_boxes=source_boxes,
                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=strengths,
                out_host=True, **extra_kwargs)

        # }}}

        # {{{ apply e2p

        ntargets = targets.shape[-1]

        box_target_starts = np.array([0], dtype=np.int32)
        box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

        evt, (pot, grad_x) = e2p(
                queue,
                expansions=mpoles,
                target_boxes=source_boxes,
                box_target_starts=box_target_starts,
                box_target_counts_nonchild=box_target_counts_nonchild,
                centers=centers,
                targets=targets,
                #flags="print_hl",
                out_host=True, **extra_kwargs)

        # }}}

        # {{{ compute (direct) reference solution

        evt, (pot_direct, grad_x_direct) = p2p(
                queue,
                targets, sources, (strengths,),
                out_host=True,
                **extra_kwargs)

        err_pot = la.norm((pot - pot_direct)/res**2)
        err_grad_x = la.norm((grad_x - grad_x_direct)/res**2)

        # }}}

        eoc_rec_pot.add_data_point(1/dist, err_pot)
        eoc_rec_grad_x.add_data_point(1/dist, err_grad_x)

    print expn_class, order
    print("POTENTIAL:")
    print(eoc_rec_pot)
    print("X TARGET DERIVATIVE:")
    print(eoc_rec_grad_x)

    tgt_order = order + 1

    assert eoc_rec_pot.order_estimate() > tgt_order - 0.5
    assert eoc_rec_grad_x.order_estimate() > tgt_order - 1.5


@pytools.test.mark_test.opencl
@pytest.mark.parametrize("order", [2, 3, 4, 5])
def no_test_translations(ctx_getter, order):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 100
    nsources = 500

    dim = 2

    from sumpy.kernel import LaplaceKernel
    knl = LaplaceKernel(dim)
    out_kernels = [knl]
    #m_expn = VolumeTaylorMultipoleExpansion(knl, order=order)
    l_expn = VolumeTaylorLocalExpansion(knl, order=order)

    from sumpy import P2E, E2P, P2P, E2E
    p2l = P2E(ctx, l_expn, out_kernels)
    l2l = E2E(ctx, l_expn, l_expn)
    l2p = E2P(ctx, l_expn, out_kernels)
    p2p = P2P(ctx, out_kernels, exclude_self=False)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for dist in [3, 5, 7]:
        l_centers = np.array(
                [
                    [0.4+dist, 0.6],
                    [0.5+dist, 0.5],
                    ],
                dtype=np.float64).T.copy()

        sources = np.random.rand(dim, nsources).astype(np.float64)
        strengths = np.ones(nsources, dtype=np.float64)
        targets = np.mgrid[dist:dist + 1:res*1j, 0:1:res*1j] \
                .reshape(dim, -1).astype(np.float64)

        source_boxes = np.array([0], dtype=np.int32)
        box_source_starts = np.array([0], dtype=np.int32)
        box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

        # {{{ apply P2L

        evt, (mpoles,) = p2l(queue,
                source_boxes=source_boxes,
                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                centers=l_centers,
                sources=sources,
                strengths=strengths,
                #iflags="print_hl_wrapper",
                out_host=True)

        # }}}

        # {{{ apply L2L

        src_box_starts = np.array([0, 0, 1], dtype=np.int32)
        src_box_lists = np.array([0], dtype=np.int32)

        box_source_starts = np.array([0], dtype=np.int32)
        evt, (mpoles,) = l2l(queue,
                src_expansions=mpoles,
                src_box_starts=src_box_starts,
                src_box_lists=src_box_lists,
                l_centers=l_centers,
                #iflags="print_hl_wrapper",
                out_host=True)

        # }}}

        # {{{ apply L2P

        ntargets = targets.shape[-1]

        box_target_starts = np.array([0], dtype=np.int32)
        box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

        evt, (pot,) = l2p(
                queue,
                expansions=mpoles,
                target_boxes=source_boxes,
                box_target_starts=box_target_starts,
                box_target_counts_nonchild=box_target_counts_nonchild,
                centers=l_centers,
                targets=targets,
                #iflags="print_hl",
                out_host=True,
                )

        # }}}

        # {{{ compute (direct) reference solution

        evt, (pot_direct,) = p2p(
                queue,
                targets, sources, (strengths,),
                out_host=True)

        err = la.norm((pot - pot_direct)/res**2)

        # }}}

        eoc_rec.add_data_point(1/dist, err)

    print eoc_rec
    tgt_order = order + 1
    assert eoc_rec.order_estimate() > tgt_order - 0.5


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
