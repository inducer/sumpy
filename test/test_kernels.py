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
from sumpy.expansion.local import VolumeTaylorLocalExpansion, H2DLocalExpansion
from sumpy.kernel import (LaplaceKernel, HelmholtzKernel, AxisTargetDerivative,
        DirectionalSourceDerivative)

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


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
@pytest.mark.parametrize(("knl", "expn_class"), [
    (LaplaceKernel(2), VolumeTaylorLocalExpansion),
    (LaplaceKernel(2), VolumeTaylorMultipoleExpansion),

    (HelmholtzKernel(2), VolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2), VolumeTaylorLocalExpansion),
    (HelmholtzKernel(2), H2DLocalExpansion),
    ])
@pytest.mark.parametrize("with_source_derivative", [
    False,
    True
    ])
def test_p2e2p(ctx_getter, knl, expn_class, order, with_source_derivative):
    #logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 100
    nsources = 100

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05

    if with_source_derivative:
        knl = DirectionalSourceDerivative(knl, "dir_vec")

    out_kernels = [
            knl,
            AxisTargetDerivative(0, knl),
            ]
    expn = expn_class(knl, order=order)

    from sumpy import P2E, E2P, P2P
    p2e = P2E(ctx, expn, out_kernels)
    e2p = E2P(ctx, expn, out_kernels)
    p2p = P2P(ctx, out_kernels, exclude_self=False)

    from pytools.convergence import EOCRecorder
    eoc_rec_pot = EOCRecorder()
    eoc_rec_grad_x = EOCRecorder()

    from sumpy.expansion.local import LocalExpansionBase
    if issubclass(expn_class, LocalExpansionBase):
        h_values = [1/5, 1/7, 1/20]
    else:
        h_values = [1/2, 1/3, 1/5]

    center = np.array([2, 1], np.float64)
    sources = (0.7*(-0.5+np.random.rand(knl.dim, nsources).astype(np.float64))
            + center[:, np.newaxis])

    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    source_boxes = np.array([0], dtype=np.int32)
    box_source_starts = np.array([0], dtype=np.int32)
    box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

    extra_source_kwargs = extra_kwargs.copy()
    if with_source_derivative:
        alpha = np.linspace(0, 2*np.pi, nsources, np.float64)
        dir_vec = np.vstack([np.cos(alpha), np.sin(alpha)])
        extra_source_kwargs["dir_vec"] = dir_vec

    from sumpy.visualization import FieldPlotter

    for h in h_values:
        if issubclass(expn_class, LocalExpansionBase):
            loc_center = np.array([5.5, 0.0]) + center
            centers = np.array(loc_center, dtype=np.float64).reshape(knl.dim, 1)
            fp = FieldPlotter(loc_center, extent=h, npoints=res)
        else:
            eval_center = np.array([1/h, 0.0]) + center
            fp = FieldPlotter(eval_center, extent=0.1, npoints=res)
            centers = (
                    np.array([0.0, 0.0], dtype=np.float64).reshape(knl.dim, 1)
                    + center[:, np.newaxis])

        targets = fp.points

        # {{{ apply p2e

        evt, (mpoles,) = p2e(queue,
                source_boxes=source_boxes,
                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=strengths,
                #flags="print_hl_cl",
                out_host=True, **extra_source_kwargs)

        # }}}

        # {{{ apply e2p

        ntargets = targets.shape[-1]

        box_target_starts = np.array([0], dtype=np.int32)
        box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

        evt, (pot, grad_x, ) = e2p(
                queue,
                expansions=mpoles,
                target_boxes=source_boxes,
                box_target_starts=box_target_starts,
                box_target_counts_nonchild=box_target_counts_nonchild,
                centers=centers,
                targets=targets,
                #flags="print_hl_cl",
                out_host=True, **extra_kwargs)

        # }}}

        # {{{ compute (direct) reference solution

        evt, (pot_direct, grad_x_direct, ) = p2p(
                queue,
                targets, sources, (strengths,),
                out_host=True,
                **extra_source_kwargs)

        err_pot = la.norm((pot - pot_direct)/res**2)
        err_grad_x = la.norm((grad_x - grad_x_direct)/res**2)

        if 1:
            err_pot = err_pot / la.norm((pot_direct)/res**2)
            err_grad_x = err_grad_x / la.norm((grad_x_direct)/res**2)

        if 0:
            import matplotlib.pyplot as pt
            from matplotlib.colors import Normalize

            pt.subplot(131)
            im = fp.show_scalar_in_matplotlib(pot.real)
            im.set_norm(Normalize(vmin=-0.1, vmax=0.1))

            pt.subplot(132)
            im = fp.show_scalar_in_matplotlib(pot_direct.real)
            im.set_norm(Normalize(vmin=-0.1, vmax=0.1))
            pt.colorbar()

            pt.subplot(133)
            im = fp.show_scalar_in_matplotlib(np.log10(1e-15+np.abs(pot-pot_direct)))
            im.set_norm(Normalize(vmin=-6, vmax=1))

            pt.colorbar()
            pt.show()

        # }}}

        eoc_rec_pot.add_data_point(h, err_pot)
        eoc_rec_grad_x.add_data_point(h, err_grad_x)

    print expn_class, knl, order
    print("POTENTIAL:")
    print(eoc_rec_pot)
    print("X TARGET DERIVATIVE:")
    print(eoc_rec_grad_x)

    tgt_order = order + 1
    if issubclass(expn_class, LocalExpansionBase):
        tgt_order_grad = tgt_order - 1
        slack = 0.7
        grad_slack = 0.5
    else:
        tgt_order_grad = tgt_order + 1

        slack = 0.5
        grad_slack = 1

        if order <= 2:
            slack += 1
            grad_slack += 1

    if isinstance(knl, DirectionalSourceDerivative):
        slack += 1
        grad_slack += 2

    assert eoc_rec_pot.order_estimate() > tgt_order - slack
    assert eoc_rec_grad_x.order_estimate() > tgt_order_grad - grad_slack


class PConvergenceVerifier(object):
    def __init__(self):
        self.orders = []
        self.errors = []

    def add_data_point(self, order, error):
        self.orders.append(order)
        self.errors.append(error)

    def __str__(self):
        from pytools import Table
        tbl = Table()
        tbl.add_row(("p", "error"))

        for p, err in zip(self.orders, self.errors):
            tbl.add_row((str(p), str(err)))

        return str(tbl)

    def __call__(self):
        orders = np.array(self.orders, np.float64)
        log_errors = np.log10(1e-20+np.abs(np.array(self.errors, np.float64)))

        constant_ish = log_errors/orders

        c_max = np.max(constant_ish)
        c_min = np.min(constant_ish)

        assert c_max < c_min + 2, constant_ish


@pytools.test.mark_test.opencl
@pytest.mark.parametrize("knl", [
    LaplaceKernel(2),
    HelmholtzKernel(2)
    ])
def test_translations(ctx_getter, knl):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 200
    nsources = 15

    out_kernels = [knl]

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05

    # Just to make sure things also work away from the origin
    origin = np.array([2, 1], np.float64)
    sources = (0.7*(-0.5+np.random.rand(knl.dim, nsources).astype(np.float64))
            + origin[:, np.newaxis])
    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    pconv_verifier = PConvergenceVerifier()

    from sumpy.visualization import FieldPlotter

    eval_offset = np.array([5.5, 0.0])

    centers = (np.array(
            [
                # box 0: particles, first mpole here
                [0, 0],

                # box 1: mpole here
                np.array([-0.2, 0.1], np.float64),

                # box 2: first local here
                eval_offset + np.array([0.3, -0.2], np.float64),

                # box 3: second local and eval here
                eval_offset
                ],
            dtype=np.float64) + origin).T.copy()

    del eval_offset

    for order in [2, 3]:
        m_expn = VolumeTaylorMultipoleExpansion(knl, order=order)
        l_expn = VolumeTaylorLocalExpansion(knl, order=order)

        from sumpy import P2E, E2P, P2P, E2E
        p2m = P2E(ctx, m_expn, out_kernels)
        m2l = E2E(ctx, m_expn, l_expn)
        l2l = E2E(ctx, l_expn, l_expn)
        l2p = E2P(ctx, l_expn, out_kernels)
        p2p = P2P(ctx, out_kernels, exclude_self=False)

        fp = FieldPlotter(centers[:, -1], extent=0.3, npoints=res)
        targets = fp.points

        # {{{ apply P2M

        p2m_source_boxes = np.array([0], dtype=np.int32)
        p2m_box_source_starts = np.array([0], dtype=np.int32)
        p2m_box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

        evt, (mpoles,) = p2m(queue,
                source_boxes=p2m_source_boxes,
                box_source_starts=p2m_box_source_starts,
                box_source_counts_nonchild=p2m_box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=strengths,
                #flags="print_hl_wrapper",
                out_host=True, **extra_kwargs)

        # }}}

        # {{{ apply M2L

        m2l_target_boxes = np.array([2], dtype=np.int32)
        m2l_src_box_starts = np.array([0, 1], dtype=np.int32)
        m2l_src_box_lists = np.array([0], dtype=np.int32)

        evt, (mpoles,) = m2l(queue,
                src_expansions=mpoles,
                target_boxes=m2l_target_boxes,
                src_box_starts=m2l_src_box_starts,
                src_box_lists=m2l_src_box_lists,
                centers=centers,
                #flags="print_hl_cl",
                out_host=True, **extra_kwargs)

        # }}}

        # {{{ apply L2L

        l2l_target_boxes = np.array([3], dtype=np.int32)
        l2l_src_box_starts = np.array([0, 1], dtype=np.int32)
        l2l_src_box_lists = np.array([2], dtype=np.int32)

        evt, (mpoles,) = l2l(queue,
                src_expansions=mpoles,
                target_boxes=l2l_target_boxes,
                src_box_starts=l2l_src_box_starts,
                src_box_lists=l2l_src_box_lists,
                centers=centers,
                #flags="print_hl_wrapper",
                out_host=True, **extra_kwargs)

        # }}}

        # {{{ apply L2P

        ntargets = targets.shape[-1]

        l2p_target_boxes = np.array([3], dtype=np.int32)
        l2p_box_target_starts = np.array([0], dtype=np.int32)
        l2p_box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

        evt, (pot,) = l2p(
                queue,
                expansions=mpoles,
                target_boxes=l2p_target_boxes,
                box_target_starts=l2p_box_target_starts,
                box_target_counts_nonchild=l2p_box_target_counts_nonchild,
                centers=centers,
                targets=targets,
                #flags="trace_assignment_values",
                out_host=True, **extra_kwargs
                )

        # }}}

        # {{{ compute (direct) reference solution

        evt, (pot_direct,) = p2p(
                queue,
                targets, sources, (strengths,),
                out_host=True, **extra_kwargs)

        err = la.norm((pot - pot_direct)/res**2)

        # }}}

        pconv_verifier.add_data_point(order, err)

    print pconv_verifier
    pconv_verifier()


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
