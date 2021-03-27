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

import pytest
import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion, H2DMultipoleExpansion,
        VolumeTaylorMultipoleExpansionBase,
        LaplaceConformingVolumeTaylorMultipoleExpansion,
        HelmholtzConformingVolumeTaylorMultipoleExpansion,
        BiharmonicConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion, H2DLocalExpansion,
        LaplaceConformingVolumeTaylorLocalExpansion,
        HelmholtzConformingVolumeTaylorLocalExpansion,
        BiharmonicConformingVolumeTaylorLocalExpansion)
from sumpy.kernel import (LaplaceKernel, HelmholtzKernel, AxisTargetDerivative,
        DirectionalSourceDerivative, BiharmonicKernel, StokesletKernel)
import sumpy.symbolic as sym
from pytools.convergence import PConvergenceVerifier

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


@pytest.mark.parametrize("exclude_self", (True, False))
def test_p2p(ctx_factory, exclude_self):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dimensions = 3
    n = 5000

    from sumpy.p2p import P2P
    lknl = LaplaceKernel(dimensions)
    knl = P2P(ctx,
            [lknl, AxisTargetDerivative(0, lknl)],
            exclude_self=exclude_self)

    targets = np.random.rand(dimensions, n)
    sources = targets if exclude_self else np.random.rand(dimensions, n)

    strengths = np.ones(n, dtype=np.float64)

    extra_kwargs = {}

    if exclude_self:
        extra_kwargs["target_to_source"] = np.arange(n, dtype=np.int32)

    evt, (potential, x_derivative) = knl(
            queue, targets, sources, [strengths],
            out_host=True, **extra_kwargs)

    potential_ref = np.empty_like(potential)

    targets = targets.T
    sources = sources.T
    for itarg in range(n):

        with np.errstate(divide="ignore"):
            invdists = np.sum((targets[itarg] - sources) ** 2, axis=-1) ** -0.5

        if exclude_self:
            assert np.isinf(invdists[itarg])
            invdists[itarg] = 0

        potential_ref[itarg] = np.sum(strengths * invdists)

    potential_ref *= 1/(4*np.pi)

    rel_err = la.norm(potential - potential_ref)/la.norm(potential_ref)
    print(rel_err)
    assert rel_err < 1e-3


@pytest.mark.parametrize(("base_knl", "expn_class"), [
    (LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion),
    (LaplaceKernel(2), LaplaceConformingVolumeTaylorMultipoleExpansion),
])
def test_p2e_multiple(ctx_factory, base_knl, expn_class):

    from sympy.core.cache import clear_cache
    clear_cache()

    order = 4
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    nsources = 100

    extra_kwargs = {}
    if isinstance(base_knl, HelmholtzKernel):
        if base_knl.allow_evanescent:
            extra_kwargs["k"] = 0.2 * (0.707 + 0.707j)
        else:
            extra_kwargs["k"] = 0.2
    if isinstance(base_knl, StokesletKernel):
        extra_kwargs["mu"] = 0.2

    source_kernels = [
        DirectionalSourceDerivative(base_knl, "dir_vec"),
        base_knl,
    ]

    knl = base_knl
    expn = expn_class(knl, order=order)

    from sumpy import P2EFromSingleBox

    center = np.array([2, 1, 0][:knl.dim], np.float64)
    sources = (0.7*(-0.5+np.random.rand(knl.dim, nsources).astype(np.float64))
            + center[:, np.newaxis])

    strengths = [
        np.ones(nsources, dtype=np.float64) * (1/nsources),
        np.ones(nsources, dtype=np.float64) * (2/nsources)
    ]

    source_boxes = np.array([0], dtype=np.int32)
    box_source_starts = np.array([0], dtype=np.int32)
    box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

    alpha = np.linspace(0, 2*np.pi, nsources, np.float64)
    dir_vec = np.vstack([np.cos(alpha), np.sin(alpha)])

    from sumpy.expansion.local import LocalExpansionBase
    if issubclass(expn_class, LocalExpansionBase):
        loc_center = np.array([5.5, 0.0, 0.0][:knl.dim]) + center
        centers = np.array(loc_center, dtype=np.float64).reshape(knl.dim, 1)
    else:
        centers = (np.array([0.0, 0.0, 0.0][:knl.dim],
                            dtype=np.float64).reshape(knl.dim, 1)
                    + center[:, np.newaxis])

    rscale = 0.5  # pick something non-1

    # apply p2e at the same time
    p2e = P2EFromSingleBox(ctx, expn, kernels=source_kernels, strength_usage=[0, 1])
    evt, (mpoles,) = p2e(queue,
            source_boxes=source_boxes,
            box_source_starts=box_source_starts,
            box_source_counts_nonchild=box_source_counts_nonchild,
            centers=centers,
            sources=sources,
            strengths=strengths,
            nboxes=1,
            tgt_base_ibox=0,
            rscale=rscale,

            #flags="print_hl_cl",
            out_host=True,
            dir_vec=dir_vec,
            **extra_kwargs)

    actual_result = mpoles

    # apply p2e separately
    expected_result = np.zeros_like(actual_result)
    for i, source_kernel in enumerate(source_kernels):
        extra_source_kwargs = extra_kwargs.copy()
        if isinstance(source_kernel, DirectionalSourceDerivative):
            extra_source_kwargs["dir_vec"] = dir_vec
        p2e = P2EFromSingleBox(ctx, expn,
            kernels=[source_kernel], strength_usage=[i])
        evt, (mpoles,) = p2e(queue,
            source_boxes=source_boxes,
            box_source_starts=box_source_starts,
            box_source_counts_nonchild=box_source_counts_nonchild,
            centers=centers,
            sources=sources,
            strengths=strengths,
            nboxes=1,
            tgt_base_ibox=0,
            rscale=rscale,

            #flags="print_hl_cl",
            out_host=True, **extra_source_kwargs)
        expected_result += mpoles

    norm = la.norm(actual_result - expected_result)/la.norm(expected_result)
    assert norm < 1e-12


@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize(("base_knl", "expn_class"), [
    (LaplaceKernel(2), VolumeTaylorLocalExpansion),
    (LaplaceKernel(2), VolumeTaylorMultipoleExpansion),
    (LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion),
    (LaplaceKernel(2), LaplaceConformingVolumeTaylorMultipoleExpansion),

    (HelmholtzKernel(2), VolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2), VolumeTaylorLocalExpansion),
    (HelmholtzKernel(2), HelmholtzConformingVolumeTaylorLocalExpansion),
    (HelmholtzKernel(2), HelmholtzConformingVolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2), H2DLocalExpansion),
    (HelmholtzKernel(2), H2DMultipoleExpansion),

    (DirectionalSourceDerivative(BiharmonicKernel(2), "dir_vec"),
        VolumeTaylorMultipoleExpansion),
    (DirectionalSourceDerivative(BiharmonicKernel(2), "dir_vec"),
        VolumeTaylorLocalExpansion),

    (HelmholtzKernel(2, allow_evanescent=True), VolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2, allow_evanescent=True), VolumeTaylorLocalExpansion),
    (HelmholtzKernel(2, allow_evanescent=True),
     HelmholtzConformingVolumeTaylorLocalExpansion),
    (HelmholtzKernel(2, allow_evanescent=True),
     HelmholtzConformingVolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2, allow_evanescent=True), H2DLocalExpansion),
    (HelmholtzKernel(2, allow_evanescent=True), H2DMultipoleExpansion),
    ])
@pytest.mark.parametrize("with_source_derivative", [
    False,
    True
    ])
# Sample: test_p2e2p(cl._csc, LaplaceKernel(2), VolumeTaylorLocalExpansion, 4, False)
def test_p2e2p(ctx_factory, base_knl, expn_class, order, with_source_derivative):
    #logging.basicConfig(level=logging.INFO)

    from sympy.core.cache import clear_cache
    clear_cache()

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 100
    nsources = 100

    extra_kwargs = {}
    if isinstance(base_knl, HelmholtzKernel):
        if base_knl.allow_evanescent:
            extra_kwargs["k"] = 0.2 * (0.707 + 0.707j)
        else:
            extra_kwargs["k"] = 0.2
    if isinstance(base_knl, StokesletKernel):
        extra_kwargs["mu"] = 0.2

    if with_source_derivative:
        knl = DirectionalSourceDerivative(base_knl, "dir_vec")
    else:
        knl = base_knl

    target_kernels = [
            knl,
            AxisTargetDerivative(0, knl),
            ]
    expn = expn_class(knl, order=order)

    from sumpy import P2EFromSingleBox, E2PFromSingleBox, P2P
    p2e = P2EFromSingleBox(ctx, expn, kernels=[knl])
    e2p = E2PFromSingleBox(ctx, expn, kernels=target_kernels)
    p2p = P2P(ctx, target_kernels, exclude_self=False)

    from pytools.convergence import EOCRecorder
    eoc_rec_pot = EOCRecorder()
    eoc_rec_grad_x = EOCRecorder()

    from sumpy.expansion.local import LocalExpansionBase
    if issubclass(expn_class, LocalExpansionBase):
        h_values = [1/5, 1/7, 1/20]
    else:
        h_values = [1/2, 1/3, 1/5]

    center = np.array([2, 1, 0][:knl.dim], np.float64)
    sources = (0.7*(-0.5+np.random.rand(knl.dim, nsources).astype(np.float64))
            + center[:, np.newaxis])

    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    source_boxes = np.array([0], dtype=np.int32)
    box_source_starts = np.array([0], dtype=np.int32)
    box_source_counts_nonchild = np.array([nsources], dtype=np.int32)

    extra_source_kwargs = extra_kwargs.copy()
    if isinstance(knl, DirectionalSourceDerivative):
        alpha = np.linspace(0, 2*np.pi, nsources, np.float64)
        dir_vec = np.vstack([np.cos(alpha), np.sin(alpha)])
        extra_source_kwargs["dir_vec"] = dir_vec

    from sumpy.visualization import FieldPlotter

    for h in h_values:
        if issubclass(expn_class, LocalExpansionBase):
            loc_center = np.array([5.5, 0.0, 0.0][:knl.dim]) + center
            centers = np.array(loc_center, dtype=np.float64).reshape(knl.dim, 1)
            fp = FieldPlotter(loc_center, extent=h, npoints=res)
        else:
            eval_center = np.array([1/h, 0.0, 0.0][:knl.dim]) + center
            fp = FieldPlotter(eval_center, extent=0.1, npoints=res)
            centers = (np.array([0.0, 0.0, 0.0][:knl.dim],
                                dtype=np.float64).reshape(knl.dim, 1)
                        + center[:, np.newaxis])

        targets = fp.points

        rscale = 0.5  # pick something non-1

        # {{{ apply p2e

        evt, (mpoles,) = p2e(queue,
                source_boxes=source_boxes,
                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=(strengths,),
                nboxes=1,
                tgt_base_ibox=0,
                rscale=rscale,

                #flags="print_hl_cl",
                out_host=True, **extra_source_kwargs)

        # }}}

        # {{{ apply e2p

        ntargets = targets.shape[-1]

        box_target_starts = np.array([0], dtype=np.int32)
        box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

        evt, (pot, grad_x, ) = e2p(
                queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                target_boxes=source_boxes,
                box_target_starts=box_target_starts,
                box_target_counts_nonchild=box_target_counts_nonchild,
                centers=centers,
                targets=targets,
                rscale=rscale,

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

    print(expn_class, knl, order)
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

    if isinstance(base_knl, DirectionalSourceDerivative):
        slack += 1
        grad_slack += 2

    if isinstance(base_knl, HelmholtzKernel):
        if base_knl.allow_evanescent:
            slack += 0.5
            grad_slack += 0.5

        if issubclass(expn_class, VolumeTaylorMultipoleExpansionBase):
            slack += 0.3
            grad_slack += 0.3

    assert eoc_rec_pot.order_estimate() > tgt_order - slack
    assert eoc_rec_grad_x.order_estimate() > tgt_order_grad - grad_slack


@pytest.mark.parametrize("knl, local_expn_class, mpole_expn_class", [
    (LaplaceKernel(2), VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion),
    (LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion,
     LaplaceConformingVolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2), VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2), HelmholtzConformingVolumeTaylorLocalExpansion,
     HelmholtzConformingVolumeTaylorMultipoleExpansion),
    (HelmholtzKernel(2), H2DLocalExpansion, H2DMultipoleExpansion),
    (StokesletKernel(2, 0, 0), VolumeTaylorLocalExpansion,
     VolumeTaylorMultipoleExpansion),
    (StokesletKernel(2, 0, 0), BiharmonicConformingVolumeTaylorLocalExpansion,
     BiharmonicConformingVolumeTaylorMultipoleExpansion),
    ])
def test_translations(ctx_factory, knl, local_expn_class, mpole_expn_class):
    logging.basicConfig(level=logging.INFO)

    from sympy.core.cache import clear_cache
    clear_cache()

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 20
    nsources = 15

    target_kernels = [knl]

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
    if isinstance(knl, StokesletKernel):
        extra_kwargs["mu"] = 0.05

    # Just to make sure things also work away from the origin
    origin = np.array([2, 1, 0][:knl.dim], np.float64)
    sources = (0.7*(-0.5+np.random.rand(knl.dim, nsources).astype(np.float64))
            + origin[:, np.newaxis])
    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    pconv_verifier_p2m2p = PConvergenceVerifier()
    pconv_verifier_p2m2m2p = PConvergenceVerifier()
    pconv_verifier_p2m2m2l2p = PConvergenceVerifier()
    pconv_verifier_full = PConvergenceVerifier()

    from sumpy.visualization import FieldPlotter

    eval_offset = np.array([5.5, 0.0, 0][:knl.dim])

    centers = (np.array(
            [
                # box 0: particles, first mpole here
                [0, 0, 0][:knl.dim],

                # box 1: second mpole here
                np.array([-0.2, 0.1, 0][:knl.dim], np.float64),

                # box 2: first local here
                eval_offset + np.array([0.3, -0.2, 0][:knl.dim], np.float64),

                # box 3: second local and eval here
                eval_offset
                ],
            dtype=np.float64) + origin).T.copy()

    del eval_offset

    from sumpy.expansion import VolumeTaylorExpansionBase

    if isinstance(knl, HelmholtzKernel) and \
           issubclass(local_expn_class, VolumeTaylorExpansionBase):
        # FIXME: Embarrassing--but we run out of memory for higher orders.
        orders = [2, 3]
    else:
        orders = [2, 3, 4]
    nboxes = centers.shape[-1]

    def eval_at(e2p, source_box_nr, rscale):
        e2p_target_boxes = np.array([source_box_nr], dtype=np.int32)

        # These are indexed by global box numbers.
        e2p_box_target_starts = np.array([0, 0, 0, 0], dtype=np.int32)
        e2p_box_target_counts_nonchild = np.array([0, 0, 0, 0],
                dtype=np.int32)
        e2p_box_target_counts_nonchild[source_box_nr] = ntargets

        evt, (pot,) = e2p(
                queue,

                src_expansions=mpoles,
                src_base_ibox=0,

                target_boxes=e2p_target_boxes,
                box_target_starts=e2p_box_target_starts,
                box_target_counts_nonchild=e2p_box_target_counts_nonchild,
                centers=centers,
                targets=targets,

                rscale=rscale,

                out_host=True, **extra_kwargs
                )

        return pot

    for order in orders:
        m_expn = mpole_expn_class(knl, order=order)
        l_expn = local_expn_class(knl, order=order)

        from sumpy import P2EFromSingleBox, E2PFromSingleBox, P2P, E2EFromCSR
        p2m = P2EFromSingleBox(ctx, m_expn)
        m2m = E2EFromCSR(ctx, m_expn, m_expn)
        m2p = E2PFromSingleBox(ctx, m_expn, target_kernels)
        m2l = E2EFromCSR(ctx, m_expn, l_expn)
        l2l = E2EFromCSR(ctx, l_expn, l_expn)
        l2p = E2PFromSingleBox(ctx, l_expn, target_kernels)
        p2p = P2P(ctx, target_kernels, exclude_self=False)

        fp = FieldPlotter(centers[:, -1], extent=0.3, npoints=res)
        targets = fp.points

        # {{{ compute (direct) reference solution

        evt, (pot_direct,) = p2p(
                queue,
                targets, sources, (strengths,),
                out_host=True, **extra_kwargs)

        # }}}

        m1_rscale = 0.5
        m2_rscale = 0.25
        l1_rscale = 0.5
        l2_rscale = 0.25

        # {{{ apply P2M

        p2m_source_boxes = np.array([0], dtype=np.int32)

        # These are indexed by global box numbers.
        p2m_box_source_starts = np.array([0, 0, 0, 0], dtype=np.int32)
        p2m_box_source_counts_nonchild = np.array([nsources, 0, 0, 0],
                dtype=np.int32)

        evt, (mpoles,) = p2m(queue,
                source_boxes=p2m_source_boxes,
                box_source_starts=p2m_box_source_starts,
                box_source_counts_nonchild=p2m_box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=(strengths,),
                nboxes=nboxes,
                rscale=m1_rscale,

                tgt_base_ibox=0,

                #flags="print_hl_wrapper",
                out_host=True, **extra_kwargs)

        # }}}

        ntargets = targets.shape[-1]

        pot = eval_at(m2p, 0, m1_rscale)

        err = la.norm((pot - pot_direct)/res**2)
        err = err / (la.norm(pot_direct) / res**2)

        pconv_verifier_p2m2p.add_data_point(order, err)

        # {{{ apply M2M

        m2m_target_boxes = np.array([1], dtype=np.int32)
        m2m_src_box_starts = np.array([0, 1], dtype=np.int32)
        m2m_src_box_lists = np.array([0], dtype=np.int32)

        evt, (mpoles,) = m2m(queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=mpoles.shape[0],

                target_boxes=m2m_target_boxes,

                src_box_starts=m2m_src_box_starts,
                src_box_lists=m2m_src_box_lists,
                centers=centers,

                src_rscale=m1_rscale,
                tgt_rscale=m2_rscale,

                #flags="print_hl_cl",
                out_host=True, **extra_kwargs)

        # }}}

        pot = eval_at(m2p, 1, m2_rscale)

        err = la.norm((pot - pot_direct)/res**2)
        err = err / (la.norm(pot_direct) / res**2)

        pconv_verifier_p2m2m2p.add_data_point(order, err)

        # {{{ apply M2L

        m2l_target_boxes = np.array([2], dtype=np.int32)
        m2l_src_box_starts = np.array([0, 1], dtype=np.int32)
        m2l_src_box_lists = np.array([1], dtype=np.int32)

        evt, (mpoles,) = m2l(queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=mpoles.shape[0],

                target_boxes=m2l_target_boxes,
                src_box_starts=m2l_src_box_starts,
                src_box_lists=m2l_src_box_lists,
                centers=centers,

                src_rscale=m2_rscale,
                tgt_rscale=l1_rscale,

                #flags="print_hl_cl",
                out_host=True, **extra_kwargs)

        # }}}

        pot = eval_at(l2p, 2, l1_rscale)

        err = la.norm((pot - pot_direct)/res**2)
        err = err / (la.norm(pot_direct) / res**2)

        pconv_verifier_p2m2m2l2p.add_data_point(order, err)

        # {{{ apply L2L

        l2l_target_boxes = np.array([3], dtype=np.int32)
        l2l_src_box_starts = np.array([0, 1], dtype=np.int32)
        l2l_src_box_lists = np.array([2], dtype=np.int32)

        evt, (mpoles,) = l2l(queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=mpoles.shape[0],

                target_boxes=l2l_target_boxes,
                src_box_starts=l2l_src_box_starts,
                src_box_lists=l2l_src_box_lists,
                centers=centers,

                src_rscale=l1_rscale,
                tgt_rscale=l2_rscale,

                #flags="print_hl_wrapper",
                out_host=True, **extra_kwargs)

        # }}}

        pot = eval_at(l2p, 3, l2_rscale)

        err = la.norm((pot - pot_direct)/res**2)
        err = err / (la.norm(pot_direct) / res**2)

        pconv_verifier_full.add_data_point(order, err)

    for name, verifier in [
            ("p2m2p", pconv_verifier_p2m2p),
            ("p2m2m2p", pconv_verifier_p2m2m2p),
            ("p2m2m2l2p", pconv_verifier_p2m2m2l2p),
            ("full", pconv_verifier_full),
            ]:
        print(30*"-")
        print(name)
        print(30*"-")
        print(verifier)
        print(30*"-")
        verifier()


@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize(("base_knl", "local_expn_class", "mpole_expn_class"), [
    (LaplaceKernel(2), VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion),
    ])
@pytest.mark.parametrize("with_source_derivative", [
    False,
    True
    ])
def test_m2m_and_l2l_exprs_simpler(base_knl, local_expn_class, mpole_expn_class,
        order, with_source_derivative):

    from sympy.core.cache import clear_cache
    clear_cache()

    np.random.seed(17)

    extra_kwargs = {}
    if isinstance(base_knl, HelmholtzKernel):
        if base_knl.allow_evanescent:
            extra_kwargs["k"] = 0.2 * (0.707 + 0.707j)
        else:
            extra_kwargs["k"] = 0.2
    if isinstance(base_knl, StokesletKernel):
        extra_kwargs["mu"] = 0.2

    if with_source_derivative:
        knl = DirectionalSourceDerivative(base_knl, "dir_vec")
    else:
        knl = base_knl

    mpole_expn = mpole_expn_class(knl, order=order)
    local_expn = local_expn_class(knl, order=order)

    from sumpy.symbolic import make_sym_vector, Symbol, USE_SYMENGINE
    dvec = make_sym_vector("d", knl.dim)
    src_coeff_exprs = [Symbol("src_coeff%d" % i) for i in range(len(mpole_expn))]

    src_rscale = 3
    tgt_rscale = 2

    faster_m2m = mpole_expn.translate_from(mpole_expn, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale)
    slower_m2m = mpole_expn.translate_from(mpole_expn, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, _fast_version=False)

    def _check_equal(expr1, expr2):
        if USE_SYMENGINE:
            return float((expr1 - expr2).expand()) == 0.0
        else:
            # with sympy we are using UnevaluatedExpr and expand doesn't expand it
            # Running doit replaces UnevaluatedExpr with evaluated exprs
            return float((expr1 - expr2).doit().expand()) == 0.0

    for expr1, expr2 in zip(faster_m2m, slower_m2m):
        assert _check_equal(expr1, expr2)

    faster_l2l = local_expn.translate_from(local_expn, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale)
    slower_l2l = local_expn.translate_from(local_expn, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, _fast_version=False)
    for expr1, expr2 in zip(faster_l2l, slower_l2l):
        assert _check_equal(expr1, expr2)


# {{{ test toeplitz

def _m2l_translate_simple(tgt_expansion, src_expansion, src_coeff_exprs, src_rscale,
                           dvec, tgt_rscale):

    if not tgt_expansion.use_rscale:
        src_rscale = 1
        tgt_rscale = 1

    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase
    if not isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
        return 1

    # We know the general form of the multipole expansion is:
    #
    #    coeff0 * diff(kernel, mi0) + coeff1 * diff(kernel, mi1) + ...
    #
    # To get the local expansion coefficients, we take derivatives of
    # the multipole expansion.
    taker = src_expansion.kernel.get_derivative_taker(dvec, src_rscale, sac=None)

    from sumpy.tools import add_mi

    result = []
    for deriv in tgt_expansion.get_coefficient_identifiers():
        local_result = []
        for coeff, term in zip(
                src_coeff_exprs,
                src_expansion.get_coefficient_identifiers()):

            kernel_deriv = taker.diff(add_mi(deriv, term)) / src_rscale**sum(deriv)

            local_result.append(
                    coeff * kernel_deriv * tgt_rscale**sum(deriv))
        result.append(sym.Add(*local_result))
    return result


def test_m2l_toeplitz():
    dim = 3
    knl = LaplaceKernel(dim)
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion

    local_expn = local_expn_class(knl, order=5)
    mpole_expn = mpole_expn_class(knl, order=5)

    dvec = sym.make_sym_vector("d", dim)
    src_coeff_exprs = list(1 + np.random.randn(len(mpole_expn)))
    src_rscale = 2.0
    tgt_rscale = 1.0

    expected_output = _m2l_translate_simple(local_expn, mpole_expn, src_coeff_exprs,
                                           src_rscale, dvec, tgt_rscale)
    actual_output = local_expn.translate_from(mpole_expn, src_coeff_exprs,
                                              src_rscale, dvec, tgt_rscale, sac=None)

    replace_dict = dict((d, np.random.rand(1)[0]) for d in dvec)
    for sym_a, sym_b in zip(expected_output, actual_output):
        num_a = sym_a.xreplace(replace_dict)
        num_b = sym_b.xreplace(replace_dict)
        assert abs(num_a - num_b)/abs(num_a) < 1e-10

# }}}


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
