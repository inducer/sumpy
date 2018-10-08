from __future__ import division, absolute_import, print_function

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

from six.moves import range

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
        HelmholtzConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion, H2DLocalExpansion,
        LaplaceConformingVolumeTaylorLocalExpansion,
        HelmholtzConformingVolumeTaylorLocalExpansion)
from sumpy.kernel import (LaplaceKernel, HelmholtzKernel, AxisTargetDerivative,
        DirectionalSourceDerivative)
from pytools.convergence import PConvergenceVerifier

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

import matplotlib.pyplot as plt

def test_m2m(ctx_getter, order=5):
    logging.basicConfig(level=logging.INFO)

    from sympy.core.cache import clear_cache
    clear_cache()

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    np.random.seed(17)

    res = 200
    nsources = 150

    knl = HelmholtzKernel(2)
    out_kernels = [knl]

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05

    # Just to make sure things also work away from the origin
    origin = np.array([2, 1], np.float64)
    sources = (0.7*(-0.5+np.random.rand(knl.dim, nsources).astype(np.float64))
            + origin[:, np.newaxis])
    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    pconv_verifier_p2m2p = PConvergenceVerifier()
    pconv_verifier_p2m2m2p = PConvergenceVerifier()
    pconv_verifier_p2m2m2l2p = PConvergenceVerifier()
    pconv_verifier_full = PConvergenceVerifier()

    from sumpy.visualization import FieldPlotter

    eval_offset = np.array([5.5, 0.0])

    centers = (np.array(
            [
                # box 0: particles, first mpole here
                [0, 0],

                # box 1: second mpole here
                np.array([-0.2, 0.1], np.float64),

                # box 2: first local here
                eval_offset + np.array([0.3, -0.2], np.float64),

                # box 3: second local and eval here
                eval_offset
                ],
            dtype=np.float64) + origin).T.copy()

    del eval_offset

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

    if isinstance(knl, LaplaceKernel):
        mpole_expn_classes = [LaplaceConformingVolumeTaylorMultipoleExpansion, VolumeTaylorMultipoleExpansion]
        local_expn_classes = [LaplaceConformingVolumeTaylorLocalExpansion, VolumeTaylorLocalExpansion]
    elif isinstance(knl, HelmholtzKernel):
        mpole_expn_classes = [HelmholtzConformingVolumeTaylorMultipoleExpansion, VolumeTaylorMultipoleExpansion]
        local_expn_classes = [HelmholtzConformingVolumeTaylorLocalExpansion, VolumeTaylorLocalExpansion]

    h_values = 1000*2.0**np.arange(-5, 3)
    for order in [6]:
        h_errs = []
        for h in h_values:
            m2m_vals = []
            for i, (mpole_expn_class, local_expn_class) in enumerate(zip(mpole_expn_classes, local_expn_classes)):
                m_expn = mpole_expn_class(knl, order=order)
                l_expn = local_expn_class(knl, order=order)

                from sumpy import P2EFromSingleBox, E2PFromSingleBox, P2P, E2EFromCSR
                p2m = P2EFromSingleBox(ctx, m_expn)
                m2m = E2EFromCSR(ctx, m_expn, m_expn)
                m2p = E2PFromSingleBox(ctx, m_expn, out_kernels)
                p2p = P2P(ctx, out_kernels, exclude_self=False)

                fp = FieldPlotter(centers[:, -1], extent=h, npoints=res)
                targets = fp.points

                m1_rscale = 0.5
                m2_rscale = 0.25

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
                        strengths=strengths,
                        nboxes=nboxes,
                        rscale=m1_rscale,

                        tgt_base_ibox=0,

                        #flags="print_hl_wrapper",
                        out_host=True, **extra_kwargs)

                # }}}

                ntargets = targets.shape[-1]

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
                
                evt, (pot_direct,) = p2p(
                        queue,
                        targets, sources, (strengths,),
                        out_host=True, **extra_kwargs)
                #if (i == 0):
                #    m2m_vals.append(pot_direct)
                #else:
                #    m2m_vals.append(pot)
                m2m_vals.append(pot)

            err = la.norm(m2m_vals[1] - m2m_vals[0])/la.norm(m2m_vals[0])
            print(err)
            h_errs.append(abs(err))
        a, b = np.polyfit(np.log(h_values), np.log(h_errs), 1)
        plt.loglog(h_values, h_errs, label="order={}".format(order), marker='x')
        plt.loglog(h_values, h_values**a / h_values[-3]**a * h_errs[-3], label="h**%.2f" % a)
        plt.loglog(h_values, h_values**(-order-1) / h_values[-3]**(-order-1) * h_errs[-3], label="h**-%.2f" % (order+1))
    plt.xlabel("h")
    plt.ylabel("Error between reduced and full")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_m2m(cl.create_some_context)

# vim: fdm=marker
