from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
import loopy as lp
import sympy as sp
from pytools import memoize_method


# {{{ E2P base class

class E2PBase(object):
    def __init__(self, ctx, expansion, kernels,
            options=[], name="e2p", device=None):
        """
        :arg expansion: a subclass of :class:`sympy.expansion.ExpansionBase`
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """

        if device is None:
            device = ctx.devices[0]

        self.ctx = ctx
        self.expansion = expansion
        self.kernels = kernels
        self.options = options
        self.name = name
        self.device = device

        self.dim = expansion.dim

        from sumpy.kernel import TargetDerivativeRemover
        tdr = TargetDerivativeRemover()
        for knl in kernels:
            assert tdr(knl) == expansion.kernel

    def get_loopy_insns_and_result_names(self):
        from sumpy.symbolic import make_sym_vector
        bvec = make_sym_vector("b", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        coeff_exprs = [sp.Symbol("coeff%d" % i)
                for i in xrange(len(self.expansion.get_coefficient_identifiers()))]
        value = self.expansion.evaluate(coeff_exprs, bvec)
        result_names = [
            sac.assign_unique("result_%d_p" % i,
                knl.postprocess_at_target(value, bvec))
            for i, knl in enumerate(self.kernels)
            ]

        sac.run_global_cse()

        from sumpy.symbolic import kill_trivial_assignments
        assignments = kill_trivial_assignments([
                (name, expr)
                for name, expr in sac.assignments.iteritems()],
                retain_names=result_names)

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(assignments,
                vector_names=set(["b"]),
                pymbolic_expr_maps=[self.expansion.transform_to_code],
                complex_dtype=np.complex128  # FIXME
                )

        from pymbolic.sympy_interface import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()
        loopy_insns.append(
                lp.ExpressionInstruction(id=None,
                    assignee="kernel_scaling",
                    expression=sympy_conv(self.expansion.kernel.get_scaling()),
                    temp_var_type=lp.auto))

        return loopy_insns, result_names

# }}}


# {{{ box-local E2P (L2P, likely)

class E2PFromLocal(E2PBase):
    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt,idim]: itgt_start<=itgt<itgt_end and 0<=idim<dim}",
                    ],
                loopy_insns
                + ["""
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[tgt_ibox]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]
                    <> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}
                    <> b[idim] = targets[idim, itgt] - center[idim] \
                            {id=compute_b}
                    <> coeff${COEFFIDX} = expansions[tgt_ibox, ${COEFFIDX}]
                    result[${RESULTIDX},itgt] = \
                            kernel_scaling * result_${RESULTIDX}_p
                """],
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets"),
                        dim_tags="sep,C"),
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.GlobalArg("result", None, shape="nresults, ntargets",
                        dim_tags="sep,C"),
                    lp.GlobalArg("expansions", None, shape=("nboxes", ncoeffs)),
                    lp.ValueArg("nboxes,naligned_boxes", np.int32),
                    lp.ValueArg("ntargets", np.int32),
                    "..."
                ] + self.expansion.get_args(),
                name=self.name, assumptions="ntgt_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    COEFFIDX=[str(i) for i in xrange(ncoeffs)],
                    RESULTIDX=[str(i) for i in xrange(len(result_names))],
                    nresults=len(result_names),
                    )
                )

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "compute_b",
                tags={"idim": "unr"})
        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        #knl = lp.split_iname(knl, "itgt_box", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        """
        :arg expansions:
        :arg target_boxes:
        :arg box_target_starts:
        :arg box_target_counts_nonchild:
        :arg centers:
        :arg targets:
        """
        knl = self.get_optimized_kernel()

        return knl(queue, **kwargs)

# }}}


# {{{ E2P from CSR-like interaction list

class E2PFromCSR(E2PBase):
    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt]: itgt_start<=itgt<itgt_end}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_end }",
                    "{[idim]: 0<=idim<dim}",
                    ],
                loopy_insns
                + ["""
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[tgt_ibox]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]

                    <> tgt[idim] = targets[idim,itgt] {id=fetch_tgt}

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_end = source_box_starts[itgt_box+1]

                    <> src_ibox = source_box_lists[isrc_box]
                    <> coeff${COEFFIDX} = expansions[src_ibox, ${COEFFIDX}]
                    <> center[idim] = centers[idim, src_ibox] {id=fetch_center}

                    <> b[idim] = tgt[idim] - center[idim]
                    result[${RESULTIDX}, itgt] = \
                            kernel_scaling * sum(isrc_box, result_${RESULTIDX}_p)
                """],
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets"),
                        dim_tags="sep,C"),
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", ncoeffs)),
                    lp.ValueArg("nboxes,aligned_nboxes", np.int32),
                    lp.ValueArg("ntargets", np.int32),
                    lp.GlobalArg("result", None, shape="nresults,ntargets",
                        dim_tags="sep,C"),
                    lp.GlobalArg("source_box_starts, source_box_lists,",
                        None, shape=None),
                    "..."
                ] + self.expansion.get_args(),
                name=self.name, assumptions="ntgt_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    COEFFIDX=[str(i) for i in xrange(ncoeffs)],
                    RESULTIDX=[str(i) for i in xrange(len(result_names))],
                    nresults=len(result_names),
                    )
                )

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt",
                tags={"idim": "unr"})
        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_center",
                tags={"idim": "unr"})
        loopy_knl = lp.set_loop_priority(loopy_knl, "itgt_box,itgt,isrc_box")
        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, dict(itgt_box="g.0"))
        return knl

    def __call__(self, queue, **kwargs):
        knl = self.get_optimized_kernel()
        return knl(queue, **kwargs)

# }}}

# vim: foldmethod=marker
