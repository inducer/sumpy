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
import loopy as lp
from pytools import memoize_method

from sumpy.tools import KernelComputation


# LATER:
# - Optimization for source == target (postpone)

# {{{ p2p base class

class P2PBase(KernelComputation):
    def __init__(self, ctx, kernels,  exclude_self, strength_usage=None,
            value_dtypes=None,
            options=[], name="p2p", device=None):
        """
        :arg kernels: list of :class:`sumpy.kernel.Kernel` instances
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """
        KernelComputation.__init__(self, ctx, kernels, strength_usage,
                value_dtypes,
                name, options, device)

        self.exclude_self = exclude_self

        from pytools import single_valued
        self.dim = single_valued(knl.dim for knl in self.kernels)

    def get_loopy_insns_and_result_names(self):
        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("d", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        result_names = [
                sac.assign_unique("knl%d" % i,
                    knl.postprocess_at_target(
                        knl.postprocess_at_source(
                            knl.get_expression(dvec), dvec),
                        dvec))
                        for i, knl in enumerate(self.kernels)]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(sac.assignments.iteritems(),
                vector_names=set(["d"]),
                pymbolic_expr_maps=[knl.transform_to_code for knl in self.kernels],
                complex_dtype=np.complex128  # FIXME
                )

        return loopy_insns, result_names

# }}}


# {{{ P2P with list of sources and list of targets

class P2P(P2PBase):
    @memoize_method
    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        from pymbolic import var
        exprs = [
                var(name)
                    * var("strength_%d" % self.strength_usage[i])[var("isrc")]
                for i, name in enumerate(result_names)]

        from sumpy.tools import gather_source_arguments
        arguments = (
                [
                    lp.GlobalArg("src", None,
                        shape=(self.dim, "nsources"), order="C"),
                    lp.GlobalArg("tgt", None,
                        shape=(self.dim, "ntargets"), order="C"),
                    lp.ValueArg("nsources", None),
                    lp.ValueArg("ntargets", np.int32),
                ]+[
                    lp.GlobalArg("strength_%d" % i, None, shape="nsources", order="C")
                    for i in xrange(self.strength_count)
                ]+[
                    lp.GlobalArg("result_%d" % i, dtype, shape="ntargets", order="C")
                    for i, dtype in enumerate(self.value_dtypes)
                ] + gather_source_arguments(self.kernels))

        if self.exclude_self:
            from pymbolic.primitives import If, ComparisonOperator, Variable
            exprs = [
                    If(
                        ComparisonOperator(Variable("isrc"), "!=", Variable("itgt")),
                        expr, 0)
                    for expr in exprs]

        loopy_knl = lp.make_kernel(self.device,
                "{[isrc,itgt,idim]: 0<=itgt<ntargets and 0<=isrc<nsources and 0<=idim<dim}",
                self.get_kernel_scaling_assignments()
                + loopy_insns
                + [
                "<> d[idim] = tgt[idim,itgt] - src[idim,isrc] {id=compute_d}",
                ]+[
                    lp.ExpressionInstruction(id=None,
                        assignee="pair_result_%d" % i, expression=expr,
                        temp_var_type=lp.auto)
                    for i, expr in enumerate(exprs)
                ]+[
                    "result_${KNLIDX}[itgt] = knl_${KNLIDX}_scaling \
                            * sum(isrc, pair_result_${KNLIDX})"
                ],
                arguments,
                name=self.name, assumptions="nsources>=1 and ntargets>=1",
                defines=dict(
                    KNLIDX=range(len(exprs)),
                    dim=self.dim,
                    ))

        for where in ["compute_a", "compute_b"]:
            loopy_knl = lp.duplicate_inames(loopy_knl, "idim", where)

        for knl in self.kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt", 1024, outer_tag="g.0")
        return knl

    def __call__(self, queue, targets, sources, src_strengths, **kwargs):
        knl = self.get_optimized_kernel()

        for i, sstr in enumerate(src_strengths):
            kwargs["strength_%d" % i] = sstr

        return knl(queue, src=sources, tgt=targets, **kwargs)

# }}}


# {{{ P2P from CSR-like interaction list

class P2PFromCSR(P2PBase):
    # FIXME: exclude_self ...?

    @memoize_method
    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        from pymbolic import var
        exprs = [
                var(name)
                * var("strength_%d" % self.strength_usage[i])[var("isrc")]
                for i, name in enumerate(result_names)]

        from sumpy.tools import gather_source_arguments
        arguments = (
                [
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild,"
                        "box_source_starts,box_source_counts_nonchild",
                        None, shape=None)
                ]+[
                    lp.GlobalArg("strength_%d" % i, None, shape="nsources")
                    for i in xrange(self.strength_count)
                ]+[
                    lp.GlobalArg("result_%d" % i, dtype, shape="ntargets")
                    for i, dtype in enumerate(self.value_dtypes)
                ] + gather_source_arguments(self.kernels))

        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<=isrc_box_end}",
                    "{[itgt,isrc,idim]: \
                            itgt_start<=itgt<itgt_end and \
                            isrc_start<=isrc<isrc_end and \
                            0<=idim<dim }",
                    ],
                self.get_kernel_scaling_assignments()
                + loopy_insns + [
                    """
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[itgt_box]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[itgt_box]

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_end = source_box_starts[itgt_box+1]

                    <> src_ibox = source_box_lists[isrc_box]
                    <> isrc_start = box_source_starts[isrc_box]
                    <> isrc_end = isrc_start+box_source_counts_nonchild[isrc_box]

                    <> d[idim] = targets[idim,itgt] - sources[idim,isrc] \
                            {id=compute_d}

                    result[${KNLIDX}, itgt] = result[${KNLIDX}, itgt] + \
                            knl_${KNLIDX}_scaling \
                            * sum(isrc, pair_result_${KNLIDX})
                    """
                ]+[
                    lp.ExpressionInstruction(id=None,
                        assignee="pair_result_%d" % i, expression=expr,
                        temp_var_type=lp.auto)
                    for i, expr in enumerate(exprs)
                ]+[
                ],
                arguments,
                name=self.name, assumptions="nsources>=1 and ntargets>=1",
                defines=dict(
                    KNLIDX=range(len(exprs)),
                    dim=self.dim,
                    ))

        for where in ["compute_a", "compute_b"]:
            loopy_knl = lp.duplicate_inames(loopy_knl, "idim", where)

        for knl in self.kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt", 1024, outer_tag="g.0")
        return knl

    def __call__(self, queue, targets, sources, src_strengths, **kwargs):
        knl = self.get_optimized_kernel()

        for i, sstr in enumerate(src_strengths):
            kwargs["strength_%d" % i] = sstr

        return knl(queue, src=sources, tgt=targets, **kwargs)

# }}}

# vim: foldmethod=marker
