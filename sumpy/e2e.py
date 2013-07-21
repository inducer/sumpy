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


class E2E(object):
    def __init__(self, ctx, src_expansion, tgt_expansion,
            options=[], name="e2e", device=None):
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
        self.src_expansion = src_expansion
        self.tgt_expansion = tgt_expansion
        self.options = options
        self.name = name
        self.device = device

        if src_expansion.dim != tgt_expansion.dim:
            raise ValueError("source and target expansions must have "
                    "same dimensionality")

        self.dim = src_expansion.dim

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("d", self.dim)

        ncoeff_src = len(self.src_expansion.get_coefficient_identifiers())
        ncoeff_tgt = len(self.tgt_expansion.get_coefficient_identifiers())

        src_coeff_exprs = [sp.Symbol("src_coeff%d" % i)
                for i in xrange(ncoeff_src)]

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique("coeff%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, dvec))]

        sac.run_global_cse()

        from sumpy.symbolic import kill_trivial_assignments
        assignments = kill_trivial_assignments([
                (name, expr)
                for name, expr in sac.assignments.iteritems()],
                retain_names=tgt_coeff_names)

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(assignments,
                vector_names=set(["d"]),
                pymbolic_expr_maps=[self.tgt_expansion.transform_to_code],
                complex_dtype=np.complex128  # FIXME
                )

        from sumpy.tools import gather_arguments
        arguments = (
                [
                    lp.GlobalArg("centers", None, shape="dim, nboxes"),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nboxes", ncoeff_src)),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("nboxes", ncoeff_tgt)),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,)),
                    lp.ValueArg("nboxes", np.int32),
                    "..."
                ] + gather_arguments([self.src_expansion, self.tgt_expansion])
                )

        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                loopy_insns
                + ["""
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> tgt_center[idim] = centers[idim, tgt_ibox] \
                            {id=fetch_tgt_center}

                    <> isrc_start = src_box_starts[itgt_box]
                    <> isrc_stop = src_box_starts[itgt_box+1]

                    <> src_ibox = src_box_lists[isrc_box]
                    <> src_center[idim] = centers[idim, src_ibox] \
                            {id=fetch_src_center}
                    <> d[idim] = tgt_center[idim] - src_center[idim]

                    <> src_coeff${SRC_COEFFIDX} = \
                        src_expansions[src_ibox, ${SRC_COEFFIDX}]
                    tgt_expansions[tgt_ibox, ${TGT_COEFFIDX}] = \
                        coeff${TGT_COEFFIDX} {id_prefix=write_expn}
                    """],
                arguments,
                name=self.name, assumptions="ntgt_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    SRC_COEFFIDX=[str(i) for i in xrange(ncoeff_src)],
                    TGT_COEFFIDX=[str(i) for i in xrange(ncoeff_tgt)],
                    ),
                silenced_warnings="write_race(write_expn*)")

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt_box", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg centers:
        """
        knl = self.get_optimized_kernel()

        return knl(queue, **kwargs)
