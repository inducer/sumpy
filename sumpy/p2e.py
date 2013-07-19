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
from pytools import memoize_method


class P2E(object):
    def __init__(self, ctx, expansion,
            options=[], name="p2e", device=None):
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
        self.options = options
        self.name = name
        self.device = device

        self.dim = expansion.dim

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector
        avec = make_sym_vector("a", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        coeff_names = [
                sac.assign_unique("coeff%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.expansion.coefficients_from_source(avec, None))]

        sac.run_global_cse()

        from sumpy.symbolic import kill_trivial_assignments
        assignments = kill_trivial_assignments([
                (name, expr)
                for name, expr in sac.assignments.iteritems()],
                retain_names=coeff_names)

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(assignments,
                vector_names=set(["a"]),
                pymbolic_expr_maps=[self.expansion.transform_to_code],
                complex_dtype=np.complex128  # FIXME
                )

        from sumpy.tools import gather_source_arguments
        arguments = (
                [
                    lp.GlobalArg("sources", None, shape=(self.dim, "nsrc")),
                    lp.GlobalArg("strengths", None, shape="nsrc"),
                    lp.GlobalArg("box_source_starts,box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, nboxes"),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", len(coeff_names))),
                    lp.ValueArg("nboxes", np.int32),
                    lp.ValueArg("nsrc", np.int32),
                    "..."
                ] + gather_source_arguments([self.expansion]))

        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[isrc_box]: 0<=isrc_box<nsrc_boxes}",
                    "{[isrc,idim]: isrc_start<=isrc<isrc_end and 0<=idim<dim}",
                    ],
                loopy_insns
                + [
                    "<> src_ibox = source_boxes[isrc_box]",
                    "<> isrc_start = box_source_starts[src_ibox]",
                    "<> isrc_end = isrc_start+box_source_counts_nonchild[src_ibox]",
                    "<> center[idim] = centers[idim, src_ibox] {id=fetch_center}",
                    "<> a[idim] = center[idim] - sources[idim, isrc]",
                    "<> strength = strengths[isrc]",
                    "expansions[src_ibox, ${COEFFIDX}] = "
                            "sum(isrc, strength*coeff${COEFFIDX})"
                            "{id_prefix=write_expn}"
                ],
                arguments,
                name=self.name, assumptions="nsrc_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    COEFFIDX=[str(i) for i in xrange(len(coeff_names))]
                    ),
                silenced_warnings="write_race(write_expn*)")

        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_center",
                tags={"idim": "unr"})

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "isrc_box", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
        """
        :arg expansions:
        :arg source_boxes:
        :arg box_source_starts:
        :arg box_source_counts_nonchild:
        :arg centers:
        :arg sources:
        :arg strengths:
        """
        knl = self.get_optimized_kernel()

        return knl(queue, **kwargs)
