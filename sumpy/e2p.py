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


class E2P(object):
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

    @memoize_method
    def get_kernel(self):
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

        arguments = (
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets")),
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, nboxes"),
                    lp.GlobalArg("expansions", None,
                        shape=(len(coeff_exprs), "nboxes")),
                    lp.ValueArg("nboxes", np.int32),
                    lp.ValueArg("ntargets", np.int32),
                    "..."
                ] + [
                    lp.GlobalArg("result_%d" % i, None, shape="ntargets")
                    for i in range(len(result_names))
                ] + self.expansion.get_args()
                )

        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt,idim]: itgt_start<=itgt<itgt_end and 0<=idim<dim}",
                    ],
                loopy_insns
                + [
                    "<> tgt_ibox = target_boxes[itgt_box]",
                    "<> itgt_start = box_target_starts[tgt_ibox]",
                    "<> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]",
                    "<> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}",
                    "<> b[idim] = targets[idim, itgt] - center[idim]",
                    "<> coeff${COEFFIDX} = expansions[tgt_ibox, ${COEFFIDX}]",
                    "result_${RESULTIDX}[itgt] = kernel_scaling * result_${RESULTIDX}_p"
                ],
                arguments,
                name=self.name, assumptions="ntgt_boxes>=1",
                preambles=self.expansion.get_preambles(),
                defines=dict(
                    dim=self.dim,
                    COEFFIDX=[str(i) for i in xrange(len(coeff_exprs))],
                    RESULTIDX=[str(i) for i in xrange(len(result_names))],
                    )
                )

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_center",
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
