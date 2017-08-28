from __future__ import division, absolute_import

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

import six
from six.moves import range

import numpy as np
import loopy as lp
import sumpy.symbolic as sym
from sumpy.tools import KernelCacheWrapper


__doc__ = """

Expansion-to-particle
---------------------

.. autoclass:: E2PBase
.. autoclass:: E2PFromCSR
.. autoclass:: E2PFromSingleBox

"""


# {{{ E2P base class

class E2PBase(KernelCacheWrapper):
    def __init__(self, ctx, expansion, kernels,
            options=[], name=None, device=None):
        """
        :arg expansion: a subclass of :class:`sympy.expansion.ExpansionBase`
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """

        if device is None:
            device = ctx.devices[0]

        from sumpy.kernel import SourceDerivativeRemover, TargetDerivativeRemover
        sdr = SourceDerivativeRemover()
        tdr = TargetDerivativeRemover()
        expansion = expansion.with_kernel(
                sdr(expansion.kernel))

        for knl in kernels:
            assert sdr(tdr(knl)) == expansion.kernel

        self.ctx = ctx
        self.expansion = expansion
        self.kernels = kernels
        self.options = options
        self.name = name or self.default_name
        self.device = device

        self.dim = expansion.dim

    def get_loopy_insns_and_result_names(self):
        from sumpy.symbolic import make_sym_vector
        bvec = make_sym_vector("b", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        coeff_exprs = [sym.Symbol("coeff%d" % i)
                for i in range(len(self.expansion.get_coefficient_identifiers()))]
        value = self.expansion.evaluate(coeff_exprs, bvec)

        result_names = [
            sac.assign_unique("result_%d_p" % i,
                knl.postprocess_at_target(value, bvec))
            for i, knl in enumerate(self.kernels)
            ]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(
                six.iteritems(sac.assignments),
                vector_names=set(["b"]),
                pymbolic_expr_maps=[self.expansion.get_code_transformer()],
                retain_names=result_names,
                complex_dtype=np.complex128  # FIXME
                )

        return loopy_insns, result_names

    def get_kernel_scaling_assignment(self):
        from sumpy.symbolic import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()
        return [lp.Assignment(id=None,
                    assignee="kernel_scaling",
                    expression=sympy_conv(self.expansion.kernel.get_scaling()),
                    temp_var_type=lp.auto)]

    def get_cache_key(self):
        return (type(self).__name__, self.expansion, tuple(self.kernels))

# }}}


# {{{ E2P to single box (L2P, likely)

class E2PFromSingleBox(E2PBase):
    default_name = "e2p_from_single_box"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt,idim]: itgt_start<=itgt<itgt_end and 0<=idim<dim}",
                    ],
                self.get_kernel_scaling_assignment()
                + ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[tgt_ibox]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]

                    <> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}

                    """] + ["""
                    <> coeff{coeffidx} = \
                            src_expansions[tgt_ibox - src_base_ibox, {coeffidx}]
                    """.format(coeffidx=i) for i in range(ncoeffs)] + ["""

                    for itgt
                        <> b[idim] = targets[idim, itgt] - center[idim] {dup=idim}

                        """] + loopy_insns + ["""

                        result[{resultidx},itgt] = \
                                kernel_scaling * result_{resultidx}_p \
                                {{id_prefix=write_result}}
                        """.format(resultidx=i) for i in range(len(result_names))
                        ] + ["""
                    end
                end
                """],
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets"),
                        dim_tags="sep,C"),
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.GlobalArg("result", None, shape="nresults, ntargets",
                        dim_tags="sep,C"),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("nsrc_level_boxes,naligned_boxes", np.int32),
                    lp.ValueArg("src_base_ibox", np.int32),
                    lp.ValueArg("ntargets", np.int32),
                    "..."
                ] + [arg.loopy_arg for arg in self.expansion.get_args()],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_result*)",
                default_offset=lp.auto)

        loopy_knl = lp.fix_parameters(loopy_knl,
                dim=self.dim,
                nresults=len(result_names))

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, dict(itgt_box="g.0"))
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
        knl = self.get_cached_optimized_kernel()

        return knl(queue, **kwargs)

# }}}


# {{{ E2P from CSR-like interaction list

class E2PFromCSR(E2PBase):
    default_name = "e2p_from_csr"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt]: itgt_start<=itgt<itgt_end}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_end }",
                    "{[idim]: 0<=idim<dim}",
                    ],
                self.get_kernel_scaling_assignment()
                + ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[tgt_ibox]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]

                    for itgt
                        <> tgt[idim] = targets[idim,itgt]

                        <> isrc_box_start = source_box_starts[itgt_box]
                        <> isrc_box_end = source_box_starts[itgt_box+1]

                        for isrc_box
                            <> src_ibox = source_box_lists[isrc_box]
                            """] + ["""
                            <> coeff{coeffidx} = \
                                src_expansions[src_ibox - src_base_ibox, {coeffidx}]
                            """.format(coeffidx=i) for i in range(ncoeffs)] + ["""

                            <> center[idim] = centers[idim, src_ibox] {dup=idim}
                            <> b[idim] = tgt[idim] - center[idim] {dup=idim}

                            """] + loopy_insns + ["""
                        end
                        """] + ["""
                        result[{resultidx}, itgt] = result[{resultidx}, itgt] + \
                                kernel_scaling * simul_reduce(sum, isrc_box,
                                result_{resultidx}_p) {{id_prefix=write_result}}
                        """.format(resultidx=i) for i in range(len(result_names))] + ["""
                    end
                end
                """],
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets"),
                        dim_tags="sep,C"),
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("src_base_ibox", np.int32),
                    lp.ValueArg("nsrc_level_boxes,aligned_nboxes", np.int32),
                    lp.ValueArg("ntargets", np.int32),
                    lp.GlobalArg("result", None, shape="nresults,ntargets",
                        dim_tags="sep,C"),
                    lp.GlobalArg("source_box_starts, source_box_lists,",
                        None, shape=None, offset=lp.auto),
                    "..."
                ] + [arg.loopy_arg for arg in self.expansion.get_args()],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_result*)",
                default_offset=lp.auto)

        loopy_knl = lp.fix_parameters(loopy_knl,
                dim=self.dim,
                nresults=len(result_names))

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.prioritize_loops(loopy_knl, "itgt_box,itgt,isrc_box")
        loopy_knl = self.expansion.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, dict(itgt_box="g.0"))
        return knl

    def __call__(self, queue, **kwargs):
        knl = self.get_cached_optimized_kernel()
        return knl(queue, **kwargs)

# }}}

# vim: foldmethod=marker
