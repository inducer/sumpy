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

from pytools.obj_array import make_obj_array

from sumpy.array_context import PyOpenCLArrayContext, make_loopy_program
from sumpy.tools import KernelCacheMixin


__doc__ = """

Expansion-to-particle
---------------------

.. autoclass:: E2PBase
.. autoclass:: E2PFromCSR
.. autoclass:: E2PFromSingleBox

"""


# {{{ E2PBase: base class

class E2PBase(KernelCacheMixin):
    def __init__(self, expansion, kernels, name=None):
        """
        :arg expansion: a subclass of :class:`sympy.expansion.ExpansionBase`
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """

        from sumpy.kernel import (
            SourceTransformationRemover, TargetTransformationRemover)
        sxr = SourceTransformationRemover()
        txr = TargetTransformationRemover()

        expansion = expansion.with_kernel(sxr(expansion.kernel))
        kernels = [sxr(knl) for knl in kernels]
        for knl in kernels:
            assert txr(knl) == expansion.kernel

        self.expansion = expansion
        self.kernels = kernels
        self.name = name or self.default_name

        self.dim = expansion.dim

    def get_loopy_insns_and_result_names(self):
        import sumpy.symbolic as sym
        bvec = sym.make_sym_vector("b", self.dim)

        import sumpy.symbolic as sp
        rscale = sp.Symbol("rscale")

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        coeff_exprs = [
                sym.Symbol(f"coeff{i}")
                for i in range(len(self.expansion.get_coefficient_identifiers()))]

        result_names = [
            sac.assign_unique(f"result_{i}_p",
                self.expansion.evaluate(knl, coeff_exprs, bvec, rscale, sac=sac))
            for i, knl in enumerate(self.kernels)
            ]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(
                sac.assignments.items(),
                vector_names={"b"},
                pymbolic_expr_maps=[
                    knl.get_code_transformer() for knl in self.kernels],
                retain_names=result_names,
                complex_dtype=np.complex128  # FIXME
                )

        return loopy_insns, result_names

    def get_kernel_scaling_assignment(self):
        from sumpy.symbolic import SympyToPymbolicMapper
        from sumpy.tools import ScalingAssignmentTag
        sympy_conv = SympyToPymbolicMapper()
        return [lp.Assignment(id=None,
                    assignee="kernel_scaling",
                    expression=sympy_conv(
                        self.expansion.kernel.get_global_scaling_const()),
                    temp_var_type=lp.Optional(None),
                    tags=frozenset([ScalingAssignmentTag()]),
                    )]

    def get_cache_key(self):
        return (type(self).__name__, self.expansion, tuple(self.kernels))

# }}}


# {{{ E2PFromSingleBox: E2P to single box (L2P, likely)

class E2PFromSingleBox(E2PBase):
    default_name = "e2p_from_single_box"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = make_loopy_program([
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
                kernel_data=[
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets"),
                        dim_tags="sep,C"),
                    lp.GlobalArg("box_target_starts,box_target_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.ValueArg("rscale", None),
                    lp.GlobalArg("result", None, shape="nresults, ntargets",
                        dim_tags="sep,C"),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("nsrc_level_boxes,naligned_boxes", np.int32),
                    lp.ValueArg("src_base_ibox", np.int32),
                    lp.ValueArg("ntargets", np.int32),
                    ...
                ] + [arg.loopy_arg for arg in self.expansion.get_args()],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_result*)",
                fixed_parameters=dict(dim=self.dim, nresults=len(result_names)),
                )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        for knl in self.kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, dict(itgt_box="g.0"))
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg expansions:
        :arg target_boxes:
        :arg box_target_starts:
        :arg box_target_counts_nonchild:
        :arg centers:
        :arg targets:
        """

        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        rscale = centers.dtype.type(kwargs.pop("rscale"))

        knl = self.get_cached_optimized_kernel()
        result = actx.call_loopy(
            knl,
            centers=centers, rscale=rscale, **kwargs)

        # FIXME: cleaner way to get the names out?
        return make_obj_array([result[f"result_s{i}"] for i in range()])

# }}}


# {{{ E2PFromCSR: E2P from CSR-like interaction list

class E2PFromCSR(E2PBase):
    default_name = "e2p_from_csr"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        loopy_insns, result_names = self.get_loopy_insns_and_result_names()

        loopy_knl = make_loopy_program([
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
                        """.format(resultidx=i) for i in range(len(result_names))]
                + ["""
                    end
                end
                """],
                kernel_data=[
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
                    ...
                ] + [arg.loopy_arg for arg in self.expansion.get_args()],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_result*)",
                fixed_parameters=dict(dim=self.dim, nresults=len(result_names)),
                )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.prioritize_loops(loopy_knl, "itgt_box,itgt,isrc_box")

        for knl in self.kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, dict(itgt_box="g.0"))
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        rscale = centers.dtype.type(kwargs.pop("rscale"))

        knl = self.get_cached_optimized_kernel()
        result = actx.call_loopy(
            knl,
            centers=centers, rscale=rscale, **kwargs)

        return make_obj_array([result[f"result_s{i}"] for i in range()])

# }}}

# vim: foldmethod=marker
