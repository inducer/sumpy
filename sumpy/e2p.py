from __future__ import annotations


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

from abc import ABC, abstractmethod

import numpy as np

import loopy as lp
import pytools.obj_array as obj_array
from loopy.version import MOST_RECENT_LANGUAGE_VERSION  # noqa: F401

from sumpy.array_context import PyOpenCLArrayContext, make_loopy_program
from sumpy.tools import KernelCacheMixin, gather_loopy_arguments


__doc__ = """

Expansion-to-particle
---------------------

.. autoclass:: E2PBase
.. autoclass:: E2PFromCSR
.. autoclass:: E2PFromSingleBox

"""


# {{{ E2PBase: base class

class E2PBase(KernelCacheMixin, ABC):
    def __init__(self, expansion, kernels, name=None):
        """
        :arg expansion: a subclass of :class:`sympy.expansion.ExpansionBase`
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """

        from sumpy.kernel import (
            SourceTransformationRemover,
            TargetTransformationRemover,
        )
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

    @property
    def nresults(self):
        return len(self.kernels)

    @abstractmethod
    def default_name(self):
        pass

    def get_cache_key(self):
        return (type(self).__name__, self.expansion, tuple(self.kernels))

    def add_loopy_eval_callable(
            self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        inner_knl = self.expansion.loopy_evaluator(self.kernels)
        loopy_knl = lp.merge([loopy_knl, inner_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "e2p")
        loopy_knl = lp.remove_unused_inames(loopy_knl)
        for kernel in self.kernels:
            loopy_knl = kernel.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.tag_array_axes(loopy_knl, "targets", "sep,C")
        return loopy_knl

    def get_loopy_args(self):
        return gather_loopy_arguments((self.expansion, *tuple(self.kernels)))

    def get_kernel_scaling_assignment(self):
        from sumpy.symbolic import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()
        return [lp.Assignment(id="kernel_scaling",
                    assignee="kernel_scaling",
                    expression=sympy_conv(
                        self.expansion.kernel.get_global_scaling_const()),
                    temp_var_type=lp.Optional(None),
                    )]
# }}}


# {{{ E2PFromSingleBox: E2P to single box (L2P, likely)

class E2PFromSingleBox(E2PBase):
    @property
    def default_name(self):
        return "e2p_from_single_box"

    def get_kernel(self):
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        loopy_knl = make_loopy_program(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt,idim]: itgt_start<=itgt<itgt_end and 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                ], [
                *self.get_kernel_scaling_assignment(),
                """
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[tgt_ibox]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]

                    <> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}

                    <> coeffs[icoeff] = \
                            src_expansions[tgt_ibox - src_base_ibox, icoeff] \
                            {id=fetch_coeffs}

                    for itgt
                        <> tgt[idim] = targets[idim, itgt] {id=fetch_tgt,dup=idim}
                        <> result_temp[iknl] = 0  {id=init_result,dup=iknl}
                        [iknl]: result_temp[iknl] = e2p(
                            [iknl]: result_temp[iknl],
                            [icoeff]: coeffs[icoeff],
                            [idim]: center[idim],
                            [idim]: tgt[idim],
                            rscale,
                            itgt,
                            ntargets,
                            targets,
                """ + ",".join(arg.name for arg in loopy_args) + """
                        )  {dep=fetch_coeffs:fetch_center:init_result:fetch_tgt,\
                                id=update_result}
                        result[iknl, itgt] = result_temp[iknl] * kernel_scaling \
                            {id=write_result,dep=update_result}
                    end
                end
                """],
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets")),
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
                    *loopy_args,
                    ...
                ],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(*_result)",
                fixed_parameters={"dim": self.dim, "nresults": len(self.kernels),
                        "ncoeffs": ncoeffs})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "iknl*:unr")
        loopy_knl = self.add_loopy_eval_callable(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, {"itgt_box": "g.0"})
        knl = lp.add_inames_to_insn(knl, "itgt_box", "id:kernel_scaling")
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

        knl = self.get_cached_kernel_executor()
        result = actx.call_loopy(
            knl,
            centers=centers, rscale=rscale, **kwargs)

        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}


# {{{ E2PFromCSR: E2P from CSR-like interaction list

class E2PFromCSR(E2PBase):
    @property
    def default_name(self):
        return "e2p_from_csr"

    def get_kernel(self):
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        loopy_knl = make_loopy_program(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt]: itgt_start<=itgt<itgt_end}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_end }",
                    "{[idim]: 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                ], [
                *self.get_kernel_scaling_assignment(),
                """
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> itgt_start = box_target_starts[tgt_ibox]
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox]

                    for itgt
                        <> tgt[idim] = targets[idim,itgt]  {id=fetch_tgt,dup=idim}

                        <> isrc_box_start = source_box_starts[itgt_box]
                        <> isrc_box_end = source_box_starts[itgt_box+1]

                        <> result_temp[iknl] = 0 {id=init_result,dup=iknl}
                        for isrc_box
                            <> src_ibox = source_box_lists[isrc_box]
                            <> coeffs[icoeff] = \
                                src_expansions[src_ibox - src_base_ibox, icoeff] \
                                {id=fetch_coeffs,dup=icoeff}
                            <> center[idim] = centers[idim, src_ibox] \
                                {dup=idim,id=fetch_center}
                            [iknl]: result_temp[iknl] = e2p(
                                [iknl]: result_temp[iknl],
                                [icoeff]: coeffs[icoeff],
                                [idim]: center[idim],
                                [idim]: tgt[idim],
                                rscale,
                                itgt,
                                ntargets,
                                targets,
                """ + ",".join(arg.name for arg in loopy_args) + """
                            )  {id=update_result, \
                              dep=fetch_coeffs:fetch_center:fetch_tgt:init_result}
                        end
                        result[iknl, itgt] = result[iknl, itgt] + result_temp[iknl] \
                                * kernel_scaling \
                                {dep=update_result:init_result,id=write_result,dup=iknl}
                    end
                end
                """],
                [
                    lp.GlobalArg("targets", None, shape=(self.dim, "ntargets")),
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
                    *loopy_args,
                    "..."
                ],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(*_result)",
                fixed_parameters={
                        "ncoeffs": ncoeffs,
                        "dim": self.dim,
                        "nresults": len(self.kernels)})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "iknl*:unr")
        loopy_knl = lp.prioritize_loops(loopy_knl, "itgt_box,itgt,isrc_box")
        loopy_knl = self.add_loopy_eval_callable(loopy_knl)
        loopy_knl = lp.tag_array_axes(loopy_knl, "targets", "sep,C")

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, {"itgt_box": "g.0"})
        knl = lp.add_inames_to_insn(knl, "itgt_box", "id:kernel_scaling")
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        rscale = centers.dtype.type(kwargs.pop("rscale"))

        knl = self.get_cached_kernel_executor()
        result = actx.call_loopy(
            knl,
            centers=centers,
            rscale=rscale,
            **kwargs)

        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}

# vim: foldmethod=marker
