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
from pytools import memoize_method

import numpy as np
import loopy as lp
from loopy.kernel.data import LocalInameTag
import pymbolic.primitives as prim

from sumpy.tools import KernelCacheMixin, gather_loopy_arguments
from loopy.version import MOST_RECENT_LANGUAGE_VERSION


__doc__ = """

Expansion-to-particle
---------------------

.. autoclass:: E2PBase
.. autoclass:: E2PFromCSR
.. autoclass:: E2PFromSingleBox

"""


# {{{ E2P base class

class E2PBase(KernelCacheMixin, ABC):
    def __init__(self, ctx, expansion, kernels,
            name=None, device=None):
        """
        :arg expansion: a subclass of :class:`sympy.expansion.ExpansionBase`
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """

        if device is None:
            device = ctx.devices[0]

        from sumpy.kernel import (SourceTransformationRemover,
                TargetTransformationRemover)
        sxr = SourceTransformationRemover()
        txr = TargetTransformationRemover()
        expansion = expansion.with_kernel(
                sxr(expansion.kernel))

        kernels = [sxr(knl) for knl in kernels]
        for knl in kernels:
            assert txr(knl) == expansion.kernel

        self.context = ctx
        self.expansion = expansion
        self.kernels = tuple(kernels)
        self.name = name or self.default_name
        self.device = device

        self.dim = expansion.dim

    @property
    @abstractmethod
    def default_name(self):
        pass

    @memoize_method
    def get_loopy_evaluator_and_optimizations(self):
        return self.expansion.loopy_evaluator_and_optimizations(self.kernels)

    def get_cache_key(self):
        return (type(self).__name__, self.expansion, tuple(self.kernels))

    def add_loopy_eval_callable(
            self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        inner_knl, _ = self.get_loopy_evaluator_and_optimizations()
        loopy_knl = lp.merge([loopy_knl, inner_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "e2p")
        for kernel in self.kernels:
            loopy_knl = kernel.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.tag_array_axes(loopy_knl, "targets", "sep,C")
        return loopy_knl

    def get_loopy_args(self):
        return gather_loopy_arguments((self.expansion,) + tuple(self.kernels))

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


# {{{ E2P to single box (L2P, likely)

class E2PFromSingleBox(E2PBase):
    @property
    def default_name(self):
        return "e2p_from_single_box"

    def get_kernel(self, max_ntargets_in_one_box, max_work_items):
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[idim]: 0<=idim<dim}",
                    "{[itgt_offset]: 0<=itgt_offset<max_ntargets_in_one_box}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                    "{[dummy]: 0<=dummy<max_work_items}",
                ],
                self.get_kernel_scaling_assignment()
                + ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]   {id=fetch_init0}
                    <> itgt_start = box_target_starts[tgt_ibox]  {id=fetch_init1}
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox] \
                            {id=fetch_init2}

                    <> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}

                    <> coeffs[icoeff] = \
                            src_expansions[tgt_ibox - src_base_ibox, icoeff] \
                            {id=fetch_coeffs}

                    for itgt_offset
                        <> itgt = itgt_start + itgt_offset
                        <> run_itgt = itgt<itgt_end
                        <> tgt[idim] = targets[idim, itgt] {id=fetch_tgt, \
                            dup=idim,if=run_itgt}
                        <> result_temp[iknl] = 0  {id=init_result,dup=iknl, \
                            if=run_itgt}
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
                                id=update_result,if=run_itgt}
                        result[iknl, itgt] = result_temp[iknl] * kernel_scaling \
                            {id=write_result,dep=update_result,if=run_itgt}
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
                    "..."
                ],
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(*_result)",
                default_offset=lp.auto,
                fixed_parameters={"dim": self.dim, "nresults": len(self.kernels),
                    "ncoeffs": ncoeffs,
                    "max_work_items": max_work_items,
                    "max_ntargets_in_one_box": max_ntargets_in_one_box},
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "iknl*:unr")
        loopy_knl = self.add_loopy_eval_callable(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self, max_ntargets_in_one_box):
        _, optimizations = self.get_loopy_evaluator_and_optimizations()

        ncoeffs = len(self.expansion)
        max_work_items = min(256, max(ncoeffs, max_ntargets_in_one_box))
        knl = self.get_kernel(max_ntargets_in_one_box=max_ntargets_in_one_box,
                              max_work_items=max_work_items)

        knl = lp.tag_inames(knl, {"itgt_box": "g.0"})
        knl = lp.split_iname(knl, "itgt_offset", max_work_items, inner_tag="l.0")
        knl = lp.split_iname(knl, "icoeff", max_work_items, inner_tag="l.0")
        knl = lp.add_inames_to_insn(knl, "dummy",
            "id:fetch_init* or id:fetch_center or id:kernel_scaling")
        knl = lp.add_inames_to_insn(knl, "itgt_box", "id:kernel_scaling")
        knl = lp.tag_inames(knl, {"dummy": "l.0"})
        knl = lp.set_temporary_address_space(knl, "coeffs", lp.AddressSpace.LOCAL)
        knl = lp.set_options(knl,
            enforce_variable_access_ordered="no_check", write_code=False)

        for transform in optimizations:
            knl = transform(knl)

        # If there are inames tagged as local in the inner kernel
        # we need to remove the iname itgt_offset_inner from instructions
        # within those inames and also remove the predicate run_itgt
        # which depends on itgt_offset_inner
        tagged_inames = [iname.name for iname in
            knl.default_entrypoint.inames.values() if
            iname.name.startswith("e2p_") and any(
                isinstance(tag, LocalInameTag) for tag in iname.tags)]
        if tagged_inames:
            insn_ids = [insn.id for insn in knl.default_entrypoint.instructions
                if any(iname in tagged_inames for iname in insn.within_inames)]
            match = " or ".join(f"id:{insn_id}" for insn_id in insn_ids)
            knl = lp.remove_inames_from_insn(knl,
                frozenset(["itgt_offset_inner"]), match)
            knl = lp.remove_predicates_from_insn(knl,
                frozenset([prim.Variable("run_itgt")]), match)

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
        max_ntargets_in_one_box = kwargs.pop("max_ntargets_in_one_box")
        knl = self.get_cached_kernel_executor(
                max_ntargets_in_one_box=max_ntargets_in_one_box)

        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        rscale = centers.dtype.type(kwargs.pop("rscale"))

        return knl(queue, centers=centers, rscale=rscale, **kwargs)

# }}}


# {{{ E2P from CSR-like interaction list

class E2PFromCSR(E2PBase):
    @property
    def default_name(self):
        return "e2p_from_csr"

    def get_kernel(self, max_ntargets_in_one_box, max_work_items):
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt_offset]: 0<=itgt_offset<max_ntargets_in_one_box}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_end }",
                    "{[idim]: 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                    "{[dummy]: 0<=dummy<max_work_items}",
                ],
                self.get_kernel_scaling_assignment()
                + ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box] {id=init_box0}
                    <> itgt_start = box_target_starts[tgt_ibox] {id=init_box1}
                    <> itgt_end = itgt_start+box_target_counts_nonchild[tgt_ibox] \
                            {id=init_box2}
                    <> isrc_box_start = source_box_starts[itgt_box] {id=init_box3}
                    <> isrc_box_end = source_box_starts[itgt_box+1] {id=init_box4}

                    <> result_temp[itgt_offset, iknl] = 0 \
                                {id=init_result,dup=iknl}
                    for isrc_box
                        <> src_ibox = source_box_lists[isrc_box] {id=fetch_src_box}
                        <> coeffs[icoeff] = \
                                src_expansions[src_ibox - src_base_ibox, icoeff] \
                                {id=fetch_coeffs}
                        <> center[idim] = centers[idim, src_ibox] \
                                {dup=idim,id=fetch_center}

                        for itgt_offset
                            <> itgt = itgt_start + itgt_offset
                            <> run_itgt = itgt<itgt_end
                            <> tgt[idim] = targets[idim,itgt]  \
                                {id=fetch_tgt,dup=idim,if=run_itgt}

                            [iknl]: result_temp[itgt_offset, iknl] = e2p(
                                [iknl]: result_temp[itgt_offset, iknl],
                                [icoeff]: coeffs[icoeff],
                                [idim]: center[idim],
                                [idim]: tgt[idim],
                                rscale,
                                itgt,
                                ntargets,
                                targets,
                """ + ",".join(arg.name for arg in loopy_args) + """
                            )  {id=update_result, \
                              dep=fetch_coeffs:fetch_center:fetch_tgt:init_result, \
                              if=run_itgt}
                        end
                    end
                    for itgt_offset
                        <> itgt2 = itgt_start + itgt_offset {id=init_itgt_for_write}
                        <> run_itgt2 = itgt_start + itgt_offset < itgt_end  \
                                {id=init_cond_for_write}
                        result[iknl, itgt2] = result[iknl, itgt2] + result_temp[ \
                            itgt_offset, iknl] * kernel_scaling \
                            {dep=update_result:init_result,id=write_result, \
                             dup=iknl,if=run_itgt2}
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
                default_offset=lp.auto,
                fixed_parameters={
                        "ncoeffs": ncoeffs,
                        "dim": self.dim,
                        "max_work_items": max_work_items,
                        "max_ntargets_in_one_box": max_ntargets_in_one_box,
                        "nresults": len(self.kernels)},
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "iknl*:unr")
        loopy_knl = lp.prioritize_loops(loopy_knl, "itgt_box,isrc_box,itgt_offset")
        loopy_knl = self.add_loopy_eval_callable(loopy_knl)
        loopy_knl = lp.tag_array_axes(loopy_knl, "targets", "sep,C")

        return loopy_knl

    def get_optimized_kernel(self, max_ntargets_in_one_box):
        _, optimizations = self.get_loopy_evaluator_and_optimizations()
        ncoeffs = len(self.expansion)
        max_work_items = min(256, max(ncoeffs, max_ntargets_in_one_box))

        knl = self.get_kernel(max_ntargets_in_one_box=max_ntargets_in_one_box,
                              max_work_items=max_work_items)
        knl = lp.tag_inames(knl, {"itgt_box": "g.0", "dummy": "l.0"})
        knl = lp.unprivatize_temporaries_with_inames(knl,
            "itgt_offset", "result_temp")
        knl = lp.split_iname(knl, "itgt_offset", max_work_items, inner_tag="l.0")
        knl = lp.split_iname(knl, "icoeff", max_work_items, inner_tag="l.0")
        knl = lp.privatize_temporaries_with_inames(knl,
            "itgt_offset_outer", "result_temp")
        knl = lp.duplicate_inames(knl, "itgt_offset_outer", "id:init_result")
        knl = lp.duplicate_inames(knl, "itgt_offset_outer",
            "id:write_result or id:init_itgt_for_write or id:init_cond_for_write")
        knl = lp.add_inames_to_insn(knl, "dummy",
            "id:init_box* or id:fetch_src_box or id:fetch_center "
            "or id:kernel_scaling")
        knl = lp.add_inames_to_insn(knl, "itgt_box", "id:kernel_scaling")
        knl = lp.add_inames_to_insn(knl, "itgt_offset_inner", "id:fetch_init*")
        knl = lp.set_temporary_address_space(knl, "coeffs", lp.AddressSpace.LOCAL)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check", write_code=False)
        for transform in optimizations:
            knl = transform(knl)
        return knl

    def __call__(self, queue, **kwargs):
        max_ntargets_in_one_box = kwargs.pop("max_ntargets_in_one_box")
        knl = self.get_cached_kernel_executor(
                max_ntargets_in_one_box=max_ntargets_in_one_box)

        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        rscale = centers.dtype.type(kwargs.pop("rscale"))

        return knl(queue, centers=centers, rscale=rscale, **kwargs)

# }}}

# vim: foldmethod=marker
