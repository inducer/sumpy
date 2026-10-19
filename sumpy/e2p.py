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
from typing import TYPE_CHECKING

import numpy as np

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION  # ruff:ignore[unused-import]
from pytools import obj_array

from sumpy.array_context import make_loopy_program
from sumpy.tools import KernelCacheMixin, gather_loopy_arguments


if TYPE_CHECKING:
    from arraycontext import ArrayContext


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
        return lp.tag_array_axes(loopy_knl, "targets", "sep,C")

    def get_loopy_args(self):
        return gather_loopy_arguments((self.expansion, *tuple(self.kernels)))

    def get_kernel_scaling_assignment(self):
        from sumpy.symbolic import to_pymbolic
        return [lp.Assignment(id="kernel_scaling",
                    assignee="kernel_scaling",
                    expression=to_pymbolic(
                        self.expansion.kernel.get_global_scaling_const(),
                    ),
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
        return self.add_loopy_eval_callable(loopy_knl)

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.tag_inames(knl, {"itgt_box": "g.0"})
        knl = lp.add_inames_to_insn(knl, "itgt_box", "id:kernel_scaling")
        return lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

    def __call__(self, actx: ArrayContext, **kwargs):
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

        knl = self.get_cached_kernel()
        result = actx.call_loopy(
            knl,
            centers=centers, rscale=rscale, **kwargs)

        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}


# {{{ E2PFromCSR: E2P from CSR-like interaction list

class E2PFromCSR(E2PBase):
    """Evaluate CSR-listed expansions at each target box's particles.

    :arg work_items_per_group: OpenCL workgroup size, with one group per
        target box and target particles distributed across its workitems.
        *None* uses 128 workitems when ``max_ntargets_in_one_box``
        is supplied to ``__call__``, and selects the serial schedule
        otherwise.
    """

    # M2P (List 3) target-box parallelization:
    # Each (leaf) target box is a workgroup with W workitems.
    # Workitem j handles target offsets j, j + W, j + 2*W, ... within its box.
    def __init__(self, expansion, kernels, name=None, *, work_items_per_group=None):
        super().__init__(expansion, kernels, name)
        self.work_items_per_group = work_items_per_group

    def get_cache_key(self):
        return (*super().get_cache_key(), self.work_items_per_group)

    @property
    def default_name(self):
        return "e2p_from_csr"

    def get_kernel(self):
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        loopy_knl = make_loopy_program(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[itgt_offset]: 0<=itgt_offset<itgt_count}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_end }",
                    "{[idim]: 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                ], [
                *self.get_kernel_scaling_assignment(),
                """
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]  {id=fetch_tgt_box_0}
                    <> itgt_start = box_target_starts[tgt_ibox]  {id=fetch_tgt_box_1}
                    <> itgt_count = box_target_counts_nonchild[tgt_ibox] \
                        {id=fetch_tgt_box_2}

                    for itgt_offset
                        <> itgt = itgt_start + itgt_offset
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
        loopy_knl = lp.prioritize_loops(loopy_knl, "itgt_box,itgt_offset,isrc_box")
        loopy_knl = self.add_loopy_eval_callable(loopy_knl)
        return lp.tag_array_axes(loopy_knl, "targets", "sep,C")

    def get_shared_coefficient_kernel(self, max_ntargets_in_one_box):
        """
        - One workgroup handles one target box with W workitems.
          For each source box, the workgroup loads the coefficients
          into local memory once and shares them across its workitems. Loopy
          inserts barriers to ensure the load finishes before evaluation and
          all workitems finish using the coefficients before they are
          overwritten for the next source box.

        - Each round covers W target slots: in round ``itgt_outer``, workitem
          ``iwork_item`` handles offset ``W*itgt_outer + iwork_item`` within the
          box. Each workitem keeps separate sums for each round, accumulating
          contributions from all source boxes. After the last source box, it
          adds those sums to the global result.

        - All boxes run ``ntgt_outer`` = ceil(*max_ntargets_in_one_box* / W)
          rounds. Slots beyond a box's actual target count are padding: guards
          skip their target reads, evaluations, and result writes.
          A zero maximum uses one guarded round for temporary-array sizing.

        *max_ntargets_in_one_box* must be at least the target count of every box,
        or the targets beyond it are skipped. The summation order per target is
        that of :meth:`get_kernel`, so the results agree exactly.

        Adapted from Isuru Fernando's https://github.com/inducer/sumpy/pull/180.
        """
        ncoeffs = len(self.expansion)
        work_items_per_group = self.work_items_per_group
        if work_items_per_group is None:
            work_items_per_group = 128
        loopy_args = self.get_loopy_args()

        loopy_knl = make_loopy_program(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[iwork_item]: 0<=iwork_item<work_items_per_group}",
                    "{[itgt_outer]: 0<=itgt_outer<ntgt_outer}",
                    "{[isrc_box]: isrc_box_start<=isrc_box<isrc_box_end }",
                    "{[idim]: 0<=idim<dim}",
                    "{[icoeff]: 0<=icoeff<ncoeffs}",
                    "{[iknl]: 0<=iknl<nresults}",
                ], [
                *self.get_kernel_scaling_assignment(),
                """
                for itgt_box
                    for iwork_item
                        <> tgt_ibox = target_boxes[itgt_box]  {id=fetch_tgt_box_0}
                        <> itgt_start = box_target_starts[tgt_ibox] \
                            {id=fetch_tgt_box_1}
                        <> itgt_end = itgt_start \
                            + box_target_counts_nonchild[tgt_ibox] \
                            {id=fetch_tgt_box_2}
                        <> isrc_box_start = source_box_starts[itgt_box] \
                            {id=fetch_src_list_start}
                        <> isrc_box_end = source_box_starts[itgt_box+1] \
                            {id=fetch_src_list_end}
                        <> result_temp[itgt_outer, iknl] = 0 \
                            {id=init_result,dup=itgt_outer:iknl}
                    end

                    for isrc_box
                        for iwork_item
                            <> src_ibox = source_box_lists[isrc_box] \
                                {id=fetch_src_box}
                            <> center[idim] = centers[idim, src_ibox] \
                                {dup=idim,id=fetch_center}
                        end
                        # Loaded the coefficients into local memory once
                        # for all workitems in the group.
                        <> coeffs[icoeff] = \
                            src_expansions[src_ibox - src_base_ibox, icoeff] \
                            {id=fetch_coeffs}

                        for iwork_item, itgt_outer
                            <> itgt = itgt_start + work_items_per_group*itgt_outer \
                                + iwork_item
                            <> cond_itgt = itgt < itgt_end
                            <> tgt[idim] = targets[idim,itgt]  \
                                {id=fetch_tgt,dup=idim,if=cond_itgt}

                            [iknl]: result_temp[itgt_outer, iknl] = e2p(
                                [iknl]: result_temp[itgt_outer, iknl],
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
                              if=cond_itgt}
                        end
                    end

                    # Workitems in the group write their results to global memory.
                    for iwork_item, itgt_outer
                        <> itgt_write = itgt_start \
                            + work_items_per_group*itgt_outer + iwork_item \
                            {id=init_itgt_for_write}
                        <> cond_itgt_write = itgt_write < itgt_end \
                            {id=init_cond_for_write}
                        result[iknl, itgt_write] = result[iknl, itgt_write] \
                            + result_temp[itgt_outer, iknl] * kernel_scaling \
                            {dep=update_result:init_result,id=write_result, \
                             dup=iknl,if=cond_itgt_write}
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
                        "work_items_per_group": work_items_per_group,
                        "ntgt_outer": max(1, -(
                            -max_ntargets_in_one_box // work_items_per_group)),
                        "nresults": len(self.kernels)})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "iknl*:unr")
        knl = self.add_loopy_eval_callable(loopy_knl)

        # When writing to the global result, these three instructions must run
        # in one loop over the same iname. Duplicate itgt_outer for all three
        # together to keep this loop separate from the evaluation loop.
        knl = lp.duplicate_inames(knl, "itgt_outer",
            "id:write_result or id:init_itgt_for_write or id:init_cond_for_write")
        knl = lp.tag_inames(knl, {"itgt_box": "g.0", "iwork_item": "l.0"})

        # The coefficient load is spread over the workitems.
        knl = lp.split_iname(knl, "icoeff", work_items_per_group,
            outer_iname="icoeff_outer", inner_iname="icoeff_inner",
            inner_tag="l.0")
        knl = lp.set_temporary_address_space(
            knl, "coeffs", lp.AddressSpace.LOCAL)

        knl = lp.add_inames_to_insn(knl, "itgt_box,iwork_item",
            "id:kernel_scaling")

        return lp.set_options(knl, enforce_variable_access_ordered="no_check")

    def get_optimized_kernel(self, max_ntargets_in_one_box=None):
        if max_ntargets_in_one_box is not None:
            return self.get_shared_coefficient_kernel(max_ntargets_in_one_box)

        knl = self.get_kernel()
        if self.work_items_per_group is not None:
            import namedisl as nisl

            # Target-parallel E2P scheduling follows the idea in Isuru Fernando's
            # https://github.com/inducer/sumpy/pull/180.
            # Unlike #180, the coefficients are not loaded into local memory:
            # each workitem reads every source box's coefficients itself and
            # accumulates the sum over source boxes.

            knl = lp.split_iname(knl, "itgt_offset", self.work_items_per_group,
                outer_iname="itgt_outer", inner_iname="iwork_item", inner_tag="l.0")

            # Loopy does not allow a parallel iname (iwork_item) in a domain
            # whose bound depends on data (itgt_count), so we split the domain.
            entry = knl.default_entrypoint
            idom = entry.get_home_domain_index("iwork_item")
            domains = list(entry.domains)
            domains[idom:idom + 1] = [
                nisl.make_set(
                    f"{{[iwork_item]: "
                    f"0<=iwork_item<{self.work_items_per_group}}}"),
                nisl.make_set(
                    "[iwork_item, itgt_count] -> {[itgt_outer]: "
                    "0 <= itgt_outer and "
                    f"{self.work_items_per_group}*itgt_outer"
                    " + iwork_item < itgt_count}"),
            ]
            knl = knl.with_kernel(entry.copy(domains=domains))

        knl = lp.tag_inames(knl, {"itgt_box": "g.0"})
        knl = lp.add_inames_to_insn(knl, "itgt_box", "id:kernel_scaling")
        if self.work_items_per_group is not None:
            knl = lp.add_inames_to_insn(knl, "iwork_item",
                    "id:fetch_tgt_box_* or id:kernel_scaling")
        return lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

    def __call__(self, actx: ArrayContext, **kwargs):
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        rscale = centers.dtype.type(kwargs.pop("rscale"))
        max_ntargets_in_one_box = kwargs.pop("max_ntargets_in_one_box", None)

        if max_ntargets_in_one_box is None:
            knl = self.get_cached_kernel()
        else:
            knl = self.get_cached_kernel(
                max_ntargets_in_one_box=int(max_ntargets_in_one_box))
        result = actx.call_loopy(
            knl,
            centers=centers,
            rscale=rscale,
            **kwargs)

        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}

# vim: foldmethod=marker
