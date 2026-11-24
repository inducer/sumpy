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

import logging
from typing import TYPE_CHECKING

import numpy as np

import loopy as lp

from sumpy.array_context import make_loopy_program
from sumpy.tools import KernelCacheMixin, KernelComputation


if TYPE_CHECKING:
    from arraycontext import ArrayContext


logger = logging.getLogger(__name__)


__doc__ = """

Particle-to-Expansion
---------------------

.. autoclass:: P2EBase
.. autoclass:: P2EFromSingleBox
.. autoclass:: P2EFromCSR
"""


# {{{ P2EBase: base class

class P2EBase(KernelCacheMixin, KernelComputation):
    """Common input processing for kernel computations.

    .. automethod:: __init__
    """

    def __init__(self, expansion, kernels=None, name=None, strength_usage=None,
                 *, work_items_per_group: int | None = None):
        """
        :arg expansion: a subclass of :class:`sumpy.expansion.ExpansionBase`
        :arg kernels: if not provided, the kernel of the *expansion* is used.
            The base kernel (after source or target transformation
            removal) of each kernel in the list should match the base kernel of
            the expansion.
        :arg strength_usage: a list of integers indicating which expression
            uses which source strength indicator. This implicitly specifies the
            number of strength arrays that need to be passed in.
            By default all kernels use the same strength.
        :arg work_items_per_group: OpenCL work-group size, with one group per
            expansion and source particles distributed across its work items.
            *None* selects the default schedule. Must fit the device's work-group
            limit and local memory (one coefficient vector per work item).
        """
        from sumpy.kernel import (
            SourceTransformationRemover,
            TargetTransformationRemover,
        )
        txr = TargetTransformationRemover()
        sxr = SourceTransformationRemover()

        kernels = [txr(expansion.kernel)] if kernels is None else kernels
        expansion = expansion.with_kernel(sxr(txr(expansion.kernel)))

        for knl in kernels:
            assert txr(knl) == knl
            assert sxr(knl) == expansion.kernel

        KernelComputation.__init__(self, target_kernels=[],
            source_kernels=kernels,
            strength_usage=strength_usage, value_dtypes=None,
            name=name)

        self.expansion = expansion
        self.dim = expansion.dim
        self.work_items_per_group = work_items_per_group

    def add_loopy_form_callable(
            self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        inner_knl = self.expansion.loopy_expansion_formation(
            self.source_kernels, self.strength_usage, self.strength_count)
        loopy_knl = lp.merge([loopy_knl, inner_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "p2e")
        loopy_knl = lp.remove_unused_inames(loopy_knl)
        for kernel in self.source_kernels:
            loopy_knl = kernel.prepare_loopy_kernel(loopy_knl)
        return lp.tag_array_axes(loopy_knl, "strengths", "sep,C")

    def get_loopy_args(self):
        from sumpy.tools import gather_loopy_source_arguments
        return gather_loopy_source_arguments(
                (self.expansion, *tuple(self.source_kernels)))

    def get_cache_key(self):
        return (type(self).__name__, self.name, self.expansion,
                tuple(self.source_kernels), tuple(self.strength_usage),
                self.work_items_per_group)

    def get_optimized_kernel(self, sources_is_obj_array, centers_is_obj_array):
        if self.work_items_per_group is None:
            knl = self.get_kernel()
        else:
            knl = self.get_kernel(work_items_per_group=self.work_items_per_group)

        if sources_is_obj_array:
            knl = lp.tag_array_axes(knl, "sources", "sep,C")
        if centers_is_obj_array:
            knl = lp.tag_array_axes(knl, "centers", "sep,C")

        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        return lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

    def _get_source_parallel_kernel(
            self, work_items_per_group: int, *, from_csr: bool) -> lp.TranslationUnit:
        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        if from_csr:
            box_iname, box_count = "itgt_box", "ntgt_boxes"
            box_ibox = "tgt_ibox"
            nboxes, aligned_nboxes = "ntgt_level_boxes", "naligned_boxes"
            box_setup = """
                <> tgt_ibox = target_boxes[itgt_box]
                <> isrc_box_start = source_box_starts[itgt_box]
                <> isrc_box_stop = source_box_starts[itgt_box + 1]
                """
            source_loop = """
                for isrc_box
                    <> src_ibox = source_box_lists[isrc_box]
                    <> isrc_start = box_source_starts[src_ibox]
                    <> nsrc_in_box = box_source_counts_nonchild[src_ibox]
                """
            source_loop_end = "end"
        else:
            box_iname, box_count = "isrc_box", "nsrc_boxes"
            box_ibox = "src_ibox"
            nboxes, aligned_nboxes = "nboxes", "aligned_nboxes"
            box_setup = """
                <> src_ibox = source_boxes[isrc_box]
                <> isrc_start = box_source_starts[src_ibox]
                <> nsrc_in_box = box_source_counts_nonchild[src_ibox]
                """
            source_loop = source_loop_end = ""

        domains = [f"{{[{box_iname}]: 0 <= {box_iname} < {box_count}}}"]
        csr_args = []
        if from_csr:
            domains.append(
                "{[isrc_box]: isrc_box_start <= isrc_box < isrc_box_stop}")
            csr_args.append(lp.GlobalArg(
                "source_box_starts,source_box_lists",
                None, shape=None, offset=lp.auto))

        domains.extend([
            "{[iwork_item]: 0 <= iwork_item < work_items_per_group}",
            (f"{{[isrc_outer]: 0 <= isrc_outer and "
             f"iwork_item + {work_items_per_group}*isrc_outer < nsrc_in_box}}"),
            "{[idim]: 0 <= idim < dim}",
            "{[icoeff]: 0 <= icoeff < ncoeffs}",
            "{[istrength]: 0 <= istrength < nstrengths}",
            "{[ireduce]: 0 <= ireduce < work_items_per_group}",
            (f"{{[icoeff_outer]: 0 <= icoeff_outer and "
             f"iwork_item + {work_items_per_group}*icoeff_outer < ncoeffs}}"),
            ])

        loopy_knl = make_loopy_program(
                domains,
                [f"""
                for {box_iname}
                    {box_setup}
                    for iwork_item
                        <> center[idim] = centers[idim, {box_ibox}] \
                                {{id=fetch_center,dup=idim}}
                        <> work_item_coeffs[icoeff] = 0 \
                                {{id=init_coeffs,dup=icoeff}}
                        {source_loop}
                            for isrc_outer
                                <> isrc = isrc_start + iwork_item \
                                        + work_items_per_group*isrc_outer
                                <> source[idim] = sources[idim, isrc] \
                                        {{id=fetch_src,dup=idim}}
                                <> strength[istrength] = strengths[istrength, isrc] \
                                        {{id=fetch_strength,dup=istrength}}
                                [icoeff]: work_item_coeffs[icoeff] = p2e(
                                        [icoeff]: work_item_coeffs[icoeff],
                                        [idim]: center[idim],
                                        [idim]: source[idim],
                                        [istrength]: strength[istrength],
                                        rscale,
                                        isrc,
                                        nsources,
                                        sources,
                                        {",".join(arg.name for arg in loopy_args)}
                                    ) {{id=update_result, \
                                       dep=fetch_center:fetch_src:init_coeffs}}
                            end
                        {source_loop_end}
                        partial[icoeff, iwork_item] = \
                                work_item_coeffs[icoeff] \
                                {{id=store_partial,dup=icoeff, \
                                  dep=update_result:init_coeffs}}
                    end

                    # After processing all source particles for the target box (P2L)
                    # or source box (P2M), each work item sums partial contributions
                    # from all work items for its assigned coefficients.
                    for iwork_item
                        for icoeff_outer
                            <> icoeff_write = iwork_item \
                                    + work_items_per_group*icoeff_outer
                            tgt_expansions[
                                {box_ibox} - tgt_base_ibox, icoeff_write] = \
                                sum(ireduce, partial[icoeff_write, ireduce]) \
                                {{id=write_expn,dep=store_partial}}
                        end
                    end
                end
                """],
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources"), order="C"),
                    lp.GlobalArg("strengths", None,
                        shape=(self.strength_count, "nsources")),
                    *csr_args,
                    lp.GlobalArg("box_source_starts,box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None,
                        shape=f"dim, {aligned_nboxes}"),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=(nboxes, ncoeffs), offset=lp.auto),
                    lp.TemporaryVariable(
                        "partial", dtype=None, shape=(ncoeffs, work_items_per_group),
                        address_space=lp.AddressSpace.LOCAL),
                    lp.ValueArg(f"{nboxes},{aligned_nboxes},tgt_base_ibox", np.int32),
                    lp.ValueArg("nsources", np.int32),
                    *loopy_args,
                    ...
                    ],
                name=self.name,
                assumptions=f"{box_count}>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={
                    "dim": self.dim,
                    "nstrengths": self.strength_count,
                    "ncoeffs": ncoeffs,
                    "work_items_per_group": work_items_per_group,
                    })

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "istrength*:unr")
        loopy_knl = self.add_loopy_form_callable(loopy_knl)
        return lp.add_barrier(
            loopy_knl,
            "id:store_partial",
            "id:write_expn",
            synchronization_kind="local",
            within_inames=frozenset({box_iname}),
            )

    def __call__(self, actx: ArrayContext, **kwargs):
        from sumpy.tools import is_obj_array_like
        sources = kwargs.pop("sources")
        centers = kwargs.pop("centers")

        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        dtype = centers[0].dtype if is_obj_array_like(centers) else centers.dtype
        rscale = dtype.type(kwargs.pop("rscale"))

        knl = self.get_cached_kernel(
                sources_is_obj_array=is_obj_array_like(sources),
                centers_is_obj_array=is_obj_array_like(centers))

        result = actx.call_loopy(
            knl,
            sources=sources, centers=centers, rscale=rscale,
            **kwargs)

        return result["tgt_expansions"]

# }}}


# {{{ P2EFromSingleBox: P2E from single box (P2M, likely)

class P2EFromSingleBox(P2EBase):
    """
    .. automethod:: __call__
    """

    @property
    def default_name(self):
        return "p2e_from_single_box"

    def get_kernel(self, *, work_items_per_group: int | None = None):
        if work_items_per_group is not None:
            return self._get_source_parallel_kernel(
                work_items_per_group, from_csr=False)

        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        loopy_knl = make_loopy_program([
                "{[isrc_box]: 0 <= isrc_box < nsrc_boxes}",
                "{[isrc]: isrc_start <= isrc < isrc_end}",
                "{[idim]: 0 <= idim < dim}",
                "{[icoeff]: 0 <= icoeff < ncoeffs}",
                "{[istrength]: 0 <= istrength < nstrengths}",
                ], ["""
                for isrc_box
                    <> src_ibox = source_boxes[isrc_box]
                    <> isrc_start = box_source_starts[src_ibox]
                    <> isrc_end = isrc_start + box_source_counts_nonchild[src_ibox]

                    <> center[idim] = centers[idim, src_ibox] {id=fetch_center}

                    <> coeffs[icoeff] = 0  {id=init_coeffs,dup=icoeff}
                    for isrc
                        <> source[idim] = sources[idim, isrc] \
                                {dup=idim,id=fetch_src}
                        <> strength[istrength] = strengths[istrength, isrc] \
                                {dup=istrength,id=fetch_strength}
                        [icoeff]: coeffs[icoeff] = p2e(
                                [icoeff]: coeffs[icoeff],
                                [idim]: center[idim],
                                [idim]: source[idim],
                                [istrength]: strength[istrength],
                                rscale,
                                isrc,
                                nsources,
                                sources,
                """ + ",".join(arg.name for arg in loopy_args) + """
                            )  {id=update_result, \
                              dep=fetch_center:fetch_src:init_coeffs}
                    end
                    tgt_expansions[src_ibox - tgt_base_ibox, icoeff] = \
                        coeffs[icoeff] {id=write_expn,dup=icoeff,\
                        dep=update_result:init_coeffs}
                end
                """],
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources"), order="C"),
                    lp.GlobalArg("strengths", None,
                        shape=(self.strength_count, "nsources")),
                    lp.GlobalArg("box_source_starts, box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("rscale", None),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("nboxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("nboxes, aligned_nboxes, tgt_base_ibox", np.int32),
                    lp.ValueArg("nsources", np.int32),
                    *loopy_args,
                    ...
                ],
                name=self.name,
                assumptions="nsrc_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={
                    "dim": self.dim, "nstrengths": self.strength_count,
                    "ncoeffs": ncoeffs})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "istrength*:unr")
        return self.add_loopy_form_callable(loopy_knl)

    def get_optimized_kernel(self, sources_is_obj_array, centers_is_obj_array):
        knl = super().get_optimized_kernel(
                sources_is_obj_array=sources_is_obj_array,
                centers_is_obj_array=centers_is_obj_array)

        if self.work_items_per_group is not None:
            knl = lp.tag_inames(knl, {
                "isrc_box": "g.0",
                "iwork_item": "l.0",
                })
            return lp.add_inames_for_unused_hw_axes(knl)

        # FIXME
        return lp.split_iname(knl, "isrc_box", 16, outer_tag="g.0")

    def __call__(self, actx: ArrayContext, **kwargs):
        """
        :arg source_boxes: an array of integer indices into *box_source_starts*
            and *box_source_counts_nonchild*.
        :arg box_source_starts: an array of integer indices into *sources*.
        :arg box_source_counts_nonchild: an array of integer sizes of each box.
        :arg centers: expansion centers.
        :arg sources: source points.
        :arg strengths: strengths at each source point. the strength count
            is given by the *strength_usage* list passed in to
            :meth:`P2EBase.__init__`.
        :arg nboxes: number of boxes.
        :arg tgt_base_ibox: integer for the base index of the target box.
        :arg rscale: expansion scale.
        :arg tgt_expansions: if given as an input, the array will be filled
            in with the expansions at boxes indexed by
            ``source_boxes[i] - tgt_base_ibox``.

        :returns: an array of *tgt_expansions*.
        """
        return super().__call__(actx, **kwargs)

# }}}


# {{{ P2EFromCSR: P2E from CSR-like interaction list

class P2EFromCSR(P2EBase):
    """
    .. automethod:: __call__
    """

    @property
    def default_name(self):
        return "p2e_from_csr"

    def get_kernel(self, *, work_items_per_group: int | None = None):
        if work_items_per_group is not None:
            return self._get_source_parallel_kernel(
                work_items_per_group, from_csr=True)

        ncoeffs = len(self.expansion)
        loopy_args = self.get_loopy_args()

        arguments = (
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources"), order="C"),
                    lp.GlobalArg("strengths", None,
                        shape=(self.strength_count, "nsources")),
                    lp.GlobalArg("source_box_starts,source_box_lists",
                        None, shape=None, offset=lp.auto),
                    lp.GlobalArg("box_source_starts,box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("naligned_boxes,ntgt_level_boxes,tgt_base_ibox",
                        np.int32),
                    lp.ValueArg("nsources", np.int32),
                    *loopy_args,
                    ...
                ])

        loopy_knl = make_loopy_program(
                [
                    "{[itgt_box]: 0 <= itgt_box < ntgt_boxes}",
                    "{[isrc_box]: isrc_box_start <= isrc_box < isrc_box_stop}",
                    "{[isrc]: isrc_start <= isrc < isrc_end}",
                    "{[idim]: 0 <= idim < dim}",
                    "{[icoeff]: 0 <= icoeff < ncoeffs}",
                    "{[istrength]: 0 <= istrength < nstrengths}",
                    ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_stop = source_box_starts[itgt_box + 1]

                    <> coeffs[icoeff] = 0  {id=init_coeffs,dup=icoeff}
                    for isrc_box
                        <> src_ibox = source_box_lists[isrc_box]
                        <> isrc_start = box_source_starts[src_ibox]
                        <> isrc_end = isrc_start \
                                + box_source_counts_nonchild[src_ibox]

                        for isrc
                            <> source[idim] = sources[idim, isrc] \
                                    {dup=idim,id=fetch_src}
                            <> strength[istrength] = strengths[istrength, isrc] \
                                    {dup=istrength,id=fetch_strength}
                            [icoeff]: coeffs[icoeff] = p2e(
                                    [icoeff]: coeffs[icoeff],
                                    [idim]: center[idim],
                                    [idim]: source[idim],
                                    [istrength]: strength[istrength],
                                    rscale,
                                    isrc,
                                    nsources,
                                    sources,
                    """ + ",".join(arg.name for arg in loopy_args) + """
                                )  {id=update_result, \
                                  dep=fetch_center:fetch_src:init_coeffs}
                        end
                    end
                    tgt_expansions[tgt_ibox - tgt_base_ibox, icoeff] = \
                            coeffs[icoeff] {id=write_expn,dup=icoeff, \
                            dep=update_result:init_coeffs}
                end
                """],
                kernel_data=arguments,
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={"dim": self.dim,
                                  "nstrengths": self.strength_count,
                                  "ncoeffs": ncoeffs})

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "istrength*:unr")
        return self.add_loopy_form_callable(loopy_knl)

    def get_optimized_kernel(self, sources_is_obj_array, centers_is_obj_array):
        knl = super().get_optimized_kernel(
                sources_is_obj_array=sources_is_obj_array,
                centers_is_obj_array=centers_is_obj_array)

        if self.work_items_per_group is not None:
            knl = lp.tag_inames(knl, {
                "itgt_box": "g.0",
                "iwork_item": "l.0",
                })
            return lp.add_inames_for_unused_hw_axes(knl)

        # FIXME
        return lp.split_iname(knl, "itgt_box", 16, outer_tag="g.0")

    def __call__(self, actx: ArrayContext, **kwargs):
        """
        :arg target_boxes: array of integer indices into *source_box_starts*
            and *centers*.
        :arg source_box_starts: array of integer indices into *source_box_lists*,
            i.e. `source_box_starts[i]:source_box_starts[i + 1]` gives all the
            source boxes for a given target box ``i = target_boxes[itgt_box]``.
        :arg source_box_lists: an array of integer indices into *box_source_starts*
            and *box_source_counts_nonchild*.

        :arg box_source_starts: see :meth:`P2EFromSingleBox.__call__`.
        :arg box_source_counts_nonchild: see :meth:`P2EFromSingleBox.__call__`.
        :arg centers: see :meth:`P2EFromSingleBox.__call__`.
        :arg sources: see :meth:`P2EFromSingleBox.__call__`.
        :arg strengths: see :meth:`P2EFromSingleBox.__call__`.
        :arg rscale: see :meth:`P2EFromSingleBox.__call__`.
        :arg tgt_base_ibox: see :meth:`P2EFromSingleBox.__call__`.
        :arg tgt_expansion: see :meth:`P2EFromSingleBox.__call__`.
        """
        return super().__call__(actx, **kwargs)

# }}}

# vim: foldmethod=marker
