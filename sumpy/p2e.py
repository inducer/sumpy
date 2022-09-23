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
from loopy.version import MOST_RECENT_LANGUAGE_VERSION

from sumpy.tools import KernelCacheMixin, KernelComputation

import logging
logger = logging.getLogger(__name__)


__doc__ = """

Particle-to-Expansion
---------------------

.. autoclass:: P2EBase
.. autoclass:: P2EFromSingleBox
.. autoclass:: P2EFromCSR
"""


# {{{ P2E base class

class P2EBase(KernelCacheMixin, KernelComputation):
    """Common input processing for kernel computations.

    .. automethod:: __init__
    """

    def __init__(self, ctx, expansion, kernels=None,
            name=None, device=None, strength_usage=None):
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
        """
        from sumpy.kernel import (TargetTransformationRemover,
                SourceTransformationRemover)
        txr = TargetTransformationRemover()
        sxr = SourceTransformationRemover()

        if kernels is None:
            kernels = [txr(expansion.kernel)]
        else:
            kernels = kernels

        expansion = expansion.with_kernel(sxr(txr(expansion.kernel)))

        for knl in kernels:
            assert txr(knl) == knl
            assert sxr(knl) == expansion.kernel

        KernelComputation.__init__(self, ctx=ctx, target_kernels=[],
            source_kernels=kernels,
            strength_usage=strength_usage, value_dtypes=None,
            name=name, device=device)

        self.expansion = expansion
        self.dim = expansion.dim

    def get_loopy_instructions(self):
        from sumpy.symbolic import make_sym_vector
        avec = make_sym_vector("a", self.dim)

        import sumpy.symbolic as sp
        rscale = sp.Symbol("rscale")

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        strengths = [sp.Symbol(f"strength_{i}") for i in self.strength_usage]
        coeffs = self.expansion.coefficients_from_source_vec(self.source_kernels,
                    avec, None, rscale, strengths, sac=sac)

        coeff_names = []
        for i, coeff in enumerate(coeffs):
            sac.add_assignment(f"coeff{i}", coeff)
            coeff_names.append(f"coeff{i}")

        sac.run_global_cse()

        code_transformers = [self.expansion.get_code_transformer()] \
            + [kernel.get_code_transformer() for kernel in self.source_kernels]

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                sac.assignments.items(),
                vector_names={"a"},
                pymbolic_expr_maps=code_transformers,
                retain_names=coeff_names,
                )

    def get_cache_key(self):
        return (type(self).__name__, self.name, self.expansion,
                tuple(self.source_kernels), tuple(self.strength_usage))

    def get_optimized_kernel(self, sources_is_obj_array, centers_is_obj_array):
        knl = self.get_kernel()

        if sources_is_obj_array:
            knl = lp.tag_array_axes(knl, "sources", "sep,C")
        if centers_is_obj_array:
            knl = lp.tag_array_axes(knl, "centers", "sep,C")

        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")
        return knl

    def __call__(self, queue, **kwargs):
        from sumpy.tools import is_obj_array_like
        sources = kwargs.pop("sources")
        centers = kwargs.pop("centers")
        knl = self.get_cached_optimized_kernel(
                sources_is_obj_array=is_obj_array_like(sources),
                centers_is_obj_array=is_obj_array_like(centers))

        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        dtype = centers[0].dtype if is_obj_array_like(centers) else centers.dtype
        rscale = dtype.type(kwargs.pop("rscale"))

        return knl(queue, sources=sources, centers=centers, rscale=rscale, **kwargs)

# }}}


# {{{ P2E from single box (P2M, likely)

class P2EFromSingleBox(P2EBase):
    """
    .. automethod:: __call__
    """

    @property
    def default_name(self):
        return "p2e_from_single_box"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        from sumpy.tools import gather_loopy_source_arguments
        loopy_knl = lp.make_kernel([
                "{[isrc_box]: 0 <= isrc_box < nsrc_boxes}",
                "{[isrc, idim]: isrc_start <= isrc < isrc_end and 0 <= idim < dim}",
                ], ["""
                for isrc_box
                    <> src_ibox = source_boxes[isrc_box]
                    <> isrc_start = box_source_starts[src_ibox]
                    <> isrc_end = isrc_start + box_source_counts_nonchild[src_ibox]

                    <> center[idim] = centers[idim, src_ibox] {id=fetch_center}

                    for isrc
                        <> a[idim] = center[idim] - sources[idim, isrc] {dup=idim}
                        """] + [
                        f"<> strength_{i} = strengths[{i}, isrc]"
                        for i in set(self.strength_usage)
                        ] + self.get_loopy_instructions() + ["""
                    end
                    """] + [f"""
                    tgt_expansions[src_ibox - tgt_base_ibox, {coeffidx}] = \
                        simul_reduce(sum, isrc, coeff{coeffidx}) \
                            {{id_prefix=write_expn}}
                    """ for coeffidx in range(ncoeffs)] + ["""
                end
                """],
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources"), order="C"),
                    lp.GlobalArg("strengths", None,
                        shape=("strength_count", "nsources"), dim_tags="sep,C"),
                    lp.GlobalArg("box_source_starts, box_source_counts_nonchild",
                        None, shape=None),
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("rscale", None),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("nboxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("nboxes, aligned_nboxes, tgt_base_ibox", np.int32),
                    lp.ValueArg("nsources", np.int32),
                    ...
                ] + gather_loopy_source_arguments(
                    self.source_kernels + (self.expansion,)),
                name=self.name,
                assumptions="nsrc_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                default_offset=lp.auto,
                fixed_parameters=dict(dim=self.dim,
                    strength_count=self.strength_count),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        for knl in self.source_kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    def get_optimized_kernel(self, sources_is_obj_array, centers_is_obj_array):
        knl = super().get_optimized_kernel(
                sources_is_obj_array=sources_is_obj_array,
                centers_is_obj_array=centers_is_obj_array)

        # FIXME
        knl = lp.split_iname(knl, "isrc_box", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
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
        return super().__call__(queue, **kwargs)

# }}}


# {{{ P2E from CSR-like interaction list

class P2EFromCSR(P2EBase):
    """
    .. automethod:: __call__
    """

    @property
    def default_name(self):
        return "p2e_from_csr"

    def get_kernel(self):
        ncoeffs = len(self.expansion)

        from sumpy.tools import gather_loopy_source_arguments
        arguments = (
                [
                    lp.GlobalArg("sources", None,
                        shape=(self.dim, "nsources"), order="C"),
                    lp.GlobalArg("strengths", None,
                        shape=("strength_count", "nsources"), dim_tags="sep,C"),
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
                    ...
                ] + gather_loopy_source_arguments(
                    self.source_kernels + (self.expansion,)))

        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0 <= itgt_box < ntgt_boxes}",
                    "{[isrc_box]: isrc_box_start <= isrc_box < isrc_box_stop}",
                    "{[isrc]: isrc_start <= isrc < isrc_end}",
                    "{[idim]: 0 <= idim < dim}",
                    ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> center[idim] = centers[idim, tgt_ibox] {id=fetch_center}

                    <> isrc_box_start = source_box_starts[itgt_box]
                    <> isrc_box_stop = source_box_starts[itgt_box + 1]

                    for isrc_box
                        <> src_ibox = source_box_lists[isrc_box]
                        <> isrc_start = box_source_starts[src_ibox]
                        <> isrc_end = isrc_start \
                                + box_source_counts_nonchild[src_ibox]

                        for isrc
                            <> a[idim] = center[idim] - sources[idim, isrc] \
                                    {dup=idim}
                    """] + [
                        f"""
                             <> strength_{i} = strengths[{i}, isrc]
                             """ for i in set(self.strength_usage)
                             ] + self.get_loopy_instructions() + ["""
                        end
                    end"""] + [f"""
                    tgt_expansions[tgt_ibox - tgt_base_ibox, {coeffidx}] = \
                            simul_reduce(sum, (isrc_box, isrc), coeff{coeffidx}) \
                            {{id_prefix=write_expn}}
                    """ for coeffidx in range(ncoeffs)] + ["""
                end
                """],
                arguments,
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                default_offset=lp.auto,
                fixed_parameters=dict(dim=self.dim,
                    strength_count=self.strength_count),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        for knl in self.source_kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)
        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    def get_optimized_kernel(self, sources_is_obj_array, centers_is_obj_array):
        knl = super().get_optimized_kernel(
                sources_is_obj_array=sources_is_obj_array,
                centers_is_obj_array=centers_is_obj_array)

        # FIXME
        knl = lp.split_iname(knl, "itgt_box", 16, outer_tag="g.0")
        return knl

    def __call__(self, queue, **kwargs):
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
        return super().__call__(queue, **kwargs)

# }}}

# vim: foldmethod=marker
