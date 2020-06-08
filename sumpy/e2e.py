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

from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from sumpy.tools import KernelCacheWrapper

import logging
logger = logging.getLogger(__name__)


__doc__ = """

Expansion-to-expansion
----------------------

.. autoclass:: E2EBase
.. autoclass:: E2EFromCSR
.. autoclass:: E2EFromParent
.. autoclass:: E2EFromChildren

"""


# {{{ translation base class

class E2EBase(KernelCacheWrapper):
    def __init__(self, ctx, src_expansion, tgt_expansion,
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

        if src_expansion is tgt_expansion:
            from sumpy.kernel import TargetDerivativeRemover, SourceDerivativeRemover
            tgt_expansion = src_expansion = src_expansion.with_kernel(
                    SourceDerivativeRemover()(
                        TargetDerivativeRemover()(src_expansion.kernel)))

        else:

            from sumpy.kernel import TargetDerivativeRemover, SourceDerivativeRemover
            src_expansion = src_expansion.with_kernel(
                    SourceDerivativeRemover()(
                        TargetDerivativeRemover()(src_expansion.kernel)))
            tgt_expansion = tgt_expansion.with_kernel(
                    SourceDerivativeRemover()(
                        TargetDerivativeRemover()(tgt_expansion.kernel)))

        self.ctx = ctx
        self.src_expansion = src_expansion
        self.tgt_expansion = tgt_expansion
        self.options = options
        self.name = name or self.default_name
        self.device = device

        if src_expansion.dim != tgt_expansion.dim:
            raise ValueError("source and target expansions must have "
                    "same dimensionality")

        self.dim = src_expansion.dim

    def get_translation_loopy_insns(self):
        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("d", self.dim)

        src_coeff_exprs = [sym.Symbol("src_coeff%d" % i)
                for i in range(len(self.src_expansion))]
        src_rscale = sym.Symbol("src_rscale")

        tgt_rscale = sym.Symbol("tgt_rscale")

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique("coeff%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, src_rscale,
                        dvec, tgt_rscale, sac))]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                six.iteritems(sac.assignments),
                vector_names=set(["d"]),
                pymbolic_expr_maps=[self.tgt_expansion.get_code_transformer()],
                retain_names=tgt_coeff_names,
                complex_dtype=np.complex128  # FIXME
                )

    def get_cache_key(self):
        return (
                type(self).__name__,
                self.src_expansion,
                self.tgt_expansion)

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt_box", 16, outer_tag="g.0")

        return knl

# }}}


# {{{ translation from "compressed sparse row"-like source box lists

class E2EFromCSR(E2EBase):
    """Implements translation from a "compressed sparse row"-like source box
    list.
    """

    default_name = "e2e_from_csr"

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)

        # To clarify terminology:
        #
        # isrc_box -> The index in a list of (in this case, source) boxes
        # src_ibox -> The (global) box number for the (in this case, source) box
        #
        # (same for itgt_box, tgt_ibox)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]

                    <> tgt_center[idim] = centers[idim, tgt_ibox] \

                    <> isrc_start = src_box_starts[itgt_box]
                    <> isrc_stop = src_box_starts[itgt_box+1]

                    for isrc_box
                        <> src_ibox = src_box_lists[isrc_box] \
                                {id=read_src_ibox}

                        <> src_center[idim] = centers[idim, src_ibox] {dup=idim}
                        <> d[idim] = tgt_center[idim] - src_center[idim] \
                            {dup=idim}

                        """] + ["""
                        <> src_coeff{coeffidx} = \
                            src_expansions[src_ibox - src_base_ibox, {coeffidx}] \
                            {{dep=read_src_ibox}}
                        """.format(coeffidx=i) for i in range(ncoeff_src)] + [

                        ] + self.get_translation_loopy_insns() + ["""
                    end

                    """] + ["""
                    tgt_expansions[tgt_ibox - tgt_base_ibox, {coeffidx}] = \
                            simul_reduce(sum, isrc_box, coeff{coeffidx}) \
                            {{id_prefix=write_expn}}
                    """.format(coeffidx=i) for i in range(ncoeff_tgt)] + ["""
                end
                """],
                [
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("src_rscale,tgt_rscale", None),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,), offset=lp.auto),
                    lp.ValueArg("aligned_nboxes,tgt_base_ibox,src_base_ibox",
                        np.int32),
                    lp.ValueArg("nsrc_level_boxes,ntgt_level_boxes",
                        np.int32),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeff_src), offset=lp.auto),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeff_tgt), offset=lp.auto),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                default_offset=lp.auto,
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION
                )

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        knl = self.get_cached_optimized_kernel()

        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))

        return knl(queue,
                centers=centers,
                src_rscale=src_rscale, tgt_rscale=tgt_rscale,
                **kwargs)

# }}}


# {{{

class E2EFromCSRTranslationInvariant(E2EFromCSR):
    """Implements translation from a "compressed sparse row"-like source box
    list for translation invariant Kernels.
    """
    default_name = "e2e_from_csr_translation_invariant"

    def get_translation_loopy_insns(self):
        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("d", self.dim)

        src_coeff_exprs = [sym.Symbol("src_coeff%d" % i)
                for i in range(len(self.src_expansion))]
        src_rscale = sym.Symbol("src_rscale")

        tgt_rscale = sym.Symbol("tgt_rscale")

        nprecomputed_exprs = \
            self.tgt_expansion.m2l_global_precompute_nexpr(self.src_expansion)

        precomputed_exprs = [sym.Symbol("precomputed_expr%d" % i)
                for i in range(nprecomputed_exprs)]

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique("coeff%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, src_rscale,
                        dvec, tgt_rscale, sac,
                        precomputed_exprs=precomputed_exprs))]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                six.iteritems(sac.assignments),
                vector_names=set(["d"]),
                pymbolic_expr_maps=[self.tgt_expansion.get_code_transformer()],
                retain_names=tgt_coeff_names,
                complex_dtype=np.complex128  # FIXME
                )

    def get_kernel(self):
        ncoeff_src = len(self.src_expansion)
        ncoeff_tgt = len(self.tgt_expansion)
        nprecomputed_exprs = \
            self.tgt_expansion.m2l_global_precompute_nexpr(self.src_expansion)

        # To clarify terminology:
        #
        # isrc_box -> The index in a list of (in this case, source) boxes
        # src_ibox -> The (global) box number for the (in this case, source) box
        #
        # (same for itgt_box, tgt_ibox)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]

                    <> tgt_center[idim] = centers[idim, tgt_ibox] \

                    <> isrc_start = src_box_starts[itgt_box]
                    <> isrc_stop = src_box_starts[itgt_box+1]

                    for isrc_box
                        <> src_ibox = src_box_lists[isrc_box] \
                                {id=read_src_ibox}

                        <> src_center[idim] = centers[idim, src_ibox] {dup=idim}
                        <> d[idim] = tgt_center[idim] - src_center[idim] \
                            {dup=idim}
                        <> translation_class = \
                                m2l_translation_classes_lists[isrc_box]
                        <> translation_class_rel = translation_class - \
                                                    translation_classes_level_start
                        """] + ["""
                        <> precomputed_expr{idx} = \
                            m2l_precomputed_exprs[translation_class_rel, {idx}]
                        """.format(idx=idx) for idx in range(
                            nprecomputed_exprs)] + ["""
                        <> src_coeff{coeffidx} = \
                            src_expansions[src_ibox - src_base_ibox, {coeffidx}] \
                            {{dep=read_src_ibox}}
                        """.format(coeffidx=i) for i in range(ncoeff_src)] + [

                        ] + self.get_translation_loopy_insns() + ["""
                    end

                    """] + ["""
                    tgt_expansions[tgt_ibox - tgt_base_ibox, {coeffidx}] = \
                            simul_reduce(sum, isrc_box, coeff{coeffidx}) \
                            {{id_prefix=write_expn}}
                    """.format(coeffidx=i) for i in range(ncoeff_tgt)] + ["""
                end
                """],
                [
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("src_rscale,tgt_rscale", None),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,), offset=lp.auto),
                    lp.ValueArg("aligned_nboxes,tgt_base_ibox,src_base_ibox",
                        np.int32),
                    lp.ValueArg("nsrc_level_boxes,ntgt_level_boxes",
                        np.int32),
                    lp.ValueArg("translation_classes_level_start",
                        np.int32),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeff_src), offset=lp.auto),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeff_tgt), offset=lp.auto),
                    lp.ValueArg("ntranslation_classes, ntranslation_classes_lists",
                        np.int32),
                    lp.GlobalArg("m2l_translation_classes_lists", np.int32,
                        shape=("ntranslation_classes_lists"), strides=(1,),
                        offset=lp.auto),
                    lp.GlobalArg("m2l_precomputed_exprs", None,
                        shape=("ntranslation_classes", nprecomputed_exprs),
                        offset=lp.auto),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                default_offset=lp.auto,
                fixed_parameters=dict(dim=self.dim,
                                      nprecomputed_exprs=nprecomputed_exprs),
                lang_version=MOST_RECENT_LANGUAGE_VERSION
                )

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl


class E2EFromCSRTranslationClassesPrecompute(E2EFromCSR):
    """Implements precomputing the translation classes dependent
    derivatives.
    """
    default_name = "e2e_from_csr_translation_classes_precompute"

    def get_translation_loopy_insns(self):
        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("d", self.dim)

        src_rscale = sym.Symbol("src_rscale")
        tgt_rscale = sym.Symbol("tgt_rscale")

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique("precomputed_expr%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.m2l_global_precompute_exprs(
                        self.src_expansion, src_rscale,
                        dvec, tgt_rscale, sac))]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                six.iteritems(sac.assignments),
                vector_names=set(["d"]),
                pymbolic_expr_maps=[self.tgt_expansion.get_code_transformer()],
                retain_names=tgt_coeff_names,
                complex_dtype=np.complex128  # FIXME
                )

    def get_kernel(self):
        nprecomputed_exprs = \
            self.tgt_expansion.m2l_global_precompute_nexpr(self.src_expansion)
        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[itr_class]: 0<=itr_class<ntranslation_classes}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for itr_class
                    <> d[idim] = m2l_translation_vectors[idim, \
                            itr_class + translation_classes_level_start] {dup=idim}

                    """] + self.get_translation_loopy_insns() + ["""
                    m2l_precomputed_exprs[itr_class, {idx}] = precomputed_expr{idx}
                    """.format(idx=i) for i in range(nprecomputed_exprs)] + ["""
                end
                """],
                [
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("m2l_precomputed_exprs", None,
                        shape=("ntranslation_classes", nprecomputed_exprs),
                        offset=lp.auto),
                    lp.GlobalArg("m2l_translation_vectors", None,
                        shape=("dim", "ntranslation_vectors")),
                    lp.ValueArg("ntranslation_classes", np.int32),
                    lp.ValueArg("ntranslation_vectors", np.int32),
                    lp.ValueArg("translation_classes_level_start", np.int32),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntranslation_classes>=1",
                default_offset=lp.auto,
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION
                )

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itr_class", 16, outer_tag="g.0")

        return knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_rscale:
        :arg translation_classes_level_start:
        :arg ntranslation_classes:
        :arg m2l_precomputed_exprs:
        :arg m2l_translation_vectors:
        """
        knl = self.get_cached_optimized_kernel()

        m2l_translation_vectors = kwargs.pop("m2l_translation_vectors")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = m2l_translation_vectors.dtype.type(kwargs.pop("src_rscale"))

        return knl(queue,
                src_rscale=src_rscale,
                m2l_translation_vectors=m2l_translation_vectors,
                **kwargs)

# }}}


# {{{ translation from a box's children

class E2EFromChildren(E2EBase):
    default_name = "e2e_from_children"

    def get_kernel(self):
        if self.src_expansion is not self.tgt_expansion:
            raise RuntimeError("%s requires that the source "
                    "and target expansion are the same object"
                    % type(self).__name__)

        ncoeffs = len(self.src_expansion)

        # To clarify terminology:
        #
        # isrc_box -> The index in a list of (in this case, source) boxes
        # src_ibox -> The (global) box number for the (in this case, source) box
        #
        # (same for itgt_box, tgt_ibox)

        loopy_insns = [
                insn.copy(
                    predicates=insn.predicates | frozenset(["is_src_box_valid"]),
                    id=lp.UniqueName("compute_coeff"))
                for insn in self.get_translation_loopy_insns()]

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: 0<=isrc_box<nchildren}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]

                    <> tgt_center[idim] = centers[idim, tgt_ibox] \

                    for isrc_box
                        <> src_ibox = box_child_ids[isrc_box,tgt_ibox] \
                                {id=read_src_ibox}
                        <> is_src_box_valid = src_ibox != 0

                        if is_src_box_valid
                            <> src_center[idim] = centers[idim, src_ibox] {dup=idim}
                            <> d[idim] = tgt_center[idim] - src_center[idim] \
                                    {dup=idim}

                            """] + ["""
                            <> src_coeff{i} = \
                                src_expansions[src_ibox - src_base_ibox, {i}] \
                                {{id_prefix=read_coeff,dep=read_src_ibox}}
                            """.format(i=i) for i in range(ncoeffs)] + [
                            ] + loopy_insns + ["""
                            tgt_expansions[tgt_ibox - tgt_base_ibox, {i}] = \
                                tgt_expansions[tgt_ibox - tgt_base_ibox, {i}] \
                                + coeff{i} \
                                {{id_prefix=write_expn,dep=compute_coeff*,
                                    nosync=read_coeff*}}
                            """.format(i=i) for i in range(ncoeffs)] + ["""
                        end
                    end
                end
                """],
                [
                    lp.GlobalArg("target_boxes", None, shape=lp.auto,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("src_rscale,tgt_rscale", None),
                    lp.GlobalArg("box_child_ids", None,
                        shape="nchildren, aligned_nboxes"),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeffs), offset=lp.auto),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs), offset=lp.auto),
                    lp.ValueArg("src_base_ibox,tgt_base_ibox", np.int32),
                    lp.ValueArg("ntgt_level_boxes,nsrc_level_boxes", np.int32),
                    lp.ValueArg("aligned_nboxes", np.int32),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters=dict(dim=self.dim, nchildren=2**self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        knl = self.get_cached_optimized_kernel()

        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))

        return knl(queue,
                centers=centers,
                src_rscale=src_rscale, tgt_rscale=tgt_rscale,
                **kwargs)

# }}}


# {{{ translation from a box's parent

class E2EFromParent(E2EBase):
    default_name = "e2e_from_parent"

    def get_kernel(self):
        if self.src_expansion is not self.tgt_expansion:
            raise RuntimeError("%s requires that the source "
                    "and target expansion are the same object"
                    % self.default_name)

        ncoeffs = len(self.src_expansion)

        # To clarify terminology:
        #
        # isrc_box -> The index in a list of (in this case, source) boxes
        # src_ibox -> The (global) box number for the (in this case, source) box
        #
        # (same for itgt_box, tgt_ibox)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = lp.make_kernel(
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]

                    <> tgt_center[idim] = centers[idim, tgt_ibox] \

                    <> src_ibox = box_parent_ids[tgt_ibox] \
                        {id=read_src_ibox}

                    <> src_center[idim] = centers[idim, src_ibox] {dup=idim}
                    <> d[idim] = tgt_center[idim] - src_center[idim] {dup=idim}

                    """] + ["""
                    <> src_coeff{i} = \
                        src_expansions[src_ibox - src_base_ibox, {i}] \
                        {{id_prefix=read_expn,dep=read_src_ibox}}
                    """.format(i=i) for i in range(ncoeffs)] + [

                    ] + self.get_translation_loopy_insns() + ["""

                    tgt_expansions[tgt_ibox - tgt_base_ibox, {i}] = \
                        tgt_expansions[tgt_ibox - tgt_base_ibox, {i}] + coeff{i} \
                        {{id_prefix=write_expn,nosync=read_expn*}}
                    """.format(i=i) for i in range(ncoeffs)] + ["""
                end
                """],
                [
                    lp.GlobalArg("target_boxes", None, shape=lp.auto,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.ValueArg("src_rscale,tgt_rscale", None),
                    lp.ValueArg("naligned_boxes,nboxes", np.int32),
                    lp.ValueArg("tgt_base_ibox,src_base_ibox", np.int32),
                    lp.ValueArg("ntgt_level_boxes,nsrc_level_boxes", np.int32),
                    lp.GlobalArg("box_parent_ids", None, shape="nboxes"),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeffs), offset=lp.auto),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs), offset=lp.auto),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters=dict(dim=self.dim, nchildren=2**self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        knl = self.get_cached_optimized_kernel()

        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))

        return knl(queue,
                centers=centers,
                src_rscale=src_rscale, tgt_rscale=tgt_rscale,
                **kwargs)

# }}}

# vim: foldmethod=marker
