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
import sympy as sp
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
        from sumpy.symbolic import make_sympy_vector
        dvec = make_sympy_vector("d", self.dim)

        src_coeff_exprs = [sp.Symbol("src_coeff%d" % i)
                for i in range(len(self.src_expansion))]

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique("coeff%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, dvec))]

        sac.run_global_cse()

        from sumpy.symbolic import kill_trivial_assignments
        assignments = kill_trivial_assignments([
                (name, expr)
                for name, expr in six.iteritems(sac.assignments)],
                retain_names=tgt_coeff_names)

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                assignments,
                vector_names=set(["d"]),
                pymbolic_expr_maps=[self.tgt_expansion.transform_to_code],
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
                            src_expansions[src_ibox, {coeffidx}] \
                            {{dep=read_src_ibox}}
                        """.format(coeffidx=i) for i in range(ncoeff_src)] + [

                        ] + self.get_translation_loopy_insns() + ["""
                    end

                    """] + ["""
                    tgt_expansions[tgt_ibox, {coeffidx}] = \
                            simul_reduce(sum, isrc_box, coeff{coeffidx}) \
                            {{id_prefix=write_expn}}
                    """.format(coeffidx=i) for i in range(ncoeff_tgt)] + ["""
                end
                """],
                [
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.GlobalArg("src_box_starts, src_box_lists",
                        None, shape=None, strides=(1,)),
                    lp.ValueArg("aligned_nboxes,nboxes", np.int32),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nboxes", ncoeff_src)),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("nboxes", ncoeff_tgt)),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)")

        loopy_knl = lp.fix_parameters(loopy_knl, dim=self.dim)

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
        :arg centers:
        """
        knl = self.get_cached_optimized_kernel()

        return knl(queue, **kwargs)

# }}}


# {{{ translation from a box's children

class E2EFromChildren(E2EBase):
    default_name = "e2e_from_children"

    def get_kernel(self):
        if self.src_expansion is not self.tgt_expansion:
            raise RuntimeError("%s requires that the source "
                    "and target expansion are the same object")

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
                                expansions[src_ibox, {i}] \
                                {{id_prefix=read_coeff,dep=read_src_ibox}}
                            """.format(i=i) for i in range(ncoeffs)] + [
                            ] + loopy_insns + ["""
                            expansions[tgt_ibox, {i}] = \
                                expansions[tgt_ibox, {i}] + coeff{i} \
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
                    lp.GlobalArg("box_child_ids", None,
                        shape="nchildren, aligned_nboxes"),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", ncoeffs)),
                    lp.ValueArg("nboxes", np.int32),
                    lp.ValueArg("aligned_nboxes", np.int32),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)")

        loopy_knl = lp.fix_parameters(loopy_knl,
                dim=self.dim,
                nchildren=2**self.dim)

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg centers:
        """
        knl = self.get_cached_optimized_kernel()

        return knl(queue, **kwargs)

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
                        expansions[src_ibox, {i}] \
                        {{id_prefix=read_expn,dep=read_src_ibox}}
                    """.format(i=i) for i in range(ncoeffs)] + [

                    ] + self.get_translation_loopy_insns() + ["""

                    expansions[tgt_ibox, {i}] = \
                        expansions[tgt_ibox, {i}] + coeff{i} \
                        {{id_prefix=write_expn,nosync=read_expn*}}
                    """.format(i=i) for i in range(ncoeffs)] + ["""
                end
                """],
                [
                    lp.GlobalArg("target_boxes", None, shape=lp.auto,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.ValueArg("naligned_boxes,nboxes", np.int32),
                    lp.GlobalArg("box_parent_ids", None, shape="nboxes"),
                    lp.GlobalArg("expansions", None,
                        shape=("nboxes", ncoeffs)),
                    "..."
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)")

        loopy_knl = lp.fix_parameters(loopy_knl,
                dim=self.dim,
                nchildren=2**self.dim)
        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg centers:
        """
        knl = self.get_cached_optimized_kernel()

        return knl(queue, **kwargs)

# }}}

# vim: foldmethod=marker
