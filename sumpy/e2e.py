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


# {{{ translation base class

class E2EBase(object):
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
        from sumpy.symbolic import make_sym_vector
        dvec = make_sym_vector("d", self.dim)

        src_coeff_exprs = [sp.Symbol("src_coeff%d" % i)
                for i in xrange(len(self.src_expansion))]

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique("coeff%d" % i, coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, dvec))]

        #sac.run_global_cse()

        from sumpy.symbolic import kill_trivial_assignments
        assignments = kill_trivial_assignments([
                (name, expr)
                for name, expr in sac.assignments.iteritems()],
                retain_names=tgt_coeff_names)

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                assignments,
                vector_names=set(["d"]),
                pymbolic_expr_maps=[self.tgt_expansion.transform_to_code],
                complex_dtype=np.complex128  # FIXME
                )

    @memoize_method
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

        from sumpy.tools import gather_arguments
        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                self.get_translation_loopy_insns()
                + ["""
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> tgt_center[idim] = centers[idim, tgt_ibox] \
                            {id=fetch_tgt_center}

                    <> isrc_start = src_box_starts[itgt_box]
                    <> isrc_stop = src_box_starts[itgt_box+1]

                    <> src_ibox = src_box_lists[isrc_box] \
                            {id=read_src_ibox}
                    <> src_center[idim] = centers[idim, src_ibox] \
                            {id=fetch_src_center}
                    <> d[idim] = tgt_center[idim] - src_center[idim]
                    <> src_coeff${SRC_COEFFIDX} = \
                        src_expansions[src_ibox, ${SRC_COEFFIDX}] \
                        {dep=read_src_ibox}

                    tgt_expansions[tgt_ibox, ${TGT_COEFFIDX}] = \
                            sum(isrc_box, coeff${TGT_COEFFIDX}) \
                            {id_prefix=write_expn}
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
                ] + gather_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ntgt_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    SRC_COEFFIDX=[str(i) for i in xrange(ncoeff_src)],
                    TGT_COEFFIDX=[str(i) for i in xrange(ncoeff_tgt)],
                    ),
                silenced_warnings="write_race(write_expn*)")

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg centers:
        """
        knl = self.get_optimized_kernel()

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

        from sumpy.tools import gather_arguments
        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[isrc_box]: 0<=isrc_box<nchildren}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                loopy_insns
                + ["""
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> tgt_center[idim] = centers[idim, tgt_ibox] \
                        {id=fetch_tgt_center}

                    <> src_ibox = box_child_ids[isrc_box,tgt_ibox] \
                            {id=read_src_ibox}
                    <> is_src_box_valid = src_ibox != 0

                    <> src_center[idim] = centers[idim, src_ibox] \
                        {id=fetch_src_center,if=is_src_box_valid}
                    <> d[idim] = tgt_center[idim] - src_center[idim] \
                        {if=is_src_box_valid}

                    <> src_coeff${COEFFIDX} = \
                        expansions[src_ibox, ${COEFFIDX}] \
                        {if=is_src_box_valid,dep=read_src_ibox}
                    expansions[tgt_ibox, ${COEFFIDX}] = \
                        expansions[tgt_ibox, ${COEFFIDX}] + coeff${COEFFIDX} \
                        {id_prefix=write_expn,if=is_src_box_valid,dep=compute_coeff*}
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
                ] + gather_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ntgt_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    nchildren=2**self.dim,
                    COEFFIDX=[str(i) for i in xrange(ncoeffs)],
                    ),
                silenced_warnings="write_race(write_expn*)")

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg centers:
        """
        knl = self.get_optimized_kernel()

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

        from sumpy.tools import gather_arguments
        loopy_knl = lp.make_kernel(self.device,
                [
                    "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                    "{[idim]: 0<=idim<dim}",
                    ],
                self.get_translation_loopy_insns()
                + ["""
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> tgt_center[idim] = centers[idim, tgt_ibox] \
                        {id=fetch_tgt_center}

                    <> src_ibox = box_parent_ids[tgt_ibox] \
                        {id=read_src_ibox}
                    <> src_center[idim] = centers[idim, src_ibox] \
                        {id=fetch_src_center}
                    <> d[idim] = tgt_center[idim] - src_center[idim]

                    <> src_coeff${COEFFIDX} = \
                        expansions[src_ibox, ${COEFFIDX}] \
                        {dep=read_src_ibox}
                    expansions[tgt_ibox, ${COEFFIDX}] = \
                        expansions[tgt_ibox, ${COEFFIDX}] + coeff${COEFFIDX} \
                        {id_prefix=write_expn}
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
                ] + gather_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name, assumptions="ntgt_boxes>=1",
                defines=dict(
                    dim=self.dim,
                    nchildren=2**self.dim,
                    COEFFIDX=[str(i) for i in xrange(ncoeffs)],
                    ),
                silenced_warnings="write_race(write_expn*)")

        for expn in [self.src_expansion, self.tgt_expansion]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.duplicate_inames(loopy_knl, "idim", "fetch_tgt_center",
                tags={"idim": "unr"})
        loopy_knl = lp.tag_inames(loopy_knl, dict(idim="unr"))

        return loopy_knl

    def __call__(self, queue, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg centers:
        """
        knl = self.get_optimized_kernel()

        return knl(queue, **kwargs)

# }}}

# vim: foldmethod=marker
