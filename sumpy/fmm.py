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

__doc__ = """Integrates :mod:`boxtree` with :mod:`sumpy`.
"""


import pyopencl as cl
import pyopencl.array  # noqa


class SumpyExpansionWranglerCodeContainer(object):
    """Objects of this type serve as a place to keep the code needed
    for :class:`SumpyExpansionWrangler`. Since :class:`SumpyExpansionWrangler`
    necessarily must have a :class:`pyopencl.CommandQueue`, but this queue
    is allowed to be more ephemeral than the code, the code's lifetime
    is decoupled by storing it in this object.
    """

    def __init__(self, cl_context,
            multipole_expansion, local_expansion, out_kernels):
        m_expn = self.multipole_expansion = multipole_expansion
        l_expn = self.local_expansion = local_expansion
        self.out_kernels = out_kernels

        self.cl_context = cl_context

        from sumpy import (
                P2EFromSingleBox, P2EFromCSR,
                E2PFromSingleBox, E2PFromCSR,
                P2PFromCSR,
                E2EFromCSR, E2EFromChildren, E2EFromParent)
        self.p2m = P2EFromSingleBox(cl_context, m_expn)
        self.p2l = P2EFromCSR(cl_context, l_expn)
        self.m2m = E2EFromChildren(cl_context, m_expn, m_expn)
        self.m2l = E2EFromCSR(cl_context, m_expn, l_expn)
        self.l2l = E2EFromParent(cl_context, l_expn, l_expn)
        self.m2p = E2PFromCSR(cl_context, m_expn, out_kernels)
        self.l2p = E2PFromSingleBox(cl_context, l_expn, out_kernels)

        # FIXME figure out what to do about exclude_self
        self.p2p = P2PFromCSR(cl_context, out_kernels, exclude_self=False)

    def get_wrangler(self, queue, tree, dtype, extra_kwargs={}):
        return SumpyExpansionWrangler(self, queue, tree, dtype, extra_kwargs)


class SumpyExpansionWrangler(object):
    """Implements the :class:`boxtree.fmm.ExpansionWranglerInterface`
    by using :mod:`sumpy` expansions/translations.
    """

    def __init__(self, code_container, queue, tree, dtype, extra_kwargs):
        self.code = code_container
        self.queue = queue
        self.tree = tree
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs

    @property
    def multipole_expansion(self):
        return self.code.multipole_expansion

    @property
    def local_expansion(self):
        return self.code.local_expansion

    def multipole_expansion_zeros(self):
        return cl.array.zeros(
                self.queue,
                (self.tree.nboxes, len(self.multipole_expansion)),
                dtype=self.dtype)

    def local_expansion_zeros(self):
        return cl.array.zeros(
                self.queue,
                (self.tree.nboxes, len(self.local_expansion)),
                dtype=self.dtype)

    def potential_zeros(self):
        from pytools.obj_array import make_obj_array
        return make_obj_array([
                cl.array.zeros(
                    self.queue,
                    self.tree.ntargets,
                    dtype=self.dtype)
                for k in self.code.out_kernels])

    def reorder_src_weights(self, src_weights):
        return src_weights.with_queue(self.queue)[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        from pytools.obj_array import is_obj_array, with_object_array_or_scalar
        assert is_obj_array(potentials)

        def reorder(x):
            return x.with_queue(self.queue)[self.tree.sorted_target_ids]

        return with_object_array_or_scalar(reorder, potentials)

    def form_multipoles(self, source_boxes, src_weights):
        mpoles = self.multipole_expansion_zeros()

        evt, (mpoles_res,) = self.code.p2m(self.queue,
                source_boxes=source_boxes,
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=self.tree.box_source_counts_nonchild,
                centers=self.tree.box_centers,
                sources=self.tree.sources,
                strengths=src_weights,
                expansions=mpoles,

                **self.extra_kwargs)

        assert mpoles_res is mpoles

        return mpoles

    def coarsen_multipoles(self, parent_boxes, mpoles):
        if not len(parent_boxes):
            return mpoles

        evt, (mpoles_res,) = self.code.m2m(self.queue,
                expansions=mpoles,
                target_boxes=parent_boxes,
                box_child_ids=self.tree.box_child_ids,
                centers=self.tree.box_centers,

                **self.extra_kwargs)

        assert mpoles_res is mpoles
        return mpoles

    def eval_direct(self, target_boxes, source_box_starts,
            source_box_lists, src_weights):
        pot = self.potential_zeros()

        evt, pot_res = self.code.p2p(self.queue,
                target_boxes=target_boxes,
                source_box_starts=source_box_starts,
                source_box_lists=source_box_lists,
                strength=(src_weights,),
                box_target_starts=self.tree.box_target_starts,
                box_target_counts_nonchild=self.tree.box_target_counts_nonchild,
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=self.tree.box_source_counts_nonchild,
                sources=self.tree.sources,
                targets=self.tree.targets,
                result=pot,

                **self.extra_kwargs)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    def multipole_to_local(self, target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        local_exps = self.local_expansion_zeros()

        evt, (local_exps_res,) = self.code.m2l(self.queue,
                src_expansions=mpole_exps,
                target_boxes=target_or_target_parent_boxes,
                src_box_starts=starts,
                src_box_lists=lists,
                centers=self.tree.box_centers,
                tgt_expansions=local_exps,

                **self.extra_kwargs)

        assert local_exps_res is local_exps

        return local_exps

    def eval_multipoles(self, target_boxes, source_box_starts,
            source_box_lists, mpole_exps):
        pot = self.potential_zeros()

        evt, pot_res = self.code.m2p(self.queue,
                expansions=mpole_exps,
                target_boxes=target_boxes,
                box_target_starts=self.tree.box_target_starts,
                box_target_counts_nonchild=self.tree.box_target_counts_nonchild,
                source_box_starts=source_box_starts,
                source_box_lists=source_box_lists,
                centers=self.tree.box_centers,
                targets=self.tree.targets,
                result=pot,

                **self.extra_kwargs)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    def form_locals(self, target_or_target_parent_boxes, starts, lists, src_weights):
        local_exps = self.local_expansion_zeros()

        evt, (result,) = self.code.p2l(self.queue,
                target_boxes=target_or_target_parent_boxes,
                source_box_starts=starts,
                source_box_lists=lists,
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=self.tree.box_source_counts_nonchild,
                centers=self.tree.box_centers,
                sources=self.tree.sources,
                strengths=src_weights,
                expansions=local_exps,

                **self.extra_kwargs)

        assert local_exps is result

        return result

    def refine_locals(self, child_boxes, local_exps):
        if not len(child_boxes):
            return local_exps

        evt, (local_exps_res,) = self.code.l2l(self.queue,
                expansions=local_exps,
                target_boxes=child_boxes,
                box_parent_ids=self.tree.box_parent_ids,
                centers=self.tree.box_centers,

                **self.extra_kwargs)

        assert local_exps_res is local_exps
        return local_exps

    def eval_locals(self, target_boxes, local_exps):
        pot = self.potential_zeros()

        evt, pot_res = self.code.l2p(self.queue,
                expansions=local_exps,
                target_boxes=target_boxes,
                box_target_starts=self.tree.box_target_starts,
                box_target_counts_nonchild=self.tree.box_target_counts_nonchild,
                centers=self.tree.box_centers,
                targets=self.tree.targets,
                result=pot,

                **self.extra_kwargs)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot
