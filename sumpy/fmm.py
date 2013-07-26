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

    def __init__(self, cl_context, tree,
            multipole_expansion, local_expansion, out_kernels):
        self.tree = tree
        m_expn = self.multipole_expansion = multipole_expansion
        l_expn = self.local_expansion = local_expansion

        self.cl_context = cl_context

        from sumpy import (
                P2EFromLocal, P2EFromCSR,
                E2PFromLocal, E2PFromCSR,
                P2PFromCSR,
                E2EFromCSR, E2EFromChildren, E2EFromParent)
        self.p2m = P2EFromLocal(cl_context, m_expn)
        self.p2l = P2EFromCSR(cl_context, l_expn)
        self.m2m = E2EFromChildren(cl_context, m_expn, m_expn)
        self.m2l = E2EFromCSR(cl_context, m_expn, l_expn)
        self.l2l = E2EFromParent(cl_context, l_expn, l_expn)
        self.m2p = E2PFromCSR(cl_context, m_expn, out_kernels)
        self.l2p = E2PFromLocal(cl_context, l_expn, out_kernels)
        self.p2p = P2PFromCSR(cl_context, out_kernels, exclude_self=False)

    def get_wrangler(self, queue, dtype, extra_kwargs={}):
        return SumpyExpansionWrangler(self, queue, dtype, extra_kwargs)


class SumpyExpansionWrangler(object):
    """Implements the :class:`boxtree.fmm.ExpansionWranglerInterface`
    by using :mod:`sumpy` expansions/translations.
    """

    def __init__(self, code_container, queue, dtype, extra_kwargs):
        self.code = code_container
        self.queue = queue
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs

    @property
    def tree(self):
        return self.code.tree

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
        return tuple(
                cl.array.zeros(
                    self.queue,
                    self.tree.ntargets,
                    dtype=self.dtype)
                for k in self.out_kernels)

    def reorder_src_weights(self, src_weights):
        return src_weights.with_queue(self.queue)[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials.with_queue(self.queue)[self.tree.sorted_target_ids]

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
                strengths=(src_weights,),
                box_target_starts=self.tree.box_target_starts,
                box_target_counts_nonchild=self.tree.box_target_counts_nonchild,
                sources=self.tree.sources,
                targets=self.tree.targets,
                result=pot,

                **self.extra_kwargs)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot_res

    def multipole_to_local(self, target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        local_exps = self.local_expansion_zeros()

        evt, (local_exps_res,) = self.m2l(self.queue,
                src_expansions=mpole_exps,
                target_boxes=target_or_target_parent_boxes,
                src_box_starts=starts,
                src_box_lists=lists,
                centers=self.tree.centers,
                tgt_expansions=local_exps,

                **self.extra_kwargs)

        assert local_exps_res is local_exps

        return local_exps

    def eval_multipoles(self, target_boxes, sep_smaller_nonsiblings_starts,
            sep_smaller_nonsiblings_lists, mpole_exps):
        pot = self.potential_zeros()

        evt, pot_res = self.m2p(self.queue,
                expansions=mpole_exps,
                target_boxes=target_boxes,
                box_target_starts=self.tree.box_target_starts,
                box_target_counts_nonchild=self.tree.box_target_counts_nonchild,
                source_box_starts=sep_smaller_nonsiblings_starts,
                source_box_lists=sep_smaller_nonsiblings_lists,
                centers=self.tree.centers,
                targets=self.tree.targets,

                **self.extra_kwargs)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    def form_locals(self, target_or_target_parent_boxes, starts, lists, src_weights):
        local_exps = self.expansion_zeros()

        evt, (local_exps,) = self.p2l(self.queue,
                source_boxes=source_boxes,
                box_source_starts=box_source_starts,
                box_source_counts_nonchild=box_source_counts_nonchild,
                centers=self.tree.centers,
                sources=self.tree.sources,
                strengths=src_weights,

                **self.extra_kwargs)
        from pyfmmlib import h2dformta

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box:itgt_box+2]

            contrib = 0

            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                tgt_center = self.tree.box_centers[:, tgt_ibox]

                if src_pslice.stop - src_pslice.start == 0:
                    continue

                ier, mpole = h2dformta(
                        self.helmholtz_k, rscale,
                        self._get_sources(src_pslice), src_weights[src_pslice],
                        tgt_center, self.nterms)
                if ier:
                    raise RuntimeError("h2dformta failed")

                contrib = contrib + mpole

            local_exps[tgt_ibox] = contrib

        return local_exps

    def refine_locals(self, child_boxes, local_exps):
        from pyfmmlib import h2dlocloc_vec

        for tgt_ibox in child_boxes:
            tgt_center = self.tree.box_centers[:, tgt_ibox]
            src_ibox = self.tree.box_parent_ids[tgt_ibox]
            src_center = self.tree.box_centers[:, src_ibox]

            tmp_loc_exp = h2dlocloc_vec(
                        self.helmholtz_k,
                        rscale, src_center, local_exps[src_ibox],
                        rscale, tgt_center, self.nterms)[:, 0]

            local_exps[tgt_ibox] += tmp_loc_exp

        return local_exps

    def eval_locals(self, target_boxes, local_exps):
        pot = self.potential_zeros()
        rscale = 1  # FIXME

        from pyfmmlib import h2dtaeval_vec

        for tgt_ibox in target_boxes:
            tgt_pslice = self._get_target_slice(tgt_ibox)

            if tgt_pslice.stop - tgt_pslice.start == 0:
                continue

            tmp_pot, _, _ = h2dtaeval_vec(self.helmholtz_k, rscale,
                    self.tree.box_centers[:, tgt_ibox], local_exps[tgt_ibox],
                    self._get_targets(tgt_pslice), ifgrad=False, ifhess=False)

            pot[tgt_pslice] += tmp_pot

        return pot
