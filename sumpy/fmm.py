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

.. autoclass:: SumpyExpansionWranglerCodeContainer
.. autoclass:: SumpyExpansionWrangler
"""


import pyopencl as cl
import pyopencl.array  # noqa

from pytools import memoize_method

from sumpy import (
        P2EFromSingleBox, P2EFromCSR,
        E2PFromSingleBox, E2PFromCSR,
        P2PFromCSR,
        E2EFromCSR, E2EFromChildren, E2EFromParent)


def level_to_rscale(tree, level):
    return tree.root_extent * (2**-level)


# {{{ expansion wrangler code container

class SumpyExpansionWranglerCodeContainer:
    """Objects of this type serve as a place to keep the code needed
    for :class:`SumpyExpansionWrangler`. Since :class:`SumpyExpansionWrangler`
    necessarily must have a :class:`pyopencl.CommandQueue`, but this queue
    is allowed to be more ephemeral than the code, the code's lifetime
    is decoupled by storing it in this object.

    Timing results returned by this wrangler contain the values *wall_elapsed*
    which measures elapsed wall time. This requires a command queue with
    profiling enabled.
    """

    def __init__(self, cl_context,
            multipole_expansion_factory,
            local_expansion_factory,
            out_kernels, exclude_self=False, use_rscale=None):
        """
        :arg multipole_expansion_factory: a callable of a single argument (order)
            that returns a multipole expansion.
        :arg local_expansion_factory: a callable of a single argument (order)
            that returns a local expansion.
        :arg out_kernels: a list of output kernels
        :arg exclude_self: whether the self contribution should be excluded
        """
        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory
        self.out_kernels = out_kernels
        self.exclude_self = exclude_self
        self.use_rscale = use_rscale

        self.cl_context = cl_context

    @memoize_method
    def get_base_kernel(self):
        from pytools import single_valued
        return single_valued(k.get_base_kernel() for k in self.out_kernels)

    @memoize_method
    def multipole_expansion(self, order):
        return self.multipole_expansion_factory(order, self.use_rscale)

    @memoize_method
    def local_expansion(self, order):
        return self.local_expansion_factory(order, self.use_rscale)

    @memoize_method
    def p2m(self, tgt_order):
        return P2EFromSingleBox(self.cl_context,
                self.multipole_expansion(tgt_order))

    @memoize_method
    def p2l(self, tgt_order):
        return P2EFromCSR(self.cl_context,
                self.local_expansion(tgt_order))

    @memoize_method
    def m2m(self, src_order, tgt_order):
        return E2EFromChildren(self.cl_context,
                self.multipole_expansion(src_order),
                self.multipole_expansion(tgt_order))

    @memoize_method
    def m2l(self, src_order, tgt_order):
        return E2EFromCSR(self.cl_context,
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def l2l(self, src_order, tgt_order):
        return E2EFromParent(self.cl_context,
                self.local_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def m2p(self, src_order):
        return E2PFromCSR(self.cl_context,
                self.multipole_expansion(src_order),
                self.out_kernels)

    @memoize_method
    def l2p(self, src_order):
        return E2PFromSingleBox(self.cl_context,
                self.local_expansion(src_order),
                self.out_kernels)

    @memoize_method
    def p2p(self):
        return P2PFromCSR(self.cl_context, self.out_kernels,
                          exclude_self=self.exclude_self)

    def get_wrangler(self, queue, tree, dtype, fmm_level_to_order,
            source_extra_kwargs={},
            kernel_extra_kwargs=None,
            self_extra_kwargs=None):
        return SumpyExpansionWrangler(self, queue, tree, dtype, fmm_level_to_order,
                source_extra_kwargs, kernel_extra_kwargs, self_extra_kwargs)

# }}}


# {{{ timing future

_SECONDS_PER_NANOSECOND = 1e-9


class UnableToCollectTimingData(UserWarning):
    pass


class SumpyTimingFuture:

    def __init__(self, queue, events):
        self.queue = queue
        self.events = events

    @memoize_method
    def result(self):
        from boxtree.fmm import TimingResult

        if not self.queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            from warnings import warn
            warn(
                    "Profiling was not enabled in the command queue. "
                    "Timing data will not be collected.",
                    category=UnableToCollectTimingData,
                    stacklevel=3)
            return TimingResult(wall_elapsed=None)

        if self.events:
            pyopencl.wait_for_events(self.events)

        result = 0
        for event in self.events:
            result += (
                    (event.profile.end - event.profile.start)
                    * _SECONDS_PER_NANOSECOND)

        return TimingResult(wall_elapsed=result)

    def done(self):
        return all(
                event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
                == cl.command_execution_status.COMPLETE
                for event in self.events)

# }}}


# {{{ expansion wrangler

class SumpyExpansionWrangler:
    """Implements the :class:`boxtree.fmm.ExpansionWranglerInterface`
    by using :mod:`sumpy` expansions/translations.

    .. attribute:: source_extra_kwargs

        Keyword arguments to be passed to interactions that involve
        source particles.

    .. attribute:: kernel_extra_kwargs

        Keyword arguments to be passed to interactions that involve
        expansions, but not source particles.

    .. attribute:: self_extra_kwargs

        Keyword arguments to be passed for handling
        self interactions (source and target particles are the same),
        provided special handling is needed
    """

    def __init__(self, code_container, queue, tree, dtype, fmm_level_to_order,
            source_extra_kwargs,
            kernel_extra_kwargs=None,
            self_extra_kwargs=None):
        self.code = code_container
        self.queue = queue
        self.tree = tree
        self.issued_timing_data_warning = False

        self.dtype = dtype

        if kernel_extra_kwargs is None:
            kernel_extra_kwargs = {}

        if self_extra_kwargs is None:
            self_extra_kwargs = {}

        if not callable(fmm_level_to_order):
            raise TypeError("fmm_level_to_order not passed")

        base_kernel = code_container.get_base_kernel()
        kernel_arg_set = frozenset(kernel_extra_kwargs.items())
        self.level_orders = [
                fmm_level_to_order(base_kernel, kernel_arg_set, tree, lev)
                for lev in range(tree.nlevels)]

        self.source_extra_kwargs = source_extra_kwargs
        self.kernel_extra_kwargs = kernel_extra_kwargs
        self.self_extra_kwargs = self_extra_kwargs

        self.extra_kwargs = source_extra_kwargs.copy()
        self.extra_kwargs.update(self.kernel_extra_kwargs)

    # {{{ data vector utilities

    def _expansions_level_starts(self, order_to_size):
        result = [0]
        for lev in range(self.tree.nlevels):
            lev_nboxes = (
                    self.tree.level_start_box_nrs[lev+1]
                    - self.tree.level_start_box_nrs[lev])

            expn_size = order_to_size(self.level_orders[lev])
            result.append(
                    result[-1]
                    + expn_size * lev_nboxes)

        return result

    @memoize_method
    def multipole_expansions_level_starts(self):
        return self._expansions_level_starts(
                lambda order: len(self.code.multipole_expansion_factory(order)))

    @memoize_method
    def local_expansions_level_starts(self):
        return self._expansions_level_starts(
                lambda order: len(self.code.local_expansion_factory(order)))

    def multipole_expansion_zeros(self):
        return cl.array.zeros(
                self.queue,
                self.multipole_expansions_level_starts()[-1],
                dtype=self.dtype)

    def local_expansion_zeros(self):
        return cl.array.zeros(
                self.queue,
                self.local_expansions_level_starts()[-1],
                dtype=self.dtype)

    def multipole_expansions_view(self, mpole_exps, level):
        expn_start, expn_stop = \
                self.multipole_expansions_level_starts()[level:level+2]
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        return (box_start,
                mpole_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

    def local_expansions_view(self, local_exps, level):
        expn_start, expn_stop = \
                self.local_expansions_level_starts()[level:level+2]
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        return (box_start,
                local_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

    def output_zeros(self):
        from pytools.obj_array import make_obj_array
        return make_obj_array([
                cl.array.zeros(
                    self.queue,
                    self.tree.ntargets,
                    dtype=self.dtype)
                for k in self.code.out_kernels])

    def reorder_sources(self, source_array):
        return source_array.with_queue(self.queue)[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        from pytools.obj_array import obj_array_vectorize
        import numpy as np
        assert (
                isinstance(potentials, np.ndarray)
                and potentials.dtype.char == "O")

        def reorder(x):
            return x.with_queue(self.queue)[self.tree.sorted_target_ids]

        return obj_array_vectorize(reorder, potentials)

    # }}}

    # {{{ source/target dispatch

    # These exist so that subclasses can override access to per-box source/target
    # lists, for example to use point sources instead of regular sources, or to
    # use a FilteredTargetListsInTreeOrder object.

    def box_source_list_kwargs(self):
        return dict(
                box_source_starts=self.tree.box_source_starts,
                box_source_counts_nonchild=self.tree.box_source_counts_nonchild,
                sources=self.tree.sources)

    def box_target_list_kwargs(self):
        return dict(
                box_target_starts=self.tree.box_target_starts,
                box_target_counts_nonchild=self.tree.box_target_counts_nonchild,
                targets=self.tree.targets)

    # }}}

    def form_multipoles(self,
            level_start_source_box_nrs, source_boxes,
            src_weights):
        mpoles = self.multipole_expansion_zeros()

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        events = []

        for lev in range(self.tree.nlevels):
            p2m = self.code.p2m(self.level_orders[lev])
            start, stop = level_start_source_box_nrs[lev:lev+2]
            if start == stop:
                continue

            level_start_ibox, mpoles_view = self.multipole_expansions_view(
                    mpoles, lev)

            evt, (mpoles_res,) = p2m(
                    self.queue,
                    source_boxes=source_boxes[start:stop],
                    centers=self.tree.box_centers,
                    strengths=src_weights,
                    tgt_expansions=mpoles_view,
                    tgt_base_ibox=level_start_ibox,

                    rscale=level_to_rscale(self.tree, lev),

                    **kwargs)
            events.append(evt)

            assert mpoles_res is mpoles_view

        return (mpoles, SumpyTimingFuture(self.queue, events))

    def coarsen_multipoles(self,
            level_start_source_parent_box_nrs,
            source_parent_boxes,
            mpoles):
        tree = self.tree

        events = []

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels-1, 2, -1):
            target_level = source_level - 1
            assert target_level > 0

            start, stop = level_start_source_parent_box_nrs[
                            target_level:target_level+2]
            if start == stop:
                print("source", source_level, "empty")
                continue

            m2m = self.code.m2m(
                    self.level_orders[source_level],
                    self.level_orders[target_level])

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpoles, source_level)
            target_level_start_ibox, target_mpoles_view = \
                    self.multipole_expansions_view(mpoles, target_level)

            evt, (mpoles_res,) = m2m(
                    self.queue,
                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_mpoles_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=source_parent_boxes[start:stop],
                    box_child_ids=self.tree.box_child_ids,
                    centers=self.tree.box_centers,

                    src_rscale=level_to_rscale(self.tree, source_level),
                    tgt_rscale=level_to_rscale(self.tree, target_level),

                    **self.kernel_extra_kwargs)
            events.append(evt)

            assert mpoles_res is target_mpoles_view

        if events:
            mpoles.add_event(events[-1])

        return (mpoles, SumpyTimingFuture(self.queue, events))

    def eval_direct(self, target_boxes, source_box_starts,
            source_box_lists, src_weights):
        pot = self.output_zeros()

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.self_extra_kwargs)
        kwargs.update(self.box_source_list_kwargs())
        kwargs.update(self.box_target_list_kwargs())

        events = []

        evt, pot_res = self.code.p2p()(self.queue,
                target_boxes=target_boxes,
                source_box_starts=source_box_starts,
                source_box_lists=source_box_lists,
                strength=src_weights,
                result=pot,

                **kwargs)
        events.append(evt)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i
            pot_i.add_event(evt)

        return (pot, SumpyTimingFuture(self.queue, events))

    def multipole_to_local(self,
            level_start_target_box_nrs,
            target_boxes, src_box_starts, src_box_lists,
            mpole_exps):
        local_exps = self.local_expansion_zeros()

        events = []

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            order = self.level_orders[lev]
            m2l = self.code.m2l(order, order)

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpole_exps, lev)
            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            evt, (local_exps_res,) = m2l(
                    self.queue,

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_local_exps_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=target_boxes[start:stop],
                    src_box_starts=src_box_starts[start:stop],
                    src_box_lists=src_box_lists,
                    centers=self.tree.box_centers,

                    src_rscale=level_to_rscale(self.tree, lev),
                    tgt_rscale=level_to_rscale(self.tree, lev),

                    **self.kernel_extra_kwargs)
            events.append(evt)

        return (local_exps, SumpyTimingFuture(self.queue, events))

    def eval_multipoles(self,
            target_boxes_by_source_level, source_boxes_by_level, mpole_exps):
        pot = self.output_zeros()

        kwargs = self.kernel_extra_kwargs.copy()
        kwargs.update(self.box_target_list_kwargs())

        events = []

        wait_for = mpole_exps.events

        for isrc_level, ssn in enumerate(source_boxes_by_level):
            if len(target_boxes_by_source_level[isrc_level]) == 0:
                continue

            m2p = self.code.m2p(self.level_orders[isrc_level])

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpole_exps, isrc_level)

            evt, pot_res = m2p(
                    self.queue,

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,

                    target_boxes=target_boxes_by_source_level[isrc_level],
                    source_box_starts=ssn.starts,
                    source_box_lists=ssn.lists,
                    centers=self.tree.box_centers,
                    result=pot,

                    rscale=level_to_rscale(self.tree, isrc_level),

                    wait_for=wait_for,

                    **kwargs)
            events.append(evt)

            wait_for = [evt]

            for pot_i, pot_res_i in zip(pot, pot_res):
                assert pot_i is pot_res_i

        if events:
            for pot_i in pot:
                pot_i.add_event(events[-1])

        return (pot, SumpyTimingFuture(self.queue, events))

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights):
        local_exps = self.local_expansion_zeros()

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        events = []

        for lev in range(self.tree.nlevels):
            start, stop = \
                    level_start_target_or_target_parent_box_nrs[lev:lev+2]
            if start == stop:
                continue

            p2l = self.code.p2l(self.level_orders[lev])

            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            evt, (result,) = p2l(
                    self.queue,
                    target_boxes=target_or_target_parent_boxes[start:stop],
                    source_box_starts=starts[start:stop+1],
                    source_box_lists=lists,
                    centers=self.tree.box_centers,
                    strengths=src_weights,

                    tgt_expansions=target_local_exps_view,
                    tgt_base_ibox=target_level_start_ibox,

                    rscale=level_to_rscale(self.tree, lev),

                    **kwargs)
            events.append(evt)

            assert result is target_local_exps_view

        return (local_exps, SumpyTimingFuture(self.queue, events))

    def refine_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps):

        events = []

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            if start == stop:
                continue

            source_lev = target_lev - 1
            l2l = self.code.l2l(
                    self.level_orders[source_lev],
                    self.level_orders[target_lev])

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, source_lev)
            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, target_lev)

            evt, (local_exps_res,) = l2l(self.queue,
                    src_expansions=source_local_exps_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_local_exps_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=target_or_target_parent_boxes[start:stop],
                    box_parent_ids=self.tree.box_parent_ids,
                    centers=self.tree.box_centers,

                    src_rscale=level_to_rscale(self.tree, source_lev),
                    tgt_rscale=level_to_rscale(self.tree, target_lev),

                    **self.kernel_extra_kwargs)
            events.append(evt)

            assert local_exps_res is target_local_exps_view

        local_exps.add_event(evt)

        return (local_exps, SumpyTimingFuture(self.queue, [evt]))

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        pot = self.output_zeros()

        kwargs = self.kernel_extra_kwargs.copy()
        kwargs.update(self.box_target_list_kwargs())

        events = []

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            l2p = self.code.l2p(self.level_orders[lev])

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            evt, pot_res = l2p(
                    self.queue,

                    src_expansions=source_local_exps_view,
                    src_base_ibox=source_level_start_ibox,

                    target_boxes=target_boxes[start:stop],
                    centers=self.tree.box_centers,
                    result=pot,

                    rscale=level_to_rscale(self.tree, lev),

                    **kwargs)
            events.append(evt)

            for pot_i, pot_res_i in zip(pot, pot_res):
                assert pot_i is pot_res_i

        return (pot, SumpyTimingFuture(self.queue, events))

    def finalize_potentials(self, potentials):
        return potentials

# }}}

# vim: foldmethod=marker
