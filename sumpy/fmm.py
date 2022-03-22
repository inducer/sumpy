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

.. autoclass:: SumpyTreeIndependentDataForWrangler
.. autoclass:: SumpyExpansionWrangler
"""


import pyopencl as cl
import pyopencl.array  # noqa

from pytools import memoize_method
from boxtree.fmm import TreeIndependentDataForWrangler, ExpansionWranglerInterface

from sumpy import (
        P2EFromSingleBox, P2EFromCSR,
        E2PFromSingleBox, E2PFromCSR,
        P2PFromCSR,
        E2EFromCSR, M2LUsingTranslationClassesDependentData,
        E2EFromChildren, E2EFromParent,
        M2LGenerateTranslationClassesDependentData,
        M2LPreprocessMultipole, M2LPostprocessLocal)
from sumpy.tools import to_complex_dtype


def level_to_rscale(tree, level):
    return tree.root_extent * (2**-level)


# {{{ tree-independent data for wrangler

class SumpyTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
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
            target_kernels, exclude_self=False, use_rscale=None,
            strength_usage=None, source_kernels=None,
            use_fft_for_m2l=False):
        """
        :arg multipole_expansion_factory: a callable of a single argument (order)
            that returns a multipole expansion.
        :arg local_expansion_factory: a callable of a single argument (order)
            that returns a local expansion.
        :arg target_kernels: a list of output kernels
        :arg exclude_self: whether the self contribution should be excluded
        :arg strength_usage: passed unchanged to p2l, p2m and p2p.
        :arg source_kernels: passed unchanged to p2l, p2m and p2p.
        """
        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory
        self.source_kernels = source_kernels
        self.target_kernels = target_kernels
        self.exclude_self = exclude_self
        self.use_rscale = use_rscale
        self.strength_usage = strength_usage
        self.use_fft_for_m2l = use_fft_for_m2l

        super().__init__()

        self.cl_context = cl_context

    @memoize_method
    def get_base_kernel(self):
        from pytools import single_valued
        return single_valued(k.get_base_kernel() for k in self.target_kernels)

    @memoize_method
    def multipole_expansion(self, order):
        return self.multipole_expansion_factory(order, self.use_rscale)

    @memoize_method
    def local_expansion(self, order):
        return self.local_expansion_factory(order, self.use_rscale,
                use_fft_for_m2l=self.use_fft_for_m2l)

    @memoize_method
    def p2m(self, tgt_order):
        return P2EFromSingleBox(self.cl_context,
                kernels=self.source_kernels,
                expansion=self.multipole_expansion(tgt_order),
                strength_usage=self.strength_usage)

    @memoize_method
    def p2l(self, tgt_order):
        return P2EFromCSR(self.cl_context,
                kernels=self.source_kernels,
                expansion=self.local_expansion(tgt_order),
                strength_usage=self.strength_usage)

    @memoize_method
    def m2m(self, src_order, tgt_order):
        return E2EFromChildren(self.cl_context,
                self.multipole_expansion(src_order),
                self.multipole_expansion(tgt_order))

    @memoize_method
    def m2l(self, src_order, tgt_order,
            m2l_use_translation_classes_dependent_data=False):
        if m2l_use_translation_classes_dependent_data:
            m2l_class = M2LUsingTranslationClassesDependentData
        else:
            m2l_class = E2EFromCSR
        return m2l_class(self.cl_context,
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def m2l_translation_class_dependent_data_kernel(self, src_order, tgt_order):
        return M2LGenerateTranslationClassesDependentData(self.cl_context,
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def m2l_preprocess_mpole_kernel(self, src_order, tgt_order):
        return M2LPreprocessMultipole(self.cl_context,
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def m2l_postprocess_local_kernel(self, src_order, tgt_order):
        return M2LPostprocessLocal(self.cl_context,
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
                self.target_kernels)

    @memoize_method
    def l2p(self, src_order):
        return E2PFromSingleBox(self.cl_context,
                self.local_expansion(src_order),
                self.target_kernels)

    @memoize_method
    def p2p(self):
        return P2PFromCSR(self.cl_context, target_kernels=self.target_kernels,
                          source_kernels=self.source_kernels,
                          exclude_self=self.exclude_self,
                          strength_usage=self.strength_usage)

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
        from boxtree.timing import TimingResult

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


# {{{ translation classes data

class SumpyTranslationClassesData:
    """A class for building and storing additional, optional data for
    precomputation of translation classes passed to the expansion wrangler."""

    def __init__(self, queue, trav, is_translation_per_level=True):
        # FIXME: Queues should not be part of data.
        self.queue = queue
        self.trav = trav
        self.tree = trav.tree
        self.is_translation_per_level = is_translation_per_level

    @property
    @memoize_method
    def translation_classes_builder(self):
        from boxtree.translation_classes import TranslationClassesBuilder
        return TranslationClassesBuilder(self.queue.context)

    @memoize_method
    def build_translation_classes_lists(self):
        return self.translation_classes_builder(self.queue, self.trav, self.tree,
            is_translation_per_level=self.is_translation_per_level)[0]

    @memoize_method
    def m2l_translation_classes_lists(self):
        return (self
                .build_translation_classes_lists()
                .from_sep_siblings_translation_classes)

    @memoize_method
    def m2l_translation_vectors(self):
        return (self
                .build_translation_classes_lists()
                .from_sep_siblings_translation_class_to_distance_vector)

    def m2l_translation_classes_level_starts(self):
        return (self
                .build_translation_classes_lists()
                .from_sep_siblings_translation_classes_level_starts)


class SumpyTranslationClassesDataNotSuppliedWarning(UserWarning):
    pass

# }}}


# {{{ expansion wrangler

class SumpyExpansionWrangler(ExpansionWranglerInterface):
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

    .. attribute:: preprocessed_mpole_dtype

        Type for the preprocessed multipole expansion if used for M2L.
    """

    def __init__(self, tree_indep, traversal, dtype, fmm_level_to_order,
            source_extra_kwargs=None,
            kernel_extra_kwargs=None,
            self_extra_kwargs=None,
            translation_classes_data=None,
            preprocessed_mpole_dtype=None):
        super().__init__(tree_indep, traversal)
        self.issued_timing_data_warning = False

        self.dtype = dtype

        if not self.tree_indep.use_fft_for_m2l:
            # If not FFT, we don't need complex dtypes
            self.preprocessed_mpole_dtype = dtype
        elif preprocessed_mpole_dtype is not None:
            self.preprocessed_mpole_dtype = preprocessed_mpole_dtype
        else:
            # FIXME: It is weird that the wrangler has to compute this.
            self.preprocessed_mpole_dtype = to_complex_dtype(dtype)

        if source_extra_kwargs is None:
            source_extra_kwargs = {}
        if kernel_extra_kwargs is None:
            kernel_extra_kwargs = {}
        if self_extra_kwargs is None:
            self_extra_kwargs = {}

        if not callable(fmm_level_to_order):
            raise TypeError("fmm_level_to_order not passed")

        base_kernel = tree_indep.get_base_kernel()
        kernel_arg_set = frozenset(kernel_extra_kwargs.items())
        self.level_orders = [
                fmm_level_to_order(base_kernel, kernel_arg_set, traversal.tree, lev)
                for lev in range(traversal.tree.nlevels)]

        self.source_extra_kwargs = source_extra_kwargs
        self.kernel_extra_kwargs = kernel_extra_kwargs
        self.self_extra_kwargs = self_extra_kwargs

        self.extra_kwargs = source_extra_kwargs.copy()
        self.extra_kwargs.update(self.kernel_extra_kwargs)

        if base_kernel.is_translation_invariant:
            if translation_classes_data is None:
                from warnings import warn
                if self.tree_indep.use_fft_for_m2l:
                    raise NotImplementedError(
                         "FFT based List 2 (multipole-to-local) translations "
                         "without translation_classes_data argument is not "
                         "implemented. Supply a translation_classes_data argument "
                         "to the wrangler for optimized List 2.")
                else:
                    warn(
                         "List 2 (multipole-to-local) translations will be "
                         "unoptimized. Supply a translation_classes_data argument "
                         "to the wrangler for optimized List 2.",
                         SumpyTranslationClassesDataNotSuppliedWarning,
                         stacklevel=2)
                self.supports_optimized_m2l = False
            else:
                self.supports_optimized_m2l = True
        else:
            self.supports_optimized_m2l = False

        self.translation_classes_data = translation_classes_data
        self.use_fft_for_m2l = self.tree_indep.use_fft_for_m2l

    # {{{ data vector utilities

    def _expansions_level_starts(self, order_to_size):
        return build_csr_level_starts(self.level_orders, order_to_size,
                self.tree.level_start_box_nrs)

    @memoize_method
    def multipole_expansions_level_starts(self):
        return self._expansions_level_starts(
                lambda order: len(self.tree_indep.multipole_expansion(order)))

    @memoize_method
    def local_expansions_level_starts(self):
        return self._expansions_level_starts(
                lambda order: len(self.tree_indep.local_expansion(order)))

    @memoize_method
    def m2l_translation_class_level_start_box_nrs(self):
        with cl.CommandQueue(self.tree_indep.cl_context) as queue:
            data = self.translation_classes_data
            return data.m2l_translation_classes_level_starts().get(queue)

    @memoize_method
    def m2l_translation_classes_dependent_data_level_starts(self):
        def order_to_size(order):
            mpole_expn = self.tree_indep.multipole_expansion(order)
            local_expn = self.tree_indep.local_expansion(order)
            return local_expn.m2l_translation_classes_dependent_ndata(mpole_expn)

        return build_csr_level_starts(self.level_orders, order_to_size,
                level_starts=self.m2l_translation_class_level_start_box_nrs())

    def multipole_expansion_zeros(self, template_ary):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        :arg template_ary: an array (not necessarily of the same shape or dtype as
            the one to be created) whose run-time environment
            (e.g. :class:`pyopencl.CommandQueue`) the returned array should
            reuse.
        """
        return cl.array.zeros(
                template_ary.queue,
                self.multipole_expansions_level_starts()[-1],
                dtype=self.dtype)

    def local_expansion_zeros(self, template_ary):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        :arg template_ary: an array (not necessarily of the same shape or dtype as
            the one to be created) whose run-time environment
            (e.g. :class:`pyopencl.CommandQueue`) the returned array should
            reuse.
        """
        return cl.array.zeros(
                template_ary.queue,
                self.local_expansions_level_starts()[-1],
                dtype=self.dtype)

    def m2l_translation_classes_dependent_data_zeros(self, queue):
        return cl.array.zeros(
                queue,
                self.m2l_translation_classes_dependent_data_level_starts()[-1],
                dtype=self.preprocessed_mpole_dtype)

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

    def m2l_translation_classes_dependent_data_view(self,
                m2l_translation_classes_dependent_data, level):
        expn_start, expn_stop = \
            self.m2l_translation_classes_dependent_data_level_starts()[level:level+2]
        translation_class_start, translation_class_stop = \
            self.m2l_translation_class_level_start_box_nrs()[level:level+2]

        exprs_level = m2l_translation_classes_dependent_data[expn_start:expn_stop]
        return (translation_class_start, exprs_level.reshape(
                            translation_class_stop - translation_class_start, -1))

    @memoize_method
    def m2l_preproc_mpole_expansions_level_starts(self):
        def order_to_size(order):
            mpole_expn = self.tree_indep.multipole_expansion(order)
            local_expn = self.tree_indep.local_expansion(order)
            res = local_expn.m2l_preprocess_multipole_nexprs(mpole_expn)
            return res

        return build_csr_level_starts(self.level_orders, order_to_size,
                level_starts=self.tree.level_start_box_nrs)

    def m2l_preproc_mpole_expansion_zeros(self, template_ary):
        return cl.array.zeros(
                template_ary.queue,
                self.m2l_preproc_mpole_expansions_level_starts()[-1],
                dtype=self.preprocessed_mpole_dtype)

    def m2l_preproc_mpole_expansions_view(self, mpole_exps, level):
        expn_start, expn_stop = \
                self.m2l_preproc_mpole_expansions_level_starts()[level:level+2]
        box_start, box_stop = self.tree.level_start_box_nrs[level:level+2]

        return (box_start,
                mpole_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

    m2l_work_array_view = m2l_preproc_mpole_expansions_view
    m2l_work_array_zeros = m2l_preproc_mpole_expansion_zeros
    m2l_work_array_level_starts = \
            m2l_preproc_mpole_expansions_level_starts

    def output_zeros(self, template_ary):
        """Return a potentials array (which must support addition) capable of
        holding a potential value for each target in the tree. Note that
        :func:`drive_fmm` makes no assumptions about *potential* other than
        that it supports addition--it may consist of potentials, gradients of
        the potential, or arbitrary other per-target output data.
        :arg template_ary: an array (not necessarily of the same shape or dtype as
            the one to be created) whose run-time environment
            (e.g. :class:`pyopencl.CommandQueue`) the returned array should
            reuse.
        """
        from pytools.obj_array import make_obj_array
        return make_obj_array([
                cl.array.zeros(
                    template_ary.queue,
                    self.tree.ntargets,
                    dtype=self.dtype)
                for k in self.tree_indep.target_kernels])

    def reorder_sources(self, source_array):
        return source_array.with_queue(source_array.queue)[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        from pytools.obj_array import obj_array_vectorize
        import numpy as np
        assert (
                isinstance(potentials, np.ndarray)
                and potentials.dtype.char == "O")

        def reorder(x):
            return x[self.tree.sorted_target_ids]

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
            src_weight_vecs):
        mpoles = self.multipole_expansion_zeros(src_weight_vecs[0])

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        events = []
        queue = src_weight_vecs[0].queue

        for lev in range(self.tree.nlevels):
            p2m = self.tree_indep.p2m(self.level_orders[lev])
            start, stop = level_start_source_box_nrs[lev:lev+2]
            if start == stop:
                continue

            level_start_ibox, mpoles_view = self.multipole_expansions_view(
                    mpoles, lev)

            evt, (mpoles_res,) = p2m(
                    queue,
                    source_boxes=source_boxes[start:stop],
                    centers=self.tree.box_centers,
                    strengths=src_weight_vecs,
                    tgt_expansions=mpoles_view,
                    tgt_base_ibox=level_start_ibox,

                    rscale=level_to_rscale(self.tree, lev),

                    **kwargs)
            events.append(evt)

            assert mpoles_res is mpoles_view

        return (mpoles, SumpyTimingFuture(queue, events))

    def coarsen_multipoles(self,
            level_start_source_parent_box_nrs,
            source_parent_boxes,
            mpoles):
        tree = self.tree

        events = []
        queue = mpoles.queue

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

            m2m = self.tree_indep.m2m(
                    self.level_orders[source_level],
                    self.level_orders[target_level])

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpoles, source_level)
            target_level_start_ibox, target_mpoles_view = \
                    self.multipole_expansions_view(mpoles, target_level)

            evt, (mpoles_res,) = m2m(
                    queue,
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

        return (mpoles, SumpyTimingFuture(queue, events))

    def eval_direct(self, target_boxes, source_box_starts,
            source_box_lists, src_weight_vecs):
        pot = self.output_zeros(src_weight_vecs[0])

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.self_extra_kwargs)
        kwargs.update(self.box_source_list_kwargs())
        kwargs.update(self.box_target_list_kwargs())

        events = []
        queue = src_weight_vecs[0].queue

        evt, pot_res = self.tree_indep.p2p()(queue,
                target_boxes=target_boxes,
                source_box_starts=source_box_starts,
                source_box_lists=source_box_lists,
                strength=src_weight_vecs,
                result=pot,

                **kwargs)
        events.append(evt)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i
            pot_i.add_event(evt)

        return (pot, SumpyTimingFuture(queue, events))

    @memoize_method
    def multipole_to_local_precompute(self):
        with cl.CommandQueue(self.tree_indep.cl_context) as queue:
            m2l_translation_classes_dependent_data = \
                    self.m2l_translation_classes_dependent_data_zeros(queue)
            for lev in range(self.tree.nlevels):
                src_rscale = level_to_rscale(self.tree, lev)
                order = self.level_orders[lev]
                precompute_kernel = \
                    self.tree_indep.m2l_translation_class_dependent_data_kernel(
                            order, order)

                translation_classes_level_start, \
                    m2l_translation_classes_dependent_data_view = \
                        self.m2l_translation_classes_dependent_data_view(
                                m2l_translation_classes_dependent_data, lev)

                ntranslation_classes = \
                        m2l_translation_classes_dependent_data_view.shape[0]

                if ntranslation_classes == 0:
                    continue

                m2l_translation_vectors = (
                        self.translation_classes_data.m2l_translation_vectors())

                evt, _ = precompute_kernel(
                    queue,
                    src_rscale=src_rscale,
                    translation_classes_level_start=translation_classes_level_start,
                    ntranslation_classes=ntranslation_classes,
                    m2l_translation_classes_dependent_data=(
                        m2l_translation_classes_dependent_data_view),
                    m2l_translation_vectors=m2l_translation_vectors,
                    ntranslation_vectors=m2l_translation_vectors.shape[1],
                    **self.kernel_extra_kwargs
                )
                m2l_translation_classes_dependent_data.add_event(evt)

            m2l_translation_classes_dependent_data.finish()

            m2l_translation_classes_dependent_data = \
                    m2l_translation_classes_dependent_data.with_queue(None)
        return m2l_translation_classes_dependent_data

    def _add_m2l_precompute_kwargs(self, kwargs_for_m2l,
            lev):
        """This method is used for adding the information needed for a
        multipole-to-local translation with precomputation to the keywords
        passed to multipole-to-local translation.
        """
        if not self.supports_optimized_m2l:
            return
        m2l_translation_classes_dependent_data = \
                self.multipole_to_local_precompute()
        translation_classes_level_start, \
            m2l_translation_classes_dependent_data_view = \
                self.m2l_translation_classes_dependent_data_view(
                        m2l_translation_classes_dependent_data, lev)
        kwargs_for_m2l["m2l_translation_classes_dependent_data"] = \
            m2l_translation_classes_dependent_data_view
        kwargs_for_m2l["translation_classes_level_start"] = \
            translation_classes_level_start
        kwargs_for_m2l["m2l_translation_classes_lists"] = \
            self.translation_classes_data.m2l_translation_classes_lists()

    def multipole_to_local(self,
            level_start_target_box_nrs,
            target_boxes, src_box_starts, src_box_lists,
            mpole_exps):

        preprocess_evts = []
        queue = mpole_exps.queue
        local_exps = self.local_expansion_zeros(mpole_exps)

        if self.supports_optimized_m2l:
            preprocessed_mpole_exps = \
                    self.m2l_preproc_mpole_expansion_zeros(mpole_exps)
            for lev in range(self.tree.nlevels):
                order = self.level_orders[lev]
                preprocess_mpole_kernel = \
                    self.tree_indep.m2l_preprocess_mpole_kernel(order, order)

                _, source_mpoles_view = \
                        self.multipole_expansions_view(mpole_exps, lev)

                _, preprocessed_source_mpoles_view = \
                        self.m2l_preproc_mpole_expansions_view(
                                preprocessed_mpole_exps, lev)

                tr_classes = self.m2l_translation_class_level_start_box_nrs()
                if tr_classes[lev] == tr_classes[lev + 1]:
                    # There is no M2L happening in this level
                    continue

                evt, _ = preprocess_mpole_kernel(
                    queue,
                    src_expansions=source_mpoles_view,
                    preprocessed_src_expansions=preprocessed_source_mpoles_view,
                    src_rscale=level_to_rscale(self.tree, lev),
                    **self.kernel_extra_kwargs
                )
                preprocess_evts.append(evt)
            mpole_exps = preprocessed_mpole_exps
            m2l_work_array = self.m2l_work_array_zeros(local_exps)
            mpole_exps_view_func = self.m2l_preproc_mpole_expansions_view
            local_exps_view_func = self.m2l_work_array_view
        else:
            m2l_work_array = local_exps
            mpole_exps_view_func = self.multipole_expansions_view
            local_exps_view_func = self.local_expansions_view

        translate_evts = []

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            order = self.level_orders[lev]
            m2l = self.tree_indep.m2l(order, order, self.supports_optimized_m2l)

            source_level_start_ibox, source_mpoles_view = \
                    mpole_exps_view_func(mpole_exps, lev)
            target_level_start_ibox, target_locals_view = \
                    local_exps_view_func(m2l_work_array, lev)

            kwargs = dict(
                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_locals_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=target_boxes[start:stop],
                    src_box_starts=src_box_starts[start:stop],
                    src_box_lists=src_box_lists,
                    centers=self.tree.box_centers,

                    src_rscale=level_to_rscale(self.tree, lev),
                    tgt_rscale=level_to_rscale(self.tree, lev),

                    **self.kernel_extra_kwargs)

            self._add_m2l_precompute_kwargs(kwargs, lev)
            if "m2l_translation_classes_dependent_data" in kwargs and \
                    kwargs["m2l_translation_classes_dependent_data"].size == 0:
                # There is nothing to do for this level
                continue
            evt, _ = m2l(queue, **kwargs, wait_for=preprocess_evts)

            translate_evts.append(evt)

        postprocess_evts = []

        if self.supports_optimized_m2l:
            for lev in range(self.tree.nlevels):
                order = self.level_orders[lev]
                postprocess_local_kernel = \
                    self.tree_indep.m2l_postprocess_local_kernel(order, order)

                _, target_locals_view = \
                        self.local_expansions_view(local_exps, lev)

                _, target_locals_before_postprocessing_view = \
                        self.m2l_work_array_view(
                                m2l_work_array, lev)

                tr_classes = self.m2l_translation_class_level_start_box_nrs()
                if tr_classes[lev] == tr_classes[lev + 1]:
                    # There is no M2L happening in this level
                    continue

                evt, _ = postprocess_local_kernel(
                    queue,
                    tgt_expansions=target_locals_view,
                    tgt_expansions_before_postprocessing=(
                        target_locals_before_postprocessing_view),
                    src_rscale=level_to_rscale(self.tree, lev),
                    tgt_rscale=level_to_rscale(self.tree, lev),
                    wait_for=translate_evts,
                    **self.kernel_extra_kwargs,
                )
                postprocess_evts.append(evt)

        events = preprocess_evts + translate_evts + postprocess_evts

        return (local_exps, SumpyTimingFuture(queue, events))

    def eval_multipoles(self,
            target_boxes_by_source_level, source_boxes_by_level, mpole_exps):
        pot = self.output_zeros(mpole_exps)

        kwargs = self.kernel_extra_kwargs.copy()
        kwargs.update(self.box_target_list_kwargs())

        events = []
        queue = mpole_exps.queue

        wait_for = mpole_exps.events

        for isrc_level, ssn in enumerate(source_boxes_by_level):
            if len(target_boxes_by_source_level[isrc_level]) == 0:
                continue

            m2p = self.tree_indep.m2p(self.level_orders[isrc_level])

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpole_exps, isrc_level)

            evt, pot_res = m2p(
                    queue,

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

        return (pot, SumpyTimingFuture(queue, events))

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weight_vecs):
        local_exps = self.local_expansion_zeros(src_weight_vecs[0])

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        events = []
        queue = src_weight_vecs[0].queue

        for lev in range(self.tree.nlevels):
            start, stop = \
                    level_start_target_or_target_parent_box_nrs[lev:lev+2]
            if start == stop:
                continue

            p2l = self.tree_indep.p2l(self.level_orders[lev])

            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            evt, (result,) = p2l(
                    queue,
                    target_boxes=target_or_target_parent_boxes[start:stop],
                    source_box_starts=starts[start:stop+1],
                    source_box_lists=lists,
                    centers=self.tree.box_centers,
                    strengths=src_weight_vecs,

                    tgt_expansions=target_local_exps_view,
                    tgt_base_ibox=target_level_start_ibox,

                    rscale=level_to_rscale(self.tree, lev),

                    **kwargs)
            events.append(evt)

            assert result is target_local_exps_view

        return (local_exps, SumpyTimingFuture(queue, events))

    def refine_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps):

        events = []
        queue = local_exps.queue

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                    target_lev:target_lev+2]
            if start == stop:
                continue

            source_lev = target_lev - 1
            l2l = self.tree_indep.l2l(
                    self.level_orders[source_lev],
                    self.level_orders[target_lev])

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, source_lev)
            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, target_lev)

            evt, (local_exps_res,) = l2l(queue,
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

        return (local_exps, SumpyTimingFuture(queue, [evt]))

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        pot = self.output_zeros(local_exps)

        kwargs = self.kernel_extra_kwargs.copy()
        kwargs.update(self.box_target_list_kwargs())

        events = []
        queue = local_exps.queue

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            l2p = self.tree_indep.l2p(self.level_orders[lev])

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            evt, pot_res = l2p(
                    queue,

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

        return (pot, SumpyTimingFuture(queue, events))

    def finalize_potentials(self, potentials, template_ary):
        return potentials

# }}}


# {{{ build_csr_level_starts

def build_csr_level_starts(level_orders, order_to_size, level_starts):
    """Given a list of starts of boxes for each level and a callable
    that outputs the length of an expansion for a level, return
    a list of starts of an expansion for each level.
    Here, a list of starts for an object for each level means the
    starting indexes in an array for each level that stores the object
    in a row-major.

    :arg level_orders: A list of orders for each level.
    :arg order_to_size: A callable that returns the length of the
            expansion for the input level.
    :arg level_starts: A list of starts of boxes for each level.
    """
    result = [0]
    for lev in range(len(level_orders)):
        lev_nboxes = level_starts[lev+1] - level_starts[lev]

        expn_size = order_to_size(level_orders[lev])
        result.append(
                result[-1]
                + expn_size * lev_nboxes)

    return result

# }}}


# vim: foldmethod=marker
