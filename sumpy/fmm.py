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

import numpy as np

from pytools import memoize_method
from boxtree.fmm import TreeIndependentDataForWrangler, ExpansionWranglerInterface

from arraycontext import Array

from sumpy.array_context import PyOpenCLArrayContext
from sumpy import (
        P2EFromSingleBox, P2EFromCSR,
        E2PFromSingleBox, E2PFromCSR,
        P2PFromCSR,
        E2EFromCSR, M2LUsingTranslationClassesDependentData,
        E2EFromChildren, E2EFromParent,
        M2LGenerateTranslationClassesDependentData,
        M2LPreprocessMultipole, M2LPostprocessLocal)
from sumpy.tools import to_complex_dtype, run_opencl_fft, get_opencl_fft_app


# {{{ tree-independent data for wrangler

class SumpyTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
    """Objects of this type serve as a place to keep the code needed
    for :class:`SumpyExpansionWrangler`. Since :class:`SumpyExpansionWrangler`
    contains data that is allowed to be more ephemeral than the code, the code's
    lifetime is decoupled by storing it in this object.
    """

    def __init__(self,
            array_context: PyOpenCLArrayContext,
            multipole_expansion_factory,
            local_expansion_factory,
            target_kernels, exclude_self=False, use_rscale=None,
            strength_usage=None, source_kernels=None):
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
        super().__init__()

        self._setup_actx = array_context

        self.multipole_expansion_factory = multipole_expansion_factory
        self.local_expansion_factory = local_expansion_factory
        self.source_kernels = source_kernels
        self.target_kernels = target_kernels
        self.exclude_self = exclude_self
        self.use_rscale = use_rscale
        self.strength_usage = strength_usage

    @memoize_method
    def get_base_kernel(self):
        from pytools import single_valued
        return single_valued(k.get_base_kernel() for k in self.target_kernels)

    @memoize_method
    def multipole_expansion(self, order):
        return self.multipole_expansion_factory(order, self.use_rscale)

    @memoize_method
    def local_expansion(self, order):
        return self.local_expansion_factory(order, self.use_rscale)

    @property
    def m2l_translation(self):
        return self.local_expansion(0).m2l_translation

    @memoize_method
    def p2m(self, tgt_order):
        return P2EFromSingleBox(
                kernels=self.source_kernels,
                expansion=self.multipole_expansion(tgt_order),
                strength_usage=self.strength_usage, name="p2m")

    @memoize_method
    def p2l(self, tgt_order):
        return P2EFromCSR(
                kernels=self.source_kernels,
                expansion=self.local_expansion(tgt_order),
                strength_usage=self.strength_usage, name="p2l")

    @memoize_method
    def m2m(self, src_order, tgt_order):
        return E2EFromChildren(
                self.multipole_expansion(src_order),
                self.multipole_expansion(tgt_order), name="m2m")

    @memoize_method
    def m2l(self, src_order, tgt_order,
            m2l_use_translation_classes_dependent_data=False):
        if m2l_use_translation_classes_dependent_data:
            m2l_class = M2LUsingTranslationClassesDependentData
        else:
            m2l_class = E2EFromCSR
        return m2l_class(
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order), name="m2l")

    @memoize_method
    def m2l_translation_class_dependent_data_kernel(self, src_order, tgt_order):
        return M2LGenerateTranslationClassesDependentData(
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def m2l_preprocess_mpole_kernel(self, src_order, tgt_order):
        return M2LPreprocessMultipole(
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def m2l_postprocess_local_kernel(self, src_order, tgt_order):
        return M2LPostprocessLocal(
                self.multipole_expansion(src_order),
                self.local_expansion(tgt_order))

    @memoize_method
    def l2l(self, src_order, tgt_order):
        return E2EFromParent(
                self.local_expansion(src_order),
                self.local_expansion(tgt_order), name="l2l")

    @memoize_method
    def m2p(self, src_order):
        return E2PFromCSR(
                self.multipole_expansion(src_order),
                self.target_kernels, name="m2p")

    @memoize_method
    def l2p(self, src_order):
        return E2PFromSingleBox(
                self.local_expansion(src_order),
                self.target_kernels, name="l2p")

    @memoize_method
    def p2p(self):
        return P2PFromCSR(target_kernels=self.target_kernels,
                          source_kernels=self.source_kernels,
                          exclude_self=self.exclude_self,
                          strength_usage=self.strength_usage, name="p2p")

    @memoize_method
    def opencl_fft_app(self, shape, dtype, inverse):
        return get_opencl_fft_app(self._setup_actx, shape, dtype, inverse=inverse)

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

    def __init__(self,
            tree_indep, traversal, dtype, fmm_level_to_order,
            source_extra_kwargs=None,
            kernel_extra_kwargs=None,
            self_extra_kwargs=None,
            translation_classes_data=None,
            preprocessed_mpole_dtype=None,
            *, _disable_translation_classes=False):
        super().__init__(tree_indep, traversal)

        self.dtype = dtype

        if not self.tree_indep.m2l_translation.use_fft:
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

        if _disable_translation_classes or not base_kernel.is_translation_invariant:
            self.supports_translation_classes = False
        else:
            if translation_classes_data is None:
                from boxtree.translation_classes import TranslationClassesBuilder

                actx = tree_indep._setup_actx
                translation_classes_builder = TranslationClassesBuilder(actx)
                translation_classes_data, _ = translation_classes_builder(
                    actx, traversal, self.tree, is_translation_per_level=True)
                translation_classes_data = actx.freeze(translation_classes_data)

            self.supports_translation_classes = True

        self.translation_classes_data = translation_classes_data

    def level_to_rscale(self, level):
        tree = self.tree
        order = self.level_orders[level]
        r = tree.root_extent * (2**-level)

        # See L. Greengard and V. Rokhlin. On the efficient implementation of the
        # fast multipole algorithm. Technical report,
        # YALE UNIV NEW HAVEN CT DEPT OF COMPUTER SCIENCE, 1988.
        # rscale that we use in sumpy is the inverse of the scaling used in the
        # paper and therefore we should use r / order. However empirically
        # we have observed that 2r / order is better for numerical stability
        # for Laplace and 4r / order for biharmonic kernel.
        knl = self.tree_indep.get_base_kernel()
        from sumpy.kernel import BiharmonicKernel
        if isinstance(knl, BiharmonicKernel):
            return r * 4 / order
        else:
            return r * 2 / order

    # {{{ data vector utilities

    @property
    @memoize_method
    def tree_level_start_box_nrs(self):
        # NOTE: a host version of `level_start_box_nrs` is used repeatedly and
        # this simply caches it to avoid repeated transfers
        actx = self.tree_indep._setup_actx
        return actx.to_numpy(self.tree.level_start_box_nrs)

    def _expansions_level_starts(self, order_to_size):
        return build_csr_level_starts(self.level_orders, order_to_size,
                self.tree_level_start_box_nrs)

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
        actx = self.tree_indep._setup_actx
        return actx.to_numpy(
            self.translation_classes_data
            .from_sep_siblings_translation_classes_level_starts)

    @memoize_method
    def m2l_translation_classes_dependent_data_level_starts(self):
        def order_to_size(order):
            mpole_expn = self.tree_indep.multipole_expansion(order)
            local_expn = self.tree_indep.local_expansion(order)
            m2l_translation = local_expn.m2l_translation
            return m2l_translation.translation_classes_dependent_ndata(
                    local_expn, mpole_expn)

        return build_csr_level_starts(self.level_orders, order_to_size,
                level_starts=self.m2l_translation_class_level_start_box_nrs())

    def multipole_expansion_zeros(self, actx: PyOpenCLArrayContext) -> Array:
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """
        return actx.zeros(
                self.multipole_expansions_level_starts()[-1],
                dtype=self.dtype)

    def local_expansion_zeros(self, actx) -> Array:
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """
        return actx.zeros(
                self.local_expansions_level_starts()[-1],
                dtype=self.dtype)

    def m2l_translation_classes_dependent_data_zeros(
            self, actx: PyOpenCLArrayContext):
        data_level_starts = (
            self.m2l_translation_classes_dependent_data_level_starts())
        level_start_box_nrs = (
            self.m2l_translation_class_level_start_box_nrs())

        result = []
        for level in range(self.tree.nlevels):
            expn_start, expn_stop = data_level_starts[level:level + 2]
            translation_class_start, translation_class_stop = (
                level_start_box_nrs[level:level + 2])

            exprs_level = actx.zeros(
                expn_stop - expn_start,
                dtype=self.preprocessed_mpole_dtype
                ).reshape(translation_class_stop - translation_class_start, -1)
            result.append(exprs_level)

        return result

    def multipole_expansions_view(self, mpole_exps, level):
        expn_start, expn_stop = (
                self.multipole_expansions_level_starts()[level:level + 2])
        box_start, box_stop = self.tree_level_start_box_nrs[level:level + 2]

        return (box_start,
                mpole_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

    def local_expansions_view(self, local_exps, level):
        expn_start, expn_stop = (
                self.local_expansions_level_starts()[level:level + 2])
        box_start, box_stop = self.tree_level_start_box_nrs[level:level + 2]

        return (box_start,
                local_exps[expn_start:expn_stop].reshape(box_stop-box_start, -1))

    def m2l_translation_classes_dependent_data_view(self,
                m2l_translation_classes_dependent_data, level):
        translation_class_start, _ = (
            self.m2l_translation_class_level_start_box_nrs()[level:level + 2])
        exprs_level = m2l_translation_classes_dependent_data[level]
        return (translation_class_start, exprs_level)

    @memoize_method
    def m2l_preproc_mpole_expansions_level_starts(self):
        def order_to_size(order):
            mpole_expn = self.tree_indep.multipole_expansion(order)
            local_expn = self.tree_indep.local_expansion(order)
            res = local_expn.m2l_translation.preprocess_multipole_nexprs(
                local_expn, mpole_expn)
            return res

        return build_csr_level_starts(self.level_orders, order_to_size,
                level_starts=self.tree_level_start_box_nrs)

    def m2l_preproc_mpole_expansion_zeros(
            self, actx: PyOpenCLArrayContext, template_ary):
        level_starts = self.m2l_preproc_mpole_expansions_level_starts()

        result = []
        for level in range(self.tree.nlevels):
            expn_start, expn_stop = level_starts[level:level+2]
            box_start, box_stop = self.tree_level_start_box_nrs[level:level+2]

            exprs_level = actx.zeros(
                expn_stop - expn_start,
                dtype=self.preprocessed_mpole_dtype,
                ).reshape(box_stop - box_start, -1)
            result.append(exprs_level)

        return result

    def m2l_preproc_mpole_expansions_view(self, mpole_exps, level):
        box_start, _ = self.tree_level_start_box_nrs[level:level+2]
        return (box_start, mpole_exps[level])

    m2l_work_array_view = m2l_preproc_mpole_expansions_view
    m2l_work_array_zeros = m2l_preproc_mpole_expansion_zeros
    m2l_work_array_level_starts = m2l_preproc_mpole_expansions_level_starts

    def output_zeros(self, actx: PyOpenCLArrayContext) -> np.ndarray:
        """Return a potentials array (which must support addition) capable of
        holding a potential value for each target in the tree. Note that
        :func:`drive_fmm` makes no assumptions about *potential* other than
        that it supports addition--it may consist of potentials, gradients of
        the potential, or arbitrary other per-target output data.
        """
        from pytools.obj_array import make_obj_array
        return make_obj_array([
                actx.zeros(self.tree.ntargets, dtype=self.dtype)
                for k in self.tree_indep.target_kernels])

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        from pytools.obj_array import obj_array_vectorize
        import numpy as np
        assert (
                isinstance(potentials, np.ndarray)
                and potentials.dtype.char == "O")

        def reorder(x):
            return x[self.tree.sorted_target_ids]

        return obj_array_vectorize(reorder, potentials)

    @property
    @memoize_method
    def max_nsources_in_one_box(self):
        actx = self.tree_indep._setup_actx
        return actx.to_numpy(
            actx.np.max(self.tree.box_source_counts_nonchild)
            ).item()

    @property
    @memoize_method
    def max_ntargets_in_one_box(self):
        actx = self.tree_indep._setup_actx
        return actx.to_numpy(
            actx.np.max(self.tree.box_target_counts_nonchild)
            ).item()

    # }}}

    # {{{ source/target dispatch

    # These exist so that subclasses can override access to per-box source/target
    # lists, for example to use point sources instead of regular sources, or to
    # use a FilteredTargetListsInTreeOrder object.

    def box_source_list_kwargs(self):
        return {
                "box_source_starts": self.tree.box_source_starts,
                "box_source_counts_nonchild": self.tree.box_source_counts_nonchild,
                "sources": self.tree.sources}

    def box_target_list_kwargs(self):
        return {
                "box_target_starts": self.tree.box_target_starts,
                "box_target_counts_nonchild": self.tree.box_target_counts_nonchild,
                "targets": self.tree.targets}

    # }}}

    def run_opencl_fft(self, actx: PyOpenCLArrayContext,
            input_vec, inverse, wait_for):
        app = self.tree_indep.opencl_fft_app(input_vec.shape, input_vec.dtype,
            inverse)
        evt, result = run_opencl_fft(
            actx, app, input_vec, inverse=inverse, wait_for=wait_for)

        from sumpy.tools import get_native_event
        input_vec.add_event(get_native_event(evt))
        result.add_event(get_native_event(evt))

        return result

    def form_multipoles(self,
            actx: PyOpenCLArrayContext,
            level_start_source_box_nrs, source_boxes,
            src_weight_vecs):
        mpoles = self.multipole_expansion_zeros(actx)
        level_start_source_box_nrs = actx.to_numpy(level_start_source_box_nrs)

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        for lev in range(self.tree.nlevels):
            p2m = self.tree_indep.p2m(self.level_orders[lev])
            start, stop = level_start_source_box_nrs[lev:lev+2]
            if start == stop:
                continue

            level_start_ibox, mpoles_view = self.multipole_expansions_view(
                    mpoles, lev)

            mpoles_res = p2m(
                    actx,
                    source_boxes=source_boxes[start:stop],
                    centers=self.tree.box_centers,
                    strengths=src_weight_vecs,
                    tgt_expansions=mpoles_view,
                    tgt_base_ibox=level_start_ibox,
                    rscale=self.level_to_rscale(lev),

                    **kwargs)

            assert mpoles_res is mpoles_view

        return mpoles

    def coarsen_multipoles(self,
            actx: PyOpenCLArrayContext,
            level_start_source_parent_box_nrs,
            source_parent_boxes,
            mpoles):
        tree = self.tree
        level_start_source_parent_box_nrs = (
            actx.to_numpy(level_start_source_parent_box_nrs))

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

            source_level_start_ibox, source_mpoles_view = (
                    self.multipole_expansions_view(mpoles, source_level))
            target_level_start_ibox, target_mpoles_view = (
                    self.multipole_expansions_view(mpoles, target_level))

            mpoles_res = m2m(
                    actx,
                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_mpoles_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=source_parent_boxes[start:stop],
                    box_child_ids=self.tree.box_child_ids,
                    centers=self.tree.box_centers,

                    src_rscale=self.level_to_rscale(source_level),
                    tgt_rscale=self.level_to_rscale(target_level),

                    **self.kernel_extra_kwargs)

            assert mpoles_res is target_mpoles_view

        return mpoles

    def eval_direct(self,
            actx: PyOpenCLArrayContext,
            target_boxes, source_box_starts,
            source_box_lists, src_weight_vecs):
        pot = self.output_zeros(actx)

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.self_extra_kwargs)
        kwargs.update(self.box_source_list_kwargs())
        kwargs.update(self.box_target_list_kwargs())

        pot_res = self.tree_indep.p2p()(actx,
                target_boxes=target_boxes,
                source_box_starts=source_box_starts,
                source_box_lists=source_box_lists,
                strength=src_weight_vecs,
                result=pot,
                max_nsources_in_one_box=self.max_nsources_in_one_box,
                max_ntargets_in_one_box=self.max_ntargets_in_one_box,
                **kwargs)

        for pot_i, pot_res_i in zip(pot, pot_res):
            assert pot_i is pot_res_i

        return pot

    @memoize_method
    def multipole_to_local_precompute(self):
        actx = self.tree_indep._setup_actx

        result = []
        m2l_translation_classes_dependent_data = (
            self.m2l_translation_classes_dependent_data_zeros(actx))

        for lev in range(self.tree.nlevels):
            src_rscale = self.level_to_rscale(lev)
            order = self.level_orders[lev]
            precompute_kernel = (
                self.tree_indep.m2l_translation_class_dependent_data_kernel(
                        order, order)
                )

            translation_classes_level_start, \
                m2l_translation_classes_dependent_data_view = \
                    self.m2l_translation_classes_dependent_data_view(
                            m2l_translation_classes_dependent_data, lev)

            ntranslation_classes = (
                    m2l_translation_classes_dependent_data_view.shape[0])

            if ntranslation_classes == 0:
                result.append(actx.np.zeros_like(
                    m2l_translation_classes_dependent_data_view))
                continue

            data = self.translation_classes_data
            m2l_translation_vectors = (
                data.from_sep_siblings_translation_class_to_distance_vector)

            precompute_kernel(
                actx,
                src_rscale=src_rscale,
                translation_classes_level_start=translation_classes_level_start,
                ntranslation_classes=ntranslation_classes,
                m2l_translation_classes_dependent_data=(
                    m2l_translation_classes_dependent_data_view),
                m2l_translation_vectors=m2l_translation_vectors,
                ntranslation_vectors=m2l_translation_vectors.shape[1],
                **self.kernel_extra_kwargs
            )

            if self.tree_indep.m2l_translation.use_fft:
                m2l_translation_classes_dependent_data_view = (
                    self.run_opencl_fft(actx,
                        m2l_translation_classes_dependent_data_view,
                        inverse=False, wait_for=None))
            result.append(m2l_translation_classes_dependent_data_view)

        result = [actx.freeze(arr) for arr in result]
        return result

    def _add_m2l_precompute_kwargs(self, kwargs_for_m2l,
            lev):
        """This method is used for adding the information needed for a
        multipole-to-local translation with precomputation to the keywords
        passed to multipole-to-local translation.
        """
        if not self.supports_translation_classes:
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
            self.translation_classes_data.from_sep_siblings_translation_classes

    def multipole_to_local(self,
            actx: PyOpenCLArrayContext,
            level_start_target_box_nrs,
            target_boxes, src_box_starts, src_box_lists,
            mpole_exps):

        local_exps = self.local_expansion_zeros(actx)
        level_start_target_box_nrs = actx.to_numpy(level_start_target_box_nrs)

        if self.tree_indep.m2l_translation.use_preprocessing:
            preprocessed_mpole_exps = (
                self.m2l_preproc_mpole_expansion_zeros(actx, mpole_exps))
            m2l_work_array = self.m2l_work_array_zeros(actx, local_exps)
            mpole_exps_view_func = self.m2l_preproc_mpole_expansions_view
            local_exps_view_func = self.m2l_work_array_view
        else:
            preprocessed_mpole_exps = mpole_exps
            m2l_work_array = local_exps
            mpole_exps_view_func = self.multipole_expansions_view
            local_exps_view_func = self.local_expansions_view

        for lev in range(self.tree.nlevels):
            wait_for = []

            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            if self.tree_indep.m2l_translation.use_preprocessing:
                order = self.level_orders[lev]
                preprocess_mpole_kernel = \
                    self.tree_indep.m2l_preprocess_mpole_kernel(order, order)

                _, source_mpoles_view = \
                        self.multipole_expansions_view(mpole_exps, lev)

                tr_classes = self.m2l_translation_class_level_start_box_nrs()
                if tr_classes[lev] == tr_classes[lev + 1]:
                    # There is no M2L happening in this level
                    continue

                preprocess_mpole_kernel(
                    actx,
                    src_expansions=source_mpoles_view,
                    preprocessed_src_expansions=preprocessed_mpole_exps[lev],
                    src_rscale=self.level_to_rscale(lev),
                    wait_for=wait_for,
                    **self.kernel_extra_kwargs
                )

                if self.tree_indep.m2l_translation.use_fft:
                    preprocessed_mpole_exps[lev] = \
                        self.run_opencl_fft(actx,
                            preprocessed_mpole_exps[lev],
                            inverse=False, wait_for=wait_for)

            order = self.level_orders[lev]
            m2l = self.tree_indep.m2l(order, order,
                    self.supports_translation_classes)

            source_level_start_ibox, source_mpoles_view = \
                    mpole_exps_view_func(preprocessed_mpole_exps, lev)
            target_level_start_ibox, target_locals_view = \
                    local_exps_view_func(m2l_work_array, lev)

            kwargs = dict(
                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_locals_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=target_boxes[start:stop],
                    src_box_starts=src_box_starts[start:stop+1],
                    src_box_lists=src_box_lists,
                    centers=self.tree.box_centers,

                    src_rscale=self.level_to_rscale(lev),
                    tgt_rscale=self.level_to_rscale(lev),

                    **self.kernel_extra_kwargs)

            self._add_m2l_precompute_kwargs(kwargs, lev)
            if "m2l_translation_classes_dependent_data" in kwargs and \
                    kwargs["m2l_translation_classes_dependent_data"].size == 0:
                # There is nothing to do for this level
                continue
            m2l(actx, **kwargs, wait_for=wait_for)

            if self.tree_indep.m2l_translation.use_preprocessing:
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

                if self.tree_indep.m2l_translation.use_fft:
                    target_locals_before_postprocessing_view = \
                        self.run_opencl_fft(actx,
                            target_locals_before_postprocessing_view,
                            inverse=True, wait_for=wait_for)

                postprocess_local_kernel(
                    actx,
                    tgt_expansions=target_locals_view,
                    tgt_expansions_before_postprocessing=(
                        target_locals_before_postprocessing_view),
                    src_rscale=self.level_to_rscale(lev),
                    tgt_rscale=self.level_to_rscale(lev),
                    wait_for=wait_for,
                    **self.kernel_extra_kwargs,
                )

        return local_exps

    def eval_multipoles(self,
            actx: PyOpenCLArrayContext,
            target_boxes_by_source_level, source_boxes_by_level, mpole_exps):
        pot = self.output_zeros(actx)

        kwargs = self.kernel_extra_kwargs.copy()
        kwargs.update(self.box_target_list_kwargs())

        wait_for = mpole_exps.events
        for isrc_level, ssn in enumerate(source_boxes_by_level):
            if len(target_boxes_by_source_level[isrc_level]) == 0:
                continue

            m2p = self.tree_indep.m2p(self.level_orders[isrc_level])

            source_level_start_ibox, source_mpoles_view = \
                    self.multipole_expansions_view(mpole_exps, isrc_level)

            pot_res = m2p(
                    actx,

                    src_expansions=source_mpoles_view,
                    src_base_ibox=source_level_start_ibox,

                    target_boxes=target_boxes_by_source_level[isrc_level],
                    source_box_starts=ssn.starts,
                    source_box_lists=ssn.lists,
                    centers=self.tree.box_centers,
                    result=pot,

                    rscale=self.level_to_rscale(isrc_level),

                    wait_for=wait_for,

                    **kwargs)

            for pot_i, pot_res_i in zip(pot, pot_res):
                assert pot_i is pot_res_i

        return pot

    def form_locals(self,
            actx: PyOpenCLArrayContext,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weight_vecs):
        local_exps = self.local_expansion_zeros(actx)
        level_start_target_or_target_parent_box_nrs = (
            actx.to_numpy(level_start_target_or_target_parent_box_nrs))

        kwargs = self.extra_kwargs.copy()
        kwargs.update(self.box_source_list_kwargs())

        for lev in range(self.tree.nlevels):
            start, stop = (
                level_start_target_or_target_parent_box_nrs[lev:lev+2])
            if start == stop:
                continue

            p2l = self.tree_indep.p2l(self.level_orders[lev])

            target_level_start_ibox, target_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            result = p2l(
                    actx,
                    target_boxes=target_or_target_parent_boxes[start:stop],
                    source_box_starts=starts[start:stop+1],
                    source_box_lists=lists,
                    centers=self.tree.box_centers,
                    strengths=src_weight_vecs,

                    tgt_expansions=target_local_exps_view,
                    tgt_base_ibox=target_level_start_ibox,

                    rscale=self.level_to_rscale(lev),

                    **kwargs)

            assert result is target_local_exps_view

        return local_exps

    def refine_locals(self,
            actx: PyOpenCLArrayContext,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            local_exps):
        level_start_target_or_target_parent_box_nrs = (
            actx.to_numpy(level_start_target_or_target_parent_box_nrs))

        for target_lev in range(1, self.tree.nlevels):
            start, stop = (
                level_start_target_or_target_parent_box_nrs[target_lev:target_lev+2])
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

            local_exps_res = l2l(actx,
                    src_expansions=source_local_exps_view,
                    src_base_ibox=source_level_start_ibox,
                    tgt_expansions=target_local_exps_view,
                    tgt_base_ibox=target_level_start_ibox,

                    target_boxes=target_or_target_parent_boxes[start:stop],
                    box_parent_ids=self.tree.box_parent_ids,
                    centers=self.tree.box_centers,

                    src_rscale=self.level_to_rscale(source_lev),
                    tgt_rscale=self.level_to_rscale(target_lev),

                    **self.kernel_extra_kwargs)

            assert local_exps_res is target_local_exps_view

        return local_exps

    def eval_locals(self,
            actx: PyOpenCLArrayContext,
            level_start_target_box_nrs, target_boxes, local_exps):
        pot = self.output_zeros(actx)
        level_start_target_box_nrs = actx.to_numpy(level_start_target_box_nrs)

        kwargs = self.kernel_extra_kwargs.copy()
        kwargs.update(self.box_target_list_kwargs())

        for lev in range(self.tree.nlevels):
            start, stop = level_start_target_box_nrs[lev:lev+2]
            if start == stop:
                continue

            l2p = self.tree_indep.l2p(self.level_orders[lev])

            source_level_start_ibox, source_local_exps_view = \
                    self.local_expansions_view(local_exps, lev)

            pot_res = l2p(
                    actx,

                    src_expansions=source_local_exps_view,
                    src_base_ibox=source_level_start_ibox,

                    target_boxes=target_boxes[start:stop],
                    centers=self.tree.box_centers,
                    result=pot,

                    rscale=self.level_to_rscale(lev),

                    **kwargs)

            for pot_i, pot_res_i in zip(pot, pot_res):
                assert pot_i is pot_res_i

        return pot

    def finalize_potentials(self, actx: PyOpenCLArrayContext, potentials):
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
