__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
"""

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

from sumpy.tools import (
        KernelComputation, KernelCacheMixin, is_obj_array_like)


__doc__ = """

Particle-to-particle
--------------------

.. autoclass:: P2PBase
.. autoclass:: P2P
.. autoclass:: P2PMatrixGenerator
.. autoclass:: P2PMatrixSubsetGenerator
.. autoclass:: P2PFromCSR

"""


# LATER:
# - Optimization for source == target (postpone)

# {{{ p2p base class

class P2PBase(KernelCacheMixin, KernelComputation):
    def __init__(self, ctx, target_kernels, exclude_self, strength_usage=None,
            value_dtypes=None, name=None, device=None, source_kernels=None):
        """
        :arg target_kernels: list of :class:`sumpy.kernel.Kernel` instances
          with only target derivatives.
        :arg source_kernels: list of :class:`sumpy.kernel.Kernel` instances
          with only source derivatives.
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """
        from pytools import single_valued
        from sumpy.kernel import (TargetTransformationRemover,
                SourceTransformationRemover)
        txr = TargetTransformationRemover()
        sxr = SourceTransformationRemover()

        if source_kernels is None:
            source_kernels = [single_valued(txr(knl) for knl in target_kernels)]
            target_kernels = [sxr(knl) for knl in target_kernels]
        else:
            for knl in source_kernels:
                assert txr(knl) == knl
            for knl in target_kernels:
                assert sxr(knl) == knl

        base_source_kernel = single_valued(sxr(knl) for knl in source_kernels)
        base_target_kernel = single_valued(txr(knl) for knl in target_kernels)
        assert base_source_kernel == base_target_kernel

        KernelComputation.__init__(self, ctx=ctx, target_kernels=target_kernels,
            source_kernels=source_kernels, strength_usage=strength_usage,
            value_dtypes=value_dtypes, name=name, device=device)

        self.exclude_self = exclude_self

        self.dim = single_valued(knl.dim for knl in
            list(self.target_kernels) + list(self.source_kernels))

    def get_cache_key(self):
        return (type(self).__name__, tuple(self.target_kernels), self.exclude_self,
                tuple(self.strength_usage), tuple(self.value_dtypes),
                tuple(self.source_kernels),
                self.device.hashable_model_and_version_identifier)

    def get_loopy_insns_and_result_names(self):
        from sumpy.symbolic import make_sym_vector
        from pymbolic import var

        dvec = make_sym_vector("d", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        isrc_sym = var("isrc")

        exprs = []
        for out_knl in self.target_kernels:
            expr_sum = 0
            for j, in_knl in enumerate(self.source_kernels):
                expr = in_knl.postprocess_at_source(
                            in_knl.get_expression(dvec),
                            dvec)
                expr *= self.get_strength_or_not(isrc_sym, j)
                expr_sum += expr
            expr_sum = out_knl.postprocess_at_target(expr_sum, dvec)
            exprs.append(expr_sum)

        if self.exclude_self:
            result_name_prefix = "pair_result_tmp"
        else:
            result_name_prefix = "pair_result"

        result_names = [
            sac.add_assignment(f"{result_name_prefix}_{i}", expr)
            for i, expr in enumerate(exprs)
        ]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(sac.assignments.items(),
                vector_names={"d"},
                pymbolic_expr_maps=(
                    [knl.get_code_transformer() for knl in self.source_kernels]
                    + [knl.get_code_transformer() for knl in self.target_kernels]),
                retain_names=result_names,
                )

        from pymbolic import var
        if self.exclude_self:
            assignments = [lp.Assignment(id=None,
                    assignee=f"pair_result_{i}", expression=var(name),
                    temp_var_type=lp.Optional(None))
                for i, name in enumerate(result_names)]

            from pymbolic.primitives import If, Variable
            assignments = [assign.copy(expression=If(Variable("is_self"), 0,
                            assign.expression)) for assign in assignments]
        else:
            assignments = []

        return assignments + loopy_insns, result_names

    def get_strength_or_not(self, isrc, kernel_idx):
        from sumpy.symbolic import Symbol
        return Symbol(f"strength_{self.strength_usage[kernel_idx]}")

    def get_default_src_tgt_arguments(self):
        from sumpy.tools import gather_loopy_source_arguments
        return ([
                lp.GlobalArg("sources", None,
                    shape=(self.dim, "nsources")),
                lp.GlobalArg("targets", None,
                   shape=(self.dim, "ntargets")),
                lp.ValueArg("nsources", None),
                lp.ValueArg("ntargets", None)]
                + ([lp.GlobalArg("target_to_source", None, shape=("ntargets",))]
                    if self.exclude_self else [])
                + gather_loopy_source_arguments(self.source_kernels))

    def get_optimized_kernel(self, targets_is_obj_array, sources_is_obj_array):
        # FIXME
        knl = self.get_kernel()

        if sources_is_obj_array:
            knl = lp.tag_array_axes(knl, "sources", "sep,C")
        if targets_is_obj_array:
            knl = lp.tag_array_axes(knl, "targets", "sep,C")

        knl = lp.split_iname(knl, "itgt", 1024, outer_tag="g.0")
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

        return knl


# }}}


# {{{ P2P point-interaction calculation

class P2P(P2PBase):
    """Direct applier for P2P interactions."""

    @property
    def default_name(self):
        return "p2p_apply"

    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        arguments = (
            self.get_default_src_tgt_arguments()
            + [
                lp.GlobalArg("strength", None,
                    shape="nstrengths, nsources", dim_tags="sep,C"),
                lp.GlobalArg("result", None,
                    shape="nresults, ntargets", dim_tags="sep,C")
            ])

        loopy_knl = lp.make_kernel(["""
            {[itgt, isrc, idim]: \
                0 <= itgt < ntargets and \
                0 <= isrc < nsources and \
                0 <= idim < dim}
            """],
            self.get_kernel_scaling_assignments()
            + ["for itgt, isrc"]
            + ["<> d[idim] = targets[idim, itgt] - sources[idim, isrc]"]
            + ["<> is_self = (isrc == target_to_source[itgt])"
                if self.exclude_self else ""]
            + [f"<> strength_{i} = strength[{i}, isrc]" for
                i in set(self.strength_usage)]
            + loopy_insns
            + [f"""
                result[{iknl}, itgt] = knl_{iknl}_scaling * \
                    simul_reduce(sum, isrc, pair_result_{iknl}) {{inames=itgt}}
               """ for iknl in range(len(self.target_kernels))]
            + ["end"],
            arguments,
            assumptions="nsources>=1 and ntargets>=1",
            name=self.name,
            default_offset=lp.auto,
            fixed_parameters=dict(
                dim=self.dim,
                nstrengths=self.strength_count,
                nresults=len(self.target_kernels)),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        for knl in self.target_kernels + self.source_kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self, queue, targets, sources, strength, **kwargs):
        knl = self.get_cached_optimized_kernel(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources))

        return knl(queue, sources=sources, targets=targets, strength=strength,
                **kwargs)

# }}}


# {{{ P2P matrix writer

class P2PMatrixGenerator(P2PBase):
    """Generator for P2P interaction matrix entries."""

    @property
    def default_name(self):
        return "p2p_matrix"

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        arguments = (
            self.get_default_src_tgt_arguments()
            + [
                lp.GlobalArg(f"result_{i}", dtype, shape="ntargets,nsources")
                for i, dtype in enumerate(self.value_dtypes)
            ])

        loopy_knl = lp.make_kernel(["""
            {[itgt, isrc, idim]: \
                0 <= itgt < ntargets and \
                0 <= isrc < nsources and \
                0 <= idim < dim}
            """],
            self.get_kernel_scaling_assignments()
            + ["for itgt, isrc"]
            + ["<> d[idim] = targets[idim, itgt] - sources[idim, isrc]"]
            + ["<> is_self = (isrc == target_to_source[itgt])"
                if self.exclude_self else ""]
            + loopy_insns
            + [f"""
                result_{iknl}[itgt, isrc] = knl_{iknl}_scaling \
                    * pair_result_{iknl} {{inames=isrc:itgt}}
                """ for iknl in range(len(self.target_kernels))]
            + ["end"],
            arguments,
            assumptions="nsources>=1 and ntargets>=1",
            name=self.name,
            fixed_parameters=dict(dim=self.dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        for knl in self.target_kernels + self.source_kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self, queue, targets, sources, **kwargs):
        knl = self.get_cached_optimized_kernel(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources))

        return knl(queue, sources=sources, targets=targets, **kwargs)

# }}}


# {{{ P2P matrix subset generator

class P2PMatrixSubsetGenerator(P2PBase):
    """Generator for a subset of P2P interaction matrix entries.

    This generator evaluates a generic set of entries in the matrix. See
    :class:`P2PFromCSR` for when a compressed row storage format is available.

    .. automethod:: __call__
    """

    @property
    def default_name(self):
        return "p2p_subset"

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        arguments = (
            self.get_default_src_tgt_arguments()
            + [
                lp.GlobalArg("srcindices", None, shape="nresult"),
                lp.GlobalArg("tgtindices", None, shape="nresult"),
                lp.ValueArg("nresult", None)
            ]
            + [
                lp.GlobalArg(f"result_{i}", dtype, shape="nresult")
                for i, dtype in enumerate(self.value_dtypes)
            ])

        loopy_knl = lp.make_kernel(
            "{[imat, idim]: 0 <= imat < nresult and 0 <= idim < dim}",
            self.get_kernel_scaling_assignments()
            # NOTE: itgt, isrc need to always be defined in case a statement
            # in loopy_insns or kernel_exprs needs them (e.g. hardcoded in
            # places like get_kernel_exprs)
            + ["""
                for imat
                    <> itgt = tgtindices[imat]
                    <> isrc = srcindices[imat]

                    <> d[idim] = targets[idim, itgt] - sources[idim, isrc]
            """]
            + ["""
                    <> is_self = (isrc == target_to_source[itgt])
                """ if self.exclude_self else ""]
            + loopy_insns
            + [f"""
                    result_{iknl}[imat] = \
                        knl_{iknl}_scaling * pair_result_{iknl} \
                            {{id_prefix=write_p2p}}
                """ for iknl in range(len(self.target_kernels))]
            + ["end"],
            arguments,
            assumptions="nresult>=1",
            silenced_warnings="write_race(write_p2p*)",
            name=self.name,
            fixed_parameters=dict(dim=self.dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.add_dtypes(loopy_knl,
            dict(nsources=np.int32, ntargets=np.int32))

        for knl in self.target_kernels + self.source_kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self, targets_is_obj_array, sources_is_obj_array):
        # FIXME
        knl = self.get_kernel()

        if sources_is_obj_array:
            knl = lp.tag_array_axes(knl, "sources", "sep,C")
        if targets_is_obj_array:
            knl = lp.tag_array_axes(knl, "targets", "sep,C")

        knl = lp.split_iname(knl, "imat", 1024, outer_tag="g.0")
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")
        return knl

    def __call__(self, queue, targets, sources, tgtindices, srcindices, **kwargs):
        """Evaluate a subset of the P2P matrix interactions.

        :arg targets: target point coordinates, which can be an object
            :class:`~numpy.ndarray`, :class:`list` or :class:`tuple` of
            coordinates or a single stacked array.
        :arg sources: source point coordinates, which can also be in any of the
            formats of the *targets*,

        :arg srcindices: an array of indices into *sources*.
        :arg tgtindices: an array of indices into *targets*, of the same size
            as *srcindices*.

        :returns: a one-dimensional array of interactions, for each index pair
            in (*srcindices*, *tgtindices*)
        """
        knl = self.get_cached_optimized_kernel(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources))

        return knl(queue,
                   targets=targets,
                   sources=sources,
                   tgtindices=tgtindices,
                   srcindices=srcindices, **kwargs)

# }}}


# {{{ P2P from CSR-like interaction list

class P2PFromCSR(P2PBase):
    @property
    def default_name(self):
        return "p2p_from_csr"

    def get_kernel(self, max_nsources_in_one_box, max_ntargets_in_one_box,
            gpu=False, nsplit=32):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        arguments = self.get_default_src_tgt_arguments() \
            + [
                lp.GlobalArg("box_target_starts",
                    None, shape=None),
                lp.GlobalArg("box_target_counts_nonchild",
                    None, shape=None),
                lp.GlobalArg("box_source_starts",
                    None, shape=None),
                lp.GlobalArg("box_source_counts_nonchild",
                    None, shape=None),
                lp.GlobalArg("source_box_starts",
                    None, shape=None),
                lp.GlobalArg("source_box_lists",
                    None, shape=None),
                lp.GlobalArg("strength", None,
                    shape="nstrengths, nsources", dim_tags="sep,C"),
                lp.GlobalArg("result", None,
                    shape="noutputs, ntargets", dim_tags="sep,C"),
                lp.TemporaryVariable("tgt_center", shape=(self.dim,)),
                "..."
            ]

        domains = [
            "{[itgt_box]: 0 <= itgt_box < ntgt_boxes}",
            "{[iknl]: 0 <= iknl < noutputs}",
            "{[isrc_box]: isrc_box_start <= isrc_box < isrc_box_end}",
            "{[idim]: 0 <= idim < dim}",
            "{[isrc]: isrc_start <= isrc < isrc_end}"
        ]

        src_outer_limit = (max_nsources_in_one_box - 1) // nsplit
        tgt_outer_limit = (max_ntargets_in_one_box - 1) // nsplit

        if gpu:
            arguments += [
                lp.TemporaryVariable("local_isrc",
                    shape=(self.dim, max_nsources_in_one_box)),
                lp.TemporaryVariable("local_isrc_strength",
                    shape=(self.strength_count, max_nsources_in_one_box)),
            ]
            domains += [
                "{[istrength]: 0 <= istrength < nstrengths}",
                "{[inner]: 0 <= inner < nsplit}",
                "{[itgt_offset_outer]: 0 <= itgt_offset_outer <= tgt_outer_limit}",
                "{[isrc_offset_outer]: 0 <= isrc_offset_outer <= src_outer_limit}",
            ]
        else:
            domains += [
                "{[itgt]: itgt_start <= itgt < itgt_end}",
            ]

        # There are two algorithms here because pocl-pthread 1.9 miscompiles
        # the "gpu" kernel with prefetching.
        if gpu:
            instructions = (self.get_kernel_scaling_assignments()
              + ["""
                for itgt_box
                <> tgt_ibox = target_boxes[itgt_box]
                <> itgt_start = box_target_starts[tgt_ibox]
                <> itgt_end = itgt_start + box_target_counts_nonchild[tgt_ibox]

                <> isrc_box_start = source_box_starts[itgt_box]
                <> isrc_box_end = source_box_starts[itgt_box+1]

                for itgt_offset_outer
                  <> itgt_offset = itgt_offset_outer * nsplit + inner
                  <> itgt = itgt_offset + itgt_start
                  <> cond_itgt = itgt < itgt_end
                  <> acc[iknl] = 0 {id=init_acc}
                  if cond_itgt
                    tgt_center[idim] = targets[idim, itgt] {id=prefetch_tgt,dup=idim}
                  end
                  for isrc_box
                    <> src_ibox = source_box_lists[isrc_box]  {id=src_box_insn_0}
                    <> isrc_start = box_source_starts[src_ibox]  {id=src_box_insn_1}
                    <> isrc_end = isrc_start + box_source_counts_nonchild[src_ibox] \
                        {id=src_box_insn_2}
                    for isrc_offset_outer
                      <> isrc_offset = isrc_offset_outer * nsplit + inner
                      <> cond_isrc = isrc_offset < isrc_end - isrc_start
                      if cond_isrc
                        local_isrc[idim, isrc_offset] = sources[idim,
                          isrc_offset + isrc_start]  {id=prefetch_src, dup=idim}
                        local_isrc_strength[istrength, isrc_offset] = strength[
                          istrength, isrc_offset + isrc_start]  {id=prefetch_charge}
                      end
                    end
                    if cond_itgt
                      for isrc
                        <> d[idim] = (tgt_center[idim] - local_isrc[idim,
                          isrc - isrc_start]) {dep=prefetch_src:prefetch_tgt}
              """] + ["""
                        <> is_self = (isrc == target_to_source[itgt])
                    """ if self.exclude_self else ""]
              + [f"""
                <> strength_{i} = local_isrc_strength[{i}, isrc - isrc_start] \
                   {{dep=prefetch_charge}}
                """ for
                i in set(self.strength_usage)]
              + loopy_insns
              + [f"""
                        acc[{iknl}] = acc[{iknl}] + \
                          pair_result_{iknl} \
                          {{id=update_acc_{iknl}, dep=init_acc}}
                """ for iknl in range(len(self.target_kernels))]
              + ["""
                      end
                    end
                  end
                 """]
              + [f"""
                  if cond_itgt
                    result[{iknl}, itgt] = knl_{iknl}_scaling * acc[{iknl}] \
                            {{id_prefix=write_csr,dep=update_acc_{iknl} }}
                  end
                """ for iknl in range(len(self.target_kernels))]
              + ["""
                end
                end
              """])
        else:
            instructions = (self.get_kernel_scaling_assignments()
              + ["""
                for itgt_box
                <> tgt_ibox = target_boxes[itgt_box]
                <> itgt_start = box_target_starts[tgt_ibox]
                <> itgt_end = itgt_start + box_target_counts_nonchild[tgt_ibox]

                <> isrc_box_start = source_box_starts[itgt_box]
                <> isrc_box_end = source_box_starts[itgt_box+1]

                for itgt
                  <> acc[iknl] = 0 {id=init_acc}
                  tgt_center[idim] = targets[idim, itgt] {id=prefetch_tgt,dup=idim}
                  for isrc_box
                    <> src_ibox = source_box_lists[isrc_box]  {id=src_box_insn_0}
                    <> isrc_start = box_source_starts[src_ibox]  {id=src_box_insn_1}
                    <> isrc_end = isrc_start + box_source_counts_nonchild[src_ibox] \
                            {id=src_box_insn_2}
                    for isrc
                        <> d[idim] = (tgt_center[idim] - sources[idim,
                          isrc]) {dep=prefetch_tgt}
              """] + ["""
                        <> is_self = (isrc == target_to_source[itgt])
                    """ if self.exclude_self else ""]
              + [f"<> strength_{i} = strength[{i}, isrc]" for
                i in set(self.strength_usage)]
              + loopy_insns
              + [f"""
                        acc[{iknl}] = acc[{iknl}] + \
                          pair_result_{iknl} \
                          {{id=update_acc_{iknl}, dep=init_acc}}
                """ for iknl in range(len(self.target_kernels))]
              + ["""
                    end
                  end
               """]
              + [f"""
                  result[{iknl}, itgt] = knl_{iknl}_scaling * acc[{iknl}] \
                          {{id_prefix=write_csr,dep=update_acc_{iknl} }}
                """ for iknl in range(len(self.target_kernels))]
              + ["""
                end
                end
              """])

        loopy_knl = lp.make_kernel(
            domains,
            instructions,
            arguments,
            assumptions="ntgt_boxes>=1",
            name=self.name,
            silenced_warnings=["write_race(write_csr*)", "write_race(prefetch_src)",
                "write_race(prefetch_charge)"],
            fixed_parameters=dict(
                dim=self.dim,
                nstrengths=self.strength_count,
                nsplit=nsplit,
                src_outer_limit=src_outer_limit,
                tgt_outer_limit=tgt_outer_limit,
                noutputs=len(self.target_kernels)),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.add_dtypes(loopy_knl,
            dict(nsources=np.int32, ntargets=np.int32))

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "istrength*:unr")
        loopy_knl = lp.tag_array_axes(loopy_knl, "targets", "sep,C")
        loopy_knl = lp.tag_array_axes(loopy_knl, "sources", "sep,C")

        for knl in self.target_kernels + self.source_kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self, max_nsources_in_one_box,
            max_ntargets_in_one_box):
        import pyopencl as cl
        dev = self.context.devices[0]
        if dev.type & cl.device_type.CPU:
            knl = self.get_kernel(max_nsources_in_one_box,
                    max_ntargets_in_one_box, gpu=False)
            knl = lp.split_iname(knl, "itgt_box", 4, outer_tag="g.0")
        else:
            knl = self.get_kernel(max_nsources_in_one_box,
                    max_ntargets_in_one_box, gpu=True, nsplit=32)
            knl = lp.tag_inames(knl, {"itgt_box": "g.0", "inner": "l.0"})
            knl = lp.set_temporary_address_space(knl,
                ["local_isrc", "local_isrc_strength"], lp.AddressSpace.LOCAL)
            knl = lp.add_inames_for_unused_hw_axes(knl)
            # knl = lp.set_options(knl, write_code=True)

        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl,
                enforce_variable_access_ordered="no_check")

        return knl

    def __call__(self, queue, **kwargs):
        max_nsources_in_one_box = kwargs.pop("max_nsources_in_one_box")
        max_ntargets_in_one_box = kwargs.pop("max_ntargets_in_one_box")
        knl = self.get_cached_optimized_kernel(
                max_nsources_in_one_box=max_nsources_in_one_box,
                max_ntargets_in_one_box=max_ntargets_in_one_box)

        return knl(queue, **kwargs)

# }}}

# vim: foldmethod=marker
