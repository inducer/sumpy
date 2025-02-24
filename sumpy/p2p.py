from __future__ import annotations


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

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import override

import loopy as lp
import pytools.obj_array as obj_array

from sumpy.array_context import PyOpenCLArrayContext, make_loopy_program
from sumpy.codegen import register_optimization_preambles
from sumpy.tools import KernelCacheMixin, KernelComputation, is_obj_array_like


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyopencl as cl
    from arraycontext import Array
    from pytools.obj_array import ObjectArray1D


logger = logging.getLogger(__name__)

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

# {{{ P2PBase: base class

class P2PBase(KernelCacheMixin, KernelComputation):
    def __init__(self, target_kernels, exclude_self, strength_usage=None,
            value_dtypes=None, name=None, source_kernels=None):
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

        from sumpy.kernel import (
            SourceTransformationRemover,
            TargetTransformationRemover,
        )
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

        KernelComputation.__init__(self, target_kernels=target_kernels,
            source_kernels=source_kernels, strength_usage=strength_usage,
            value_dtypes=value_dtypes, name=name)

        self.exclude_self = exclude_self
        self.dim = single_valued([
            knl.dim for knl in self.target_kernels + self.source_kernels
            ])

    def get_cache_key(self):
        return (type(self).__name__, tuple(self.target_kernels), self.exclude_self,
                tuple(self.strength_usage), tuple(self.value_dtypes),
                tuple(self.source_kernels))

    def get_loopy_insns_and_result_names(self):
        from pymbolic import var

        from sumpy.symbolic import make_sym_vector

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

        result_name_prefix = "pair_result_tmp" if self.exclude_self else "pair_result"
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

    def get_optimized_kernel(self, *,
                             targets_is_obj_array: bool = False,
                             sources_is_obj_array: bool = False,
                             **kwargs: Any) -> lp.TranslationUnit:
        # FIXME
        knl = self.get_kernel()

        if sources_is_obj_array:
            knl = lp.tag_array_axes(knl, "sources", "sep,C")
        if targets_is_obj_array:
            knl = lp.tag_array_axes(knl, "targets", "sep,C")

        knl = lp.split_iname(knl, "itgt", 1024, outer_tag="g.0")
        knl = self._allow_redundant_execution_of_knl_scaling(knl)
        knl = lp.set_options(knl, enforce_variable_access_ordered="no_check")

        return knl

# }}}


# {{{ P2P: point-interaction calculation

class P2P(P2PBase):
    """Direct applier for P2P interactions."""

    @property
    def default_name(self):
        return "p2p_apply"

    def get_kernel(self):
        loopy_insns, _result_names = self.get_loopy_insns_and_result_names()
        arguments = [
                *self.get_default_src_tgt_arguments(),
                lp.GlobalArg("strength", None,
                    shape="nstrengths, nsources", dim_tags="sep,C"),
                lp.GlobalArg("result", None,
                    shape="nresults, ntargets", dim_tags="sep,C")
            ]

        loopy_knl = make_loopy_program(["""
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
            kernel_data=arguments,
            assumptions="nsources>=1 and ntargets>=1",
            name=self.name,
            fixed_parameters={
                "dim": self.dim,
                "nstrengths": self.strength_count,
                "nresults": len(self.target_kernels)},
            )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        for knl in [*self.target_kernels, *self.source_kernels]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self,
            actx: PyOpenCLArrayContext,
            targets: ObjectArray1D[Array] | Array,
            sources: ObjectArray1D[Array] | Array,
            strength: Sequence[Array],
            **kwargs: Any,
        ) -> tuple[cl.Event, Sequence[Array]]:
        knl = self.get_cached_kernel_executor(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources))

        from sumpy.codegen import register_optimization_preambles
        knl = register_optimization_preambles(knl, actx.queue.device)

        result = actx.call_loopy(
            knl,
            sources=sources,
            targets=targets,
            strength=strength,
            **kwargs)

        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}


# {{{ P2PMatrixGenerator: matrix writer

class P2PMatrixGenerator(P2PBase):
    """Generator for P2P interaction matrix entries."""

    @property
    def default_name(self):
        return "p2p_matrix"

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    def get_kernel(self):
        loopy_insns, _result_names = self.get_loopy_insns_and_result_names()
        arguments = (
            self.get_default_src_tgt_arguments()
            + [
                lp.GlobalArg(f"result_{i}", dtype, shape="ntargets,nsources")
                for i, dtype in enumerate(self.value_dtypes)
            ])

        loopy_knl = make_loopy_program(["""
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
            fixed_parameters={"dim": self.dim},
            )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        for knl in [*self.target_kernels, *self.source_kernels]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self,
            actx: PyOpenCLArrayContext,
            targets: ObjectArray1D[Array] | Array,
            sources: ObjectArray1D[Array] | Array,
            **kwargs: Any,
        ) -> Sequence[Array]:
        knl = self.get_cached_kernel_executor(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources))

        from sumpy.codegen import register_optimization_preambles
        knl = register_optimization_preambles(knl, actx.queue.device)

        result = actx.call_loopy(knl, sources=sources, targets=targets, **kwargs)
        return obj_array.new_1d([result[f"result_{i}"] for i in range(self.nresults)])

# }}}


# {{{ P2PMatrixSubsetGenerator: matrix subset generator

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
        loopy_insns, _result_names = self.get_loopy_insns_and_result_names()
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

        loopy_knl = make_loopy_program(
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
            fixed_parameters={"dim": self.dim},
            )

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.add_dtypes(
                loopy_knl, {"nsources": np.int32, "ntargets": np.int32})

        for knl in [*self.target_kernels, *self.source_kernels]:
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
        knl = register_optimization_preambles(knl, self.device)

        return knl

    def __call__(self,
            actx: PyOpenCLArrayContext,
            targets: ObjectArray1D[Array] | Array,
            sources: ObjectArray1D[Array] | Array,
            *,
            tgtindices: Array,
            srcindices: Array,
            **kwargs: Any,
        ) -> tuple[cl.Event, Sequence[Array]]:
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
        knl = self.get_cached_kernel_executor(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources))

        from sumpy.codegen import register_optimization_preambles
        knl = register_optimization_preambles(knl, actx.queue.device)

        result = actx.call_loopy(
            knl,
            targets=targets,
            sources=sources,
            tgtindices=tgtindices,
            srcindices=srcindices, **kwargs)

        return obj_array.new_1d([result[f"result_{i}"] for i in range(self.nresults)])

# }}}


# {{{ P2PFromCSR: P2P from CSR-like interaction list

class P2PFromCSR(P2PBase):
    @property
    @override
    def default_name(self):
        return "p2p_from_csr"

    @override
    def get_kernel(self, *,
            max_nsources_in_one_box: int = 32,
            max_ntargets_in_one_box: int = 32,
            work_items_per_group: int = 32,
            is_gpu: bool = False, **kwargs: Any) -> lp.TranslationUnit:
        loopy_insns, _result_names = self.get_loopy_insns_and_result_names()
        arguments = [
                *self.get_default_src_tgt_arguments(),
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
                ...
            ]

        domains = [
            "{[itgt_box]: 0 <= itgt_box < ntgt_boxes}",
            "{[iknl]: 0 <= iknl < noutputs}",
            "{[isrc_box]: isrc_box_start <= isrc_box < isrc_box_end}",
            "{[idim]: 0 <= idim < dim}",
        ]

        tgt_outer_limit = (max_ntargets_in_one_box - 1) // work_items_per_group

        if is_gpu:
            arguments += [
                lp.TemporaryVariable("local_isrc",
                    shape=(self.dim, max_nsources_in_one_box)),
                lp.TemporaryVariable("local_isrc_strength",
                    shape=(self.strength_count, max_nsources_in_one_box)),
            ]
            domains += [
                "{[istrength]: 0 <= istrength < nstrengths}",
                "{[inner]: 0 <= inner < work_items_per_group}",
                "{[itgt_offset_outer]: 0 <= itgt_offset_outer <= tgt_outer_limit}",
                "{[isrc_prefetch]: 0 <= isrc_prefetch < max_nsources_in_one_box}",
                "{[isrc_offset]: 0 <= isrc_offset < max_nsources_in_one_box"
                " and isrc_offset < isrc_end - isrc_start}",
            ]
        else:
            domains += [
                "{[itgt]: itgt_start <= itgt < itgt_end}",
                "{[isrc]: isrc_start <= isrc < isrc_end}"
            ]

        # There are two algorithms here because pocl-pthread 1.9 miscompiles
        # the "gpu" kernel with prefetching.
        if is_gpu:
            instructions = (self.get_kernel_scaling_assignments()
              + ["""
                for itgt_box
                <> tgt_ibox = target_boxes[itgt_box]  {id=init_0}
                <> itgt_start = box_target_starts[tgt_ibox]  {id=init_1}
                <> itgt_end = itgt_start + box_target_counts_nonchild[tgt_ibox] \
                        {id=init_2}
                <> isrc_box_start = source_box_starts[itgt_box]  {id=init_3}
                <> isrc_box_end = source_box_starts[itgt_box+1]  {id=init_4}

                for itgt_offset_outer
                  for inner
                    <> itgt_offset = itgt_offset_outer * work_items_per_group + inner
                    <> itgt = itgt_offset + itgt_start
                    <> cond_itgt = itgt < itgt_end
                    <> acc[iknl] = 0  {id=init_acc}
                    if cond_itgt
                      tgt_center[idim] = targets[idim, itgt] {id=set_tgt,dup=idim}
                    end
                  end
                  for isrc_box
                    <> src_ibox = source_box_lists[isrc_box]  {id=src_box_insn_0}
                    <> isrc_start = box_source_starts[src_ibox]  {id=src_box_insn_1}
                    <> isrc_end = isrc_start + box_source_counts_nonchild[src_ibox] \
                        {id=src_box_insn_2}
                    for isrc_prefetch
                      <> cond_isrc_prefetch = isrc_prefetch < isrc_end - isrc_start \
                              {id=cond_isrc_prefetch}
                      if cond_isrc_prefetch
                        local_isrc[idim, isrc_prefetch] = sources[idim,
                          isrc_prefetch + isrc_start]  {id=prefetch_src, dup=idim}
                        local_isrc_strength[istrength, isrc_prefetch] = strength[
                          istrength, isrc_prefetch + isrc_start] {id=prefetch_charge}
                      end
                    end
                    for inner
                      if cond_itgt
                        for isrc_offset
                          <> isrc = isrc_offset + isrc_start
                          <> d[idim] = (tgt_center[idim] - local_isrc[idim,
                            isrc_offset]) \
                            {id=set_d,dep=prefetch_src:set_tgt}
              """] + ["""
                          <> is_self = (isrc == target_to_source[itgt])
                    """ if self.exclude_self else ""]
              + [f"""
                          <> strength_{i} = local_isrc_strength[{i}, isrc_offset] \
                            {{id=set_strength{i},dep=prefetch_charge}}
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
                  end
                 """]
              + [f"""
                  for inner
                  if cond_itgt
                    result[{iknl}, itgt] = knl_{iknl}_scaling * acc[{iknl}] \
                            {{id_prefix=write_csr,dep=update_acc_{iknl} }}
                  end
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

        loopy_knl = make_loopy_program(
            domains,
            instructions,
            kernel_data=arguments,
            assumptions="ntgt_boxes>=1",
            name=self.name,
            silenced_warnings=[
                "write_race(write_csr*)",
                "write_race(prefetch_src)",
                "write_race(prefetch_charge)"],
            fixed_parameters={
                "dim": self.dim,
                "nstrengths": self.strength_count,
                "max_nsources_in_one_box": max_nsources_in_one_box,
                "max_ntargets_in_one_box": max_ntargets_in_one_box,
                "work_items_per_group": work_items_per_group,
                "tgt_outer_limit": tgt_outer_limit,
                "noutputs": len(self.target_kernels)},
            )

        loopy_knl = lp.add_dtypes(loopy_knl, {
            "nsources": np.dtype(np.int32),
            "ntargets": np.dtype(np.int32),
            })

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.tag_inames(loopy_knl, "istrength*:unr")
        loopy_knl = lp.tag_array_axes(loopy_knl, "targets", "sep,C")
        loopy_knl = lp.tag_array_axes(loopy_knl, "sources", "sep,C")

        for knl in [*self.target_kernels, *self.source_kernels]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    @override
    def get_optimized_kernel(self, *,
            max_nsources_in_one_box: int = 32,
            max_ntargets_in_one_box: int = 32,
            strength_dtype: np.dtype[Any] | None = None,
            source_dtype: np.dtype[Any] | None = None,
            local_mem_size: int = 32,
            is_gpu: bool = False, **kwargs) -> lp.TranslationUnit:
        if not is_gpu:
            knl = self.get_kernel(
                    max_nsources_in_one_box=max_nsources_in_one_box,
                    max_ntargets_in_one_box=max_ntargets_in_one_box,
                    is_gpu=is_gpu)
            knl = lp.split_iname(knl, "itgt_box", 4, outer_tag="g.0")
            knl = self._allow_redundant_execution_of_knl_scaling(knl)
        else:
            assert strength_dtype is not None
            assert source_dtype is not None

            dtype_size = np.dtype(strength_dtype).alignment
            work_items_per_group = min(256, max_ntargets_in_one_box)
            total_local_mem = max_nsources_in_one_box * \
                    (self.dim + self.strength_count) * dtype_size
            # multiplying by 2 here to make sure at least 2 work groups
            # can be scheduled at the same time for latency hiding
            nprefetch = (2 * total_local_mem - 1) // local_mem_size + 1

            knl = self.get_kernel(
                    max_nsources_in_one_box=max_nsources_in_one_box,
                    max_ntargets_in_one_box=max_ntargets_in_one_box,
                    work_items_per_group=work_items_per_group,
                    is_gpu=is_gpu)
            knl = lp.tag_inames(knl, {"itgt_box": "g.0", "inner": "l.0"})
            knl = lp.set_temporary_address_space(knl,
                ["local_isrc", "local_isrc_strength"], lp.AddressSpace.LOCAL)

            local_arrays = ["local_isrc", "local_isrc_strength"]
            local_array_isrc_axis = [1, 1]
            local_array_sizes = [self.dim, self.strength_count]
            local_array_dtypes = [source_dtype, strength_dtype]
            # By having a concatenated memory layout of the temporaries
            # and marking the first axis as vec, we are transposing the
            # the arrays and also making the access of the source
            # co-ordinates and the strength for each source a coalesced
            # access of 256 bits (assuming double precision) which is
            # optimized for NVIDIA GPUs. On an NVIDIA Titan V, this
            # optimization led to a 8% speedup in the performance.
            if strength_dtype == source_dtype:
                knl = lp.concatenate_arrays(knl, local_arrays, "local_isrc")
                local_arrays = ["local_isrc"]
                local_array_sizes = [self.dim + self.strength_count]
                local_array_dtypes = [source_dtype]
            # We try to mark the local arrays (sources, strengths)
            # as vec for the first dimension
            for i, (array_name, array_size, array_dtype) in \
                    enumerate(zip(local_arrays, local_array_sizes,
                                  local_array_dtypes, strict=True)):
                if issubclass(array_dtype.type, np.complexfloating):
                    # pyopencl does not support complex data type vectors
                    continue
                if array_size in [2, 3, 4, 8, 16]:
                    knl = lp.tag_array_axes(knl, array_name, "vec,C")
                else:
                    # FIXME: check if CUDA
                    n = 16 // dtype_size
                    if n in [1, 2, 4, 8]:
                        knl = lp.split_array_axis(knl, array_name, 0, n)
                        knl = lp.tag_array_axes(knl, array_name, "C,vec,C")
                        local_array_isrc_axis[i] = 2

            # We need to split isrc_prefetch and isrc_offset into chunks.
            nsources = (max_nsources_in_one_box + nprefetch - 1) // nprefetch
            for local_array, axis in zip(local_arrays, local_array_isrc_axis,
                                         strict=True):
                knl = lp.split_array_axis(knl, local_array, axis, nsources)
            knl = lp.split_iname(knl, "isrc_prefetch", nsources,
                    outer_iname="iprefetch")
            knl = lp.split_iname(knl, "isrc_prefetch_inner", work_items_per_group)
            knl = lp.tag_inames(knl, {"isrc_prefetch_inner_inner": "l.0"})
            knl = lp.split_iname(knl, "isrc_offset", nsources,
                    outer_iname="iprefetch")

            # After splitting, the temporary array local_isrc need not
            # be as large as before. Need to simplify before unprivatizing
            knl = lp.simplify_indices(knl)
            knl = lp.unprivatize_temporaries_with_inames(knl,
                    "iprefetch", only_var_names=frozenset(local_arrays))

            knl = lp.add_inames_to_insn(knl,
                    "inner", "id:init_* or id:*_scaling or id:src_box_insn_*")
            knl = lp.add_inames_to_insn(knl, "itgt_box", "id:*_scaling")

        knl = lp.set_options(knl, enforce_variable_access_ordered="no_check")
        return knl

    def __call__(self,
            actx: PyOpenCLArrayContext,
            targets: ObjectArray1D[Array] | Array,
            sources: ObjectArray1D[Array] | Array,
            *,
            max_nsources_in_one_box: int,
            max_ntargets_in_one_box: int,
            **kwargs: Any,
        ) -> tuple[cl.Event, Sequence[Array]]:
        from sumpy.array_context import is_cl_cpu

        is_gpu = not is_cl_cpu(actx)
        if is_gpu:
            source_dtype = kwargs["sources"][0].dtype
            strength_dtype = kwargs["strength"].dtype
        else:
            # these are unused for not GPU and defeats the caching
            # set them to None to keep the caching across dtypes
            source_dtype = None
            strength_dtype = None

        knl = self.get_cached_kernel_executor(
                max_nsources_in_one_box=max_nsources_in_one_box,
                max_ntargets_in_one_box=max_ntargets_in_one_box,
                local_mem_size=actx.queue.device.local_mem_size,
                is_gpu=is_gpu,
                source_dtype=source_dtype,
                strength_dtype=strength_dtype,
            )

        from sumpy.codegen import register_optimization_preambles
        knl = register_optimization_preambles(knl, actx.queue.device)

        result = actx.call_loopy(knl, targets=targets, sources=sources, **kwargs)
        return obj_array.new_1d([result[f"result_s{i}"] for i in range(self.nresults)])

# }}}

# vim: foldmethod=marker
