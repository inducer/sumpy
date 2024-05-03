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

from abc import ABC, abstractmethod

import numpy as np
import loopy as lp

from pytools import memoize_method
from sumpy.array_context import PyOpenCLArrayContext, make_loopy_program
from sumpy.codegen import register_optimization_preambles
from sumpy.tools import KernelCacheMixin, to_complex_dtype

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


# {{{ E2EBase: base class

class E2EBase(KernelCacheMixin, ABC):
    def __init__(self, actx, src_expansion, tgt_expansion, name=None):
        """
        :arg expansion: a subclass of :class:`sympy.expansion.ExpansionBase`
        :arg strength_usage: A list of integers indicating which expression
            uses which source strength indicator. This implicitly specifies the
            number of strength arrays that need to be passed.
            Default: all kernels use the same strength.
        """
        from sumpy.kernel import (
            TargetTransformationRemover, SourceTransformationRemover)
        txr = TargetTransformationRemover()
        sxr = SourceTransformationRemover()

        if src_expansion is tgt_expansion:
            tgt_expansion = src_expansion = (
                src_expansion.with_kernel(sxr(txr(src_expansion.kernel))))
        else:
            src_expansion = (
                src_expansion.with_kernel(sxr(txr(src_expansion.kernel))))
            tgt_expansion = (
                tgt_expansion.with_kernel(sxr(txr(tgt_expansion.kernel))))

        self.src_expansion = src_expansion
        self.tgt_expansion = tgt_expansion
        self.name = name or self.default_name

        self.actx = actx

        if src_expansion.dim != tgt_expansion.dim:
            raise ValueError("source and target expansions must have "
                    "same dimensionality")

        self.dim = src_expansion.dim

    @property
    @abstractmethod
    def default_name(self):
        pass

    @memoize_method
    def get_translation_loopy_insns(self):
        import sumpy.symbolic as sym
        dvec = sym.make_sym_vector("d", self.dim)

        src_coeff_exprs = [
            sym.Symbol(f"src_coeff{i}")
            for i in range(len(self.src_expansion))]
        src_rscale = sym.Symbol("src_rscale")

        tgt_rscale = sym.Symbol("tgt_rscale")

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique(f"coeff{i}", coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, src_rscale,
                        dvec=dvec, tgt_rscale=tgt_rscale, sac=sac))]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                sac.assignments.items(),
                vector_names={"d"},
                pymbolic_expr_maps=[self.tgt_expansion.get_code_transformer()],
                retain_names=tgt_coeff_names,
                )

    def get_cache_key(self):
        return (type(self).__name__, self.src_expansion, self.tgt_expansion)

    @abstractmethod
    def get_kernel(self):
        pass

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_iname(knl, "itgt_box", 64, outer_tag="g.0", inner_tag="l.0")
        knl = register_optimization_preambles(knl, self.actx.queue.device)

        return knl

# }}}


# {{{ E2EFromCSR: translation from "compressed sparse row"-like source box lists

class E2EFromCSR(E2EBase):
    """Implements translation from a "compressed sparse row"-like source box
    list.
    """

    @property
    def default_name(self):
        return "e2e_from_csr"

    def get_translation_loopy_insns(self):
        import sumpy.symbolic as sym
        dvec = sym.make_sym_vector("d", self.dim)

        src_rscale = sym.Symbol("src_rscale")
        tgt_rscale = sym.Symbol("tgt_rscale")

        ncoeff_src = len(self.src_expansion)
        src_coeff_exprs = [sym.Symbol(f"src_coeff{i}") for i in range(ncoeff_src)]

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique(f"coeff{i}", coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, src_rscale,
                        dvec, tgt_rscale, sac))]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                sac.assignments.items(),
                vector_names={"d"},
                pymbolic_expr_maps=[self.tgt_expansion.get_code_transformer()],
                retain_names=tgt_coeff_names,
                )

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
        loopy_knl = make_loopy_program([
                "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                "{[idim]: 0<=idim<dim}",
                ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> tgt_center[idim] = centers[idim, tgt_ibox]
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

                    """] + [f"""
                    tgt_expansions[tgt_ibox - tgt_base_ibox, {coeffidx}] = \
                        simul_reduce(sum, isrc_box, coeff{coeffidx}) \
                            {{id_prefix=write_expn}}
                    """ for coeffidx in range(ncoeff_tgt)] + ["""
                end
                """],
                kernel_data=[
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
                    ...
                ] + gather_loopy_arguments([self.src_expansion,
                                            self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={"dim": self.dim},
                )

        for knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.set_options(loopy_knl,
                enforce_variable_access_ordered="no_check")

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        t_unit = self.get_kernel()
        t_unit = lp.split_iname(
            t_unit, "itgt_box", 64, outer_tag="g.0", inner_tag="l.0")
        t_unit = register_optimization_preambles(t_unit, self.actx.queue.device)

        from sumpy.transform.metadata import E2EFromCSRKernelTag
        default_ep = t_unit.default_entrypoint
        t_unit = t_unit.with_kernel(default_ep.tagged(E2EFromCSRKernelTag()))

        return t_unit

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))

        t_unit = self.get_cached_kernel_executor()

        result = actx.call_loopy(
            t_unit,
            centers=centers,
            src_rscale=src_rscale, tgt_rscale=tgt_rscale,
            **kwargs)

        return result["tgt_expansions"]
# }}}


# {{{ M2LUsingTranslationClassesDependentData

class M2LUsingTranslationClassesDependentData(E2EFromCSR):
    """Implements translation from a "compressed sparse row"-like source box
    list using M2L translation classes dependent data
    """

    @property
    def default_name(self):
        return "m2l_using_translation_classes_dependent_data"

    def get_translation_loopy_insns(self, result_dtype):
        import sumpy.symbolic as sym
        dvec = sym.make_sym_vector("d", self.dim)

        src_rscale = sym.Symbol("src_rscale")
        tgt_rscale = sym.Symbol("tgt_rscale")

        m2l_translation = self.tgt_expansion.m2l_translation
        m2l_translation_classes_dependent_ndata = (
            m2l_translation.translation_classes_dependent_ndata(self.tgt_expansion,
                self.src_expansion))
        m2l_translation_classes_dependent_data = \
                [sym.Symbol(f"data{i}")
            for i in range(m2l_translation_classes_dependent_ndata)]

        ncoeff_src = len(self.src_expansion)

        src_coeff_exprs = [sym.Symbol(f"src_coeffs{i}") for i in range(ncoeff_src)]

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()
        tgt_coeff_names = [
                sac.assign_unique(f"tgt_coeff{i}", coeff_i)
                for i, coeff_i in enumerate(
                    self.tgt_expansion.translate_from(
                        self.src_expansion, src_coeff_exprs, src_rscale,
                        dvec, tgt_rscale, sac,
                        m2l_translation_classes_dependent_data=(
                            m2l_translation_classes_dependent_data)))]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                sac.assignments.items(),
                vector_names={"d", "src_coeffs", "data"},
                pymbolic_expr_maps=[self.tgt_expansion.get_code_transformer()],
                retain_names=tgt_coeff_names,
                complex_dtype=to_complex_dtype(result_dtype),
                )

    def get_inner_loopy_kernel(self, result_dtype):
        try:
            return self.tgt_expansion.loopy_translate_from(
                self.src_expansion)
        except NotImplementedError:
            pass

        m2l_translation = self.tgt_expansion.m2l_translation
        ndata = m2l_translation.translation_classes_dependent_ndata(
            self.tgt_expansion, self.src_expansion)
        if m2l_translation.use_preprocessing:
            ncoeff_src = m2l_translation.preprocess_multipole_nexprs(
                self.tgt_expansion, self.src_expansion)
            ncoeff_tgt = m2l_translation.postprocess_local_nexprs(
                self.tgt_expansion, self.src_expansion)
        else:
            ncoeff_src = len(self.src_expansion)
            ncoeff_tgt = len(self.tgt_expansion)

        import pymbolic as prim

        domains = []
        insns = self.get_translation_loopy_insns(result_dtype)
        tgt_coeffs = prim.var("tgt_coeffs")
        for i in range(ncoeff_tgt):
            expr = prim.var(f"tgt_coeff{i}")
            insn = lp.Assignment(assignee=tgt_coeffs[i],
                    expression=tgt_coeffs[i] + expr)
            insns.append(insn)

        return lp.make_function(domains, insns,
                        kernel_data=[
                            lp.GlobalArg("tgt_coeffs", shape=(ncoeff_tgt,),
                                is_output=True, is_input=True),
                            lp.GlobalArg("src_coeffs", shape=(ncoeff_src,)),
                            lp.GlobalArg("data", shape=(ndata,)),
                            lp.ValueArg("src_rscale"),
                            lp.ValueArg("tgt_rscale"),
                            ...],
                        name="e2e",
                        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
                        )

    def get_kernel(self, result_dtype):
        m2l_translation = self.tgt_expansion.m2l_translation
        m2l_translation_classes_dependent_ndata = \
            m2l_translation.translation_classes_dependent_ndata(
                self.tgt_expansion, self.src_expansion)

        if m2l_translation.use_preprocessing:
            ncoeff_src = m2l_translation.preprocess_multipole_nexprs(
                self.tgt_expansion, self.src_expansion)
            ncoeff_tgt = m2l_translation.postprocess_local_nexprs(
                self.tgt_expansion, self.src_expansion)
        else:
            ncoeff_src = len(self.src_expansion)
            ncoeff_tgt = len(self.tgt_expansion)

        # To clarify terminology:
        #
        # isrc_box -> The index in a list of (in this case, source) boxes
        # src_ibox -> The (global) box number for the (in this case, source) box
        #
        # (same for itgt_box, tgt_ibox)

        translation_knl = self.get_inner_loopy_kernel(result_dtype)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program([
                "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                "{[isrc_box]: isrc_start<=isrc_box<isrc_stop}",
                "{[icoeff_tgt]: 0<=icoeff_tgt<ncoeff_tgt}",
                "{[icoeff_src]: 0<=icoeff_src<ncoeff_src}",
                "{[idep]: 0<=idep<m2l_translation_classes_dependent_ndata}",
                ],
                ["""
                for itgt_box
                    <> tgt_ibox = target_boxes[itgt_box]
                    <> isrc_start = src_box_starts[itgt_box]
                    <> isrc_stop = src_box_starts[itgt_box+1]
                    for icoeff_tgt
                        <> tgt_expansion[icoeff_tgt] = 0 \
                            {id=init_coeffs, dup=icoeff_tgt}
                    end
                    for isrc_box
                        <> src_ibox = src_box_lists[isrc_box] \
                                {id=read_src_ibox}
                        <> translation_class = \
                                m2l_translation_classes_lists[isrc_box]
                        <> translation_class_rel = \
                                translation_class - translation_classes_level_start \
                                {id=translation_offset}
                        [icoeff_tgt]: tgt_expansion[icoeff_tgt] = e2e(
                            [icoeff_tgt]: tgt_expansion[icoeff_tgt],
                            [icoeff_src]: src_expansions[src_ibox - src_base_ibox,
                                icoeff_src],
                            [idep]: m2l_translation_classes_dependent_data[
                                translation_class_rel, idep],
                            src_rscale,
                            tgt_rscale,
                            )  {dep=init_coeffs,id=update_coeffs}
                    end
                    tgt_expansions[tgt_ibox - tgt_base_ibox, icoeff_tgt] = \
                            tgt_expansion[icoeff_tgt] \
                            {dep=update_coeffs, dup=icoeff_tgt,id=write_e2e}
                end
                """],
                kernel_data=[
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
                        shape=("ntgt_level_boxes", ncoeff_tgt),
                        offset=lp.auto),
                    lp.ValueArg("translation_classes_level_start",
                        np.int32),
                    lp.GlobalArg("m2l_translation_classes_dependent_data", None,
                        shape=("ntranslation_classes",
                            m2l_translation_classes_dependent_ndata),
                        offset=lp.auto),
                    lp.GlobalArg("m2l_translation_classes_lists", np.int32,
                        shape=("ntranslation_classes_lists"), strides=(1,),
                        offset=lp.auto),
                    lp.ValueArg("ntranslation_classes, ntranslation_classes_lists",
                        np.int32),
                    ...
                ] + gather_loopy_arguments([self.src_expansion,
                                            self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                fixed_parameters={
                        "dim": self.dim,
                        "m2l_translation_classes_dependent_ndata": (
                            m2l_translation_classes_dependent_ndata),
                        "ncoeff_tgt": ncoeff_tgt,
                        "ncoeff_src": ncoeff_src},
                silenced_warnings="write_race(write_e2e*)",
                )

        loopy_knl = lp.merge([translation_knl, loopy_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "e2e")
        loopy_knl = lp.add_dependency(
                loopy_knl, "id:e2e_insn*", "read_src_ibox")
        loopy_knl = lp.add_dependency(
                loopy_knl, "id:e2e_insn*", "translation_offset")

        for knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self, result_dtype):
        knl = self.get_kernel(result_dtype)
        knl = self.tgt_expansion.m2l_translation.optimize_loopy_kernel(
                knl, self.tgt_expansion, self.src_expansion)
        knl = register_optimization_preambles(knl, self.actx.queue.device)

        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))
        src_expansions = kwargs.pop("src_expansions")

        t_unit = self.get_kernel(result_dtype=src_expansions.dtype)
        result = actx.call_loopy(
            t_unit,
            src_expansions=src_expansions,
            centers=centers,
            src_rscale=src_rscale, tgt_rscale=tgt_rscale,
            **kwargs)

        return result["tgt_expansions"]

# }}}


# {{{ M2LGenerateTranslationClassesDependentData

class M2LGenerateTranslationClassesDependentData(E2EBase):
    """Implements precomputing the M2L kernel dependent data which are
    translation classes dependent derivatives.
    """

    @property
    def default_name(self):
        return "m2l_generate_translation_classes_dependent_data"

    def get_kernel(self, result_dtype):
        m2l_translation = self.tgt_expansion.m2l_translation
        m2l_translation_classes_dependent_ndata = \
            m2l_translation.translation_classes_dependent_ndata(
                self.tgt_expansion, self.src_expansion)

        translation_classes_data_knl = \
            m2l_translation.loopy_translation_classes_dependent_data(
                self.tgt_expansion, self.src_expansion, result_dtype)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program([
                "{[itr_class]: 0<=itr_class<ntranslation_classes}",
                "{[idim]: 0<=idim<dim}",
                "{[idata]: 0<=idata<m2l_translation_classes_dependent_ndata}",
                ],
                ["""
                for itr_class
                    <> d[idim] = m2l_translation_vectors[idim, \
                            itr_class + translation_classes_level_start] \
                            {id=set_d,dup=idim}
                    [idata]: m2l_translation_classes_dependent_data[
                            itr_class, idata] = \
                        m2l_data(
                            src_rscale,
                            [idim]: d[idim],
                        ) {id=update,dep=set_d}
                end
                """],
                kernel_data=[
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("m2l_translation_classes_dependent_data", None,
                        shape=("ntranslation_classes",
                            m2l_translation_classes_dependent_ndata),
                        offset=lp.auto),
                    lp.GlobalArg("m2l_translation_vectors", None,
                        shape=("dim", "ntranslation_vectors")),
                    lp.ValueArg("ntranslation_classes", np.int32),
                    lp.ValueArg("ntranslation_vectors", np.int32),
                    lp.ValueArg("translation_classes_level_start", np.int32),
                    ...
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntranslation_classes>=1",
                fixed_parameters={
                    "dim": self.dim,
                    "m2l_translation_classes_dependent_ndata": (
                        m2l_translation_classes_dependent_ndata)},
                )

        for expr_knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = expr_knl.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.merge([loopy_knl, translation_classes_data_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "m2l_data")
        loopy_knl = lp.set_options(loopy_knl,
                enforce_variable_access_ordered="no_check",
                # FIXME: Without this, Loopy spends an eternity checking
                # scattered writes to global variables to see whether barriers
                # need to be inserted.
                disable_global_barriers=True)

        return loopy_knl

    def get_optimized_kernel(self, result_dtype):
        # FIXME
        knl = self.get_kernel(result_dtype)
        knl = lp.tag_inames(knl, "idim*:unr")
        knl = lp.tag_inames(knl, {"itr_class": "g.0"})
        knl = register_optimization_preambles(knl, self.actx.queue.device)

        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg src_rscale:
        :arg translation_classes_level_start:
        :arg ntranslation_classes:
        :arg m2l_translation_classes_dependent_data:
        :arg m2l_translation_vectors:
        """
        m2l_translation_vectors = kwargs.pop("m2l_translation_vectors")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = m2l_translation_vectors.dtype.type(kwargs.pop("src_rscale"))

        m2l_translation_classes_dependent_data = kwargs.pop(
                "m2l_translation_classes_dependent_data")
        result_dtype = m2l_translation_classes_dependent_data.dtype

        t_unit = self.get_kernel(result_dtype=result_dtype)

        result = actx.call_loopy(
            t_unit,
            src_rscale=src_rscale,
            m2l_translation_vectors=m2l_translation_vectors,
            **kwargs)

        return result["m2l_translation_classes_dependent_data"]

# }}}


# {{{ M2LPreprocessMultipole

class M2LPreprocessMultipole(E2EBase):
    """Computes the preprocessed multipole expansion for accelerated M2L"""

    @property
    def default_name(self):
        return "m2l_preprocess_multipole"

    @memoize_method
    def get_inner_knl_and_optimizations(self, result_dtype):
        m2l_translation = self.tgt_expansion.m2l_translation
        return m2l_translation.loopy_preprocess_multipole(
            self.tgt_expansion, self.src_expansion, result_dtype)

    def get_kernel(self, result_dtype):
        m2l_translation = self.tgt_expansion.m2l_translation
        nsrc_coeffs = len(self.src_expansion)
        npreprocessed_src_coeffs = \
            m2l_translation.preprocess_multipole_nexprs(self.tgt_expansion,
                self.src_expansion)
        single_box_preprocess_knl, _ = self.get_inner_knl_and_optimizations(
                result_dtype)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program(
                [
                    "{[isrc_box]: 0<=isrc_box<nsrc_boxes}",
                    "{[isrc_coeff]: 0<=isrc_coeff<nsrc_coeffs}",
                    "{[itgt_coeff]: 0<=itgt_coeff<npreprocessed_src_coeffs}",
                ],
                ["""
                for isrc_box
                    [itgt_coeff]: preprocessed_src_expansions[isrc_box, itgt_coeff] \
                        = m2l_preprocess_inner(
                            src_rscale,
                            [isrc_coeff]: src_expansions[isrc_box, isrc_coeff],
                        )
                end
                """],
                kernel_data=[
                    lp.ValueArg("nsrc_boxes", np.int32),
                    lp.ValueArg("src_rscale", None),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_boxes", nsrc_coeffs), offset=lp.auto),
                    lp.GlobalArg("preprocessed_src_expansions", None,
                        shape=("nsrc_boxes", npreprocessed_src_coeffs),
                        offset=lp.auto),
                    ...
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="nsrc_boxes>=1",
                fixed_parameters={
                    "nsrc_coeffs": nsrc_coeffs,
                    "npreprocessed_src_coeffs": npreprocessed_src_coeffs},
                )

        for expn in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.merge([loopy_knl, single_box_preprocess_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "m2l_preprocess_inner")

        return loopy_knl

    def get_optimized_kernel(self, result_dtype):
        knl = self.get_kernel(result_dtype)
        knl = lp.tag_inames(knl, "isrc_box:g.0")
        _, optimizations = self.get_inner_knl_and_optimizations(result_dtype)
        for optimization in optimizations:
            knl = optimization(knl)
        knl = register_optimization_preambles(knl, self.actx.queue.device)
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg src_expansions
        :arg preprocessed_src_expansions
        """
        preprocessed_src_expansions = kwargs.pop("preprocessed_src_expansions")
        result_dtype = preprocessed_src_expansions.dtype

        knl = self.get_cached_kernel_executor(result_dtype=result_dtype)
        result = actx.call_loopy(
            knl,
            preprocessed_src_expansions=preprocessed_src_expansions,
            **kwargs)

        return result["preprocessed_src_expansions"]

# }}}


# {{{ M2LPostprocessLocal

class M2LPostprocessLocal(E2EBase):
    """Postprocesses locals expansions for accelerated M2L"""

    @property
    def default_name(self):
        return "m2l_postprocess_local"

    @memoize_method
    def get_inner_knl_and_optimizations(self, result_dtype):
        m2l_translation = self.tgt_expansion.m2l_translation
        return m2l_translation.loopy_postprocess_local(
            self.tgt_expansion, self.src_expansion, result_dtype)

    def get_kernel(self, result_dtype):
        m2l_translation = self.tgt_expansion.m2l_translation
        ntgt_coeffs = len(self.tgt_expansion)
        ntgt_coeffs_before_postprocessing = \
            m2l_translation.postprocess_local_nexprs(self.tgt_expansion,
                self.src_expansion)

        single_box_postprocess_knl, _ = self.get_inner_knl_and_optimizations(
                result_dtype)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program([
                "{[itgt_box]: 0<=itgt_box<ntgt_boxes}",
                "{[isrc_coeff]: 0<=isrc_coeff<nsrc_coeffs}",
                "{[itgt_coeff]: 0<=itgt_coeff<ntgt_coeffs}",
                ],
                ["""
                for itgt_box
                    [itgt_coeff]: tgt_expansions[itgt_box, itgt_coeff] = \
                        m2l_postprocess_inner(
                            src_rscale,
                            tgt_rscale,
                            [isrc_coeff]: tgt_expansions_before_postprocessing[ \
                            itgt_box, isrc_coeff],
                       )
                end
                """],
                kernel_data=[
                    lp.ValueArg("ntgt_boxes", np.int32),
                    lp.ValueArg("src_rscale", None),
                    lp.ValueArg("tgt_rscale", None),
                    lp.GlobalArg("tgt_expansions", result_dtype,
                        shape=("ntgt_boxes", ntgt_coeffs), offset=lp.auto),
                    lp.GlobalArg("tgt_expansions_before_postprocessing", None,
                        shape=("ntgt_boxes", ntgt_coeffs_before_postprocessing),
                        offset=lp.auto),
                    ...
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                fixed_parameters={
                    "dim": self.dim,
                    "nsrc_coeffs": ntgt_coeffs_before_postprocessing,
                    "ntgt_coeffs": ntgt_coeffs,
                },
                )

        for expn in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.merge([loopy_knl, single_box_postprocess_knl])
        loopy_knl = lp.inline_callable_kernel(loopy_knl, "m2l_postprocess_inner")

        loopy_knl = lp.set_options(loopy_knl,
                enforce_variable_access_ordered="no_check")
        return loopy_knl

    def get_optimized_kernel(self, result_dtype):
        knl = self.get_kernel(result_dtype)
        knl = lp.tag_inames(knl, "itgt_box:g.0")
        _, optimizations = self.get_inner_knl_and_optimizations(result_dtype)
        for optimization in optimizations:
            knl = optimization(knl)
        knl = lp.add_inames_for_unused_hw_axes(knl)
        knl = register_optimization_preambles(knl, self.actx.queue.device)
        return knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg tgt_expansions
        :arg tgt_expansions_before_postprocessing
        """
        tgt_expansions = kwargs.pop("tgt_expansions")
        result_dtype = tgt_expansions.dtype

        knl = self.get_cached_kernel_executor(result_dtype=result_dtype)
        result = actx.call_loopy(
            knl,
            tgt_expansions=tgt_expansions,
            **kwargs)

        return result["tgt_expansions"]

# }}}


# {{{ E2EFromChildren: translation from a box's children

class E2EFromChildren(E2EBase):
    @property
    def default_name(self):
        return "e2e_from_children"

    def get_kernel(self):
        ncoeffs_src = len(self.src_expansion)
        ncoeffs_tgt = len(self.tgt_expansion)

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
        loopy_knl = make_loopy_program([
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
                            """.format(i=i) for i in range(ncoeffs_src)] + [
                            ] + loopy_insns + ["""
                            tgt_expansions[tgt_ibox - tgt_base_ibox, {i}] = \
                                + coeff{i} \
                                {{id_prefix=write_expn,dep=compute_coeff*,
                                    nosync=read_coeff*}}
                            """.format(i=i) for i in range(ncoeffs_tgt)] + ["""
                        end
                    end
                end
                """],
                kernel_data=[
                    lp.GlobalArg("target_boxes", None, shape=lp.auto,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, aligned_nboxes"),
                    lp.ValueArg("src_rscale,tgt_rscale", None),
                    lp.GlobalArg("box_child_ids", None,
                        shape="nchildren, aligned_nboxes"),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeffs_tgt), offset=lp.auto),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs_src), offset=lp.auto),
                    lp.ValueArg("src_base_ibox,tgt_base_ibox", np.int32),
                    lp.ValueArg("ntgt_level_boxes,nsrc_level_boxes", np.int32),
                    lp.ValueArg("aligned_nboxes", np.int32),
                    ...
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={"dim": self.dim, "nchildren": 2**self.dim},
                )

        for knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.set_options(loopy_knl,
                enforce_variable_access_ordered="no_check")

        from sumpy.transform.metadata import E2EFromChildrenKernelTag
        default_ep = loopy_knl.default_entrypoint
        loopy_knl = loopy_knl.with_kernel(
            default_ep.tagged(E2EFromChildrenKernelTag()))

        return loopy_knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))

        t_unit = self.get_cached_kernel_executor()

        kwargs["aligned_nboxes"] = centers.shape[1]

        result = actx.call_loopy(
            t_unit,
            centers=centers,
            src_rscale=src_rscale, tgt_rscale=tgt_rscale,
            **kwargs)

        return result["tgt_expansions"]

# }}}


# {{{ E2EFromParent: translation from a box's parent

class E2EFromParent(E2EBase):
    @property
    def default_name(self):
        return "e2e_from_parent"

    def get_kernel(self):
        ncoeffs_src = len(self.src_expansion)
        ncoeffs_tgt = len(self.tgt_expansion)

        # To clarify terminology:
        #
        # isrc_box -> The index in a list of (in this case, source) boxes
        # src_ibox -> The (global) box number for the (in this case, source) box
        #
        # (same for itgt_box, tgt_ibox)

        from sumpy.tools import gather_loopy_arguments
        loopy_knl = make_loopy_program([
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
                    """.format(i=i) for i in range(ncoeffs_src)] + [

                    ] + self.get_translation_loopy_insns() + ["""

                    tgt_expansions[tgt_ibox - tgt_base_ibox, {i}] = coeff{i} \
                        {{id_prefix=write_expn,nosync=read_expn*}}
                    """.format(i=i) for i in range(ncoeffs_tgt)] + ["""
                end
                """],
                kernel_data=[
                    lp.GlobalArg("target_boxes", None, shape=lp.auto,
                        offset=lp.auto),
                    lp.GlobalArg("centers", None, shape="dim, naligned_boxes"),
                    lp.ValueArg("src_rscale,tgt_rscale", None),
                    lp.ValueArg("naligned_boxes,nboxes", np.int32),
                    lp.ValueArg("tgt_base_ibox,src_base_ibox", np.int32),
                    lp.ValueArg("ntgt_level_boxes,nsrc_level_boxes", np.int32),
                    lp.GlobalArg("box_parent_ids", None, shape="nboxes"),
                    lp.GlobalArg("tgt_expansions", None,
                        shape=("ntgt_level_boxes", ncoeffs_tgt), offset=lp.auto),
                    lp.GlobalArg("src_expansions", None,
                        shape=("nsrc_level_boxes", ncoeffs_src), offset=lp.auto),
                    ...
                ] + gather_loopy_arguments([self.src_expansion, self.tgt_expansion]),
                name=self.name,
                assumptions="ntgt_boxes>=1",
                silenced_warnings="write_race(write_expn*)",
                fixed_parameters={"dim": self.dim, "nchildren": 2**self.dim},
                )

        for knl in [self.src_expansion.kernel, self.tgt_expansion.kernel]:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.set_options(loopy_knl,
                enforce_variable_access_ordered="no_check")

        from sumpy.transform.metadata import E2EFromParentKernelTag
        default_ep = loopy_knl.default_entrypoint
        loopy_knl = loopy_knl.with_kernel(
            default_ep.tagged(E2EFromParentKernelTag()))

        return loopy_knl

    def __call__(self, actx: PyOpenCLArrayContext, **kwargs):
        """
        :arg src_expansions:
        :arg src_box_starts:
        :arg src_box_lists:
        :arg src_rscale:
        :arg tgt_rscale:
        :arg centers:
        """
        centers = kwargs.pop("centers")
        # "1" may be passed for rscale, which won't have its type
        # meaningfully inferred. Make the type of rscale explicit.
        src_rscale = centers.dtype.type(kwargs.pop("src_rscale"))
        tgt_rscale = centers.dtype.type(kwargs.pop("tgt_rscale"))

        t_unit= self.get_cached_kernel_executor()

        # update kwargs for lazy
        kwargs["naligned_boxes"] = centers.shape[1]
        kwargs["nboxes"] = kwargs["box_parent_ids"].shape[0]
        kwargs["ntgt_level_boxes"] = kwargs["tgt_expansions"].shape[0]
        kwargs["nsrc_level_boxes"] = kwargs["src_expansions"].shape[0]
        kwargs["ntgt_boxes"] = kwargs["target_boxes"].shape[0]

        result = actx.call_loopy(
            t_unit,
            centers=centers,
            src_rscale=src_rscale, tgt_rscale=tgt_rscale,
            **kwargs)

        return result["tgt_expansions"]

# }}}

# vim: foldmethod=marker
