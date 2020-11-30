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
import sumpy.symbolic as sym
from pytools import memoize_method
from pymbolic import parse, var

from sumpy.tools import (
        KernelComputation, KernelCacheWrapper, is_obj_array_like)

import logging
logger = logging.getLogger(__name__)


__doc__ = """

QBX for Layer Potentials
------------------------

.. autoclass:: LayerPotentialBase
.. autoclass:: LayerPotential
.. autoclass:: LayerPotentialMatrixGenerator
.. autoclass:: LayerPotentialMatrixBlockGenerator

"""


def stringify_expn_index(i):
    if isinstance(i, tuple):
        return "_".join(stringify_expn_index(i_i) for i_i in i)
    else:
        assert isinstance(i, int)
        if i < 0:
            return "m%d" % (-i)
        else:
            return str(i)


# {{{ layer potential computation

# {{{ base class

class LayerPotentialBase(KernelComputation, KernelCacheWrapper):
    def __init__(self, ctx, expansion, strength_usage=None,
            value_dtypes=None, name=None, device=None,
            in_kernels=None, out_kernels=None):

        from sumpy.kernel import SourceDerivativeRemover, TargetDerivativeRemover
        from pytools import single_valued
        sdr = SourceDerivativeRemover()
        tdr = TargetDerivativeRemover()
        if in_kernels is None and out_kernels is None:
            from warnings import warn
            warn("A list for the second argument of the constructor is deprecated."
                "Use a single expansion and pass in_kernels and out_kernels",
                DeprecationWarning, stacklevel=2)
            in_kernel = single_valued(tdr(exp.kernel) for exp in expansion)
            base_kernel = sdr(in_kernel)
            out_kernels = tuple(sdr(exp.kernel) for exp in expansion)
            in_kernels = (in_kernel,)
            expansion = expansion[0].copy(kernel=base_kernel)

        KernelComputation.__init__(self, ctx=ctx, out_kernels=out_kernels,
                strength_usage=strength_usage, in_kernels=in_kernels,
                value_dtypes=value_dtypes, name=name, device=device)

        self.dim = single_valued(knl.dim for knl in self.out_kernels)
        self.expansion = expansion

    def get_cache_key(self):
        return (type(self).__name__, self.expansion, tuple(self.out_kernels),
                tuple(self.in_kernels), tuple(self.strength_usage),
                tuple(self.value_dtypes))

    def _expand(self, sac, avec, bvec, rscale, isrc):
        coefficients = np.zeros(len(self.expansion), dtype=object)
        from sumpy.symbolic import PymbolicToSympyMapper
        conv = PymbolicToSympyMapper()
        for idx, knl in enumerate(self.in_kernels):
            strength = conv(self.get_strength_or_not(isrc, idx))
            coefficients += np.array([coeff * strength for
                                i, coeff in enumerate(
                                    self.expansion.coefficients_from_source(knl,
                                          avec, bvec, rscale))])

        return list(coefficients)

    def _evaluate(self, sac, avec, bvec, rscale, expansion_nr, coefficients):
        expn = self.out_kernels[expansion_nr]
        assigned_coeffs = [
            sym.Symbol(
                sac.assign_unique("expn%dcoeff%s" % (
                    expansion_nr, stringify_expn_index(i)),
                    expn.postprocess_at_target(
                        coefficients[self.expansion.get_storage_index(i)], bvec)))
            for i in self.expansion.get_coefficient_identifiers()]

        return sac.assign_unique("expn%d_result" % expansion_nr,
            self.expansion.evaluate(expn, assigned_coeffs, bvec, rscale))

    def get_loopy_insns_and_result_names(self):
        from sumpy.symbolic import make_sym_vector
        avec = make_sym_vector("a", self.dim)
        bvec = make_sym_vector("b", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        logger.info("compute expansion expressions: start")

        rscale = sym.Symbol("rscale")
        isrc_sym = var("isrc")

        coefficients = self._expand(sac, avec, bvec, rscale, isrc_sym)
        result_names = [self._evaluate(sac, avec, bvec, rscale, i, coefficients)
                        for i in range(len(self.out_kernels))]

        logger.info("compute expansion expressions: done")

        sac.run_global_cse()

        pymbolic_expr_maps = [knl.get_code_transformer() for knl in (
            self.out_kernels + self.in_kernels)]

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(
                sac.assignments.items(),
                vector_names={"a", "b"},
                pymbolic_expr_maps=pymbolic_expr_maps,
                retain_names=result_names,
                complex_dtype=np.complex128  # FIXME
                )

        return loopy_insns, result_names

    def get_strength_or_not(self, isrc, kernel_idx):
        return var("strength_%d_isrc" % self.strength_usage[kernel_idx])

    def get_kernel_exprs(self, result_names):
        exprs = [var(name) for i, name in enumerate(result_names)]

        return [lp.Assignment(id=None,
                    assignee="pair_result_%d" % i, expression=expr,
                    temp_var_type=lp.Optional(None))
                for i, expr in enumerate(exprs)]

    def get_default_src_tgt_arguments(self):
        from sumpy.tools import gather_loopy_source_arguments
        return ([
                lp.GlobalArg("src", None,
                    shape=(self.dim, "nsources"), order="C"),
                lp.GlobalArg("tgt", None,
                    shape=(self.dim, "ntargets"), order="C"),
                lp.GlobalArg("center", None,
                    shape=(self.dim, "ntargets"), order="C"),
                lp.GlobalArg("expansion_radii",
                    None, shape="ntargets"),
                lp.ValueArg("nsources", None),
                lp.ValueArg("ntargets", None)]
                + gather_loopy_source_arguments(self.out_kernels + self.in_kernels))

    def get_kernel(self):
        raise NotImplementedError

    def get_optimized_kernel(self,
            targets_is_obj_array, sources_is_obj_array, centers_is_obj_array):
        # FIXME specialize/tune for GPU/CPU
        loopy_knl = self.get_kernel()

        if targets_is_obj_array:
            loopy_knl = lp.tag_array_axes(loopy_knl, "tgt", "sep,C")
        if sources_is_obj_array:
            loopy_knl = lp.tag_array_axes(loopy_knl, "src", "sep,C")
        if centers_is_obj_array:
            loopy_knl = lp.tag_array_axes(loopy_knl, "center", "sep,C")

        import pyopencl as cl
        dev = self.context.devices[0]
        if dev.type & cl.device_type.CPU:
            loopy_knl = lp.split_iname(loopy_knl, "itgt", 16, outer_tag="g.0",
                    inner_tag="l.0")
            loopy_knl = lp.split_iname(loopy_knl, "isrc", 256)
            loopy_knl = lp.prioritize_loops(loopy_knl,
                    ["isrc_outer", "itgt_inner"])
        else:
            from warnings import warn
            warn("don't know how to tune layer potential computation for '%s'" % dev)
            loopy_knl = lp.split_iname(loopy_knl, "itgt", 128, outer_tag="g.0")

        return loopy_knl

# }}}


# {{{ direct applier

class LayerPotential(LayerPotentialBase):
    """Direct applier for the layer potential.

    .. automethod:: __call__
    """

    default_name = "qbx_apply"

    @memoize_method
    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        kernel_exprs = self.get_kernel_exprs(result_names)
        arguments = (
            self.get_default_src_tgt_arguments()
            + [lp.GlobalArg("strength_%d" % i,
                None, shape="nsources", order="C")
            for i in range(self.strength_count)]
            + [lp.GlobalArg("result_%d" % i,
                None, shape="ntargets", order="C")
            for i in range(len(self.out_kernels))])

        loopy_knl = lp.make_kernel(["""
            {[itgt, isrc, idim]: \
                0 <= itgt < ntargets and \
                0 <= isrc < nsources and \
                0 <= idim < dim}
            """],
            self.get_kernel_scaling_assignments()
            + ["for itgt, isrc"]
            + ["<> a[idim] = center[idim, itgt] - src[idim, isrc] {dup=idim}"]
            + ["<> b[idim] = tgt[idim, itgt] - center[idim, itgt] {dup=idim}"]
            + ["<> rscale = expansion_radii[itgt]"]
            + [f"<> strength_{i}_isrc = strength_{i}[isrc]" for i in
                    range(self.strength_count)]
            + loopy_insns + kernel_exprs
            + ["""
                result_{i}[itgt] = knl_{i}_scaling * \
                    simul_reduce(sum, isrc, pair_result_{i}) \
                        {{id_prefix=write_lpot,inames=itgt}}
                """.format(i=iknl)
                for iknl in range(len(self.out_kernels))]
            + ["end"],
            arguments,
            name=self.name,
            assumptions="ntargets>=1 and nsources>=1",
            default_offset=lp.auto,
            silenced_warnings="write_race(write_lpot*)",
            fixed_parameters=dict(dim=self.dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        for expn in self.out_kernels + self.in_kernels:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self, queue, targets, sources, centers, strengths, expansion_radii,
            **kwargs):
        """
        :arg strengths: are required to have area elements and quadrature weights
            already multiplied in.
        """

        knl = self.get_cached_optimized_kernel(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources),
                centers_is_obj_array=is_obj_array_like(centers))

        for i, dens in enumerate(strengths):
            kwargs["strength_%d" % i] = dens

        return knl(queue, src=sources, tgt=targets, center=centers,
                expansion_radii=expansion_radii, **kwargs)

# }}}


# {{{ matrix writer

class LayerPotentialMatrixGenerator(LayerPotentialBase):
    """Generator for layer potential matrix entries."""

    default_name = "qbx_matrix"

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    @memoize_method
    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        kernel_exprs = self.get_kernel_exprs(result_names)
        arguments = (
            self.get_default_src_tgt_arguments()
            + [lp.GlobalArg("result_%d" % i,
                dtype, shape="ntargets, nsources", order="C")
             for i, dtype in enumerate(self.value_dtypes)])

        loopy_knl = lp.make_kernel(["""
            {[itgt, isrc, idim]: \
                0 <= itgt < ntargets and \
                0 <= isrc < nsources and \
                0 <= idim < dim}
            """],
            self.get_kernel_scaling_assignments()
            + ["for itgt, isrc"]
            + ["<> a[idim] = center[idim, itgt] - src[idim, isrc] {dup=idim}"]
            + ["<> b[idim] = tgt[idim, itgt] - center[idim, itgt] {dup=idim}"]
            + ["<> rscale = expansion_radii[itgt]"]
            + loopy_insns + kernel_exprs
            + ["""
                result_{i}[itgt, isrc] = \
                    knl_{i}_scaling * pair_result_{i} \
                        {{inames=isrc:itgt}}
                """.format(i=iknl)
                for iknl in range(len(self.out_kernels))]
            + ["end"],
            arguments,
            name=self.name,
            assumptions="ntargets>=1 and nsources>=1",
            default_offset=lp.auto,
            fixed_parameters=dict(dim=self.dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        for expn in self.in_kernels + self.out_kernels:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def __call__(self, queue, targets, sources, centers, expansion_radii, **kwargs):
        knl = self.get_cached_optimized_kernel(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources),
                centers_is_obj_array=is_obj_array_like(centers))

        return knl(queue, src=sources, tgt=targets, center=centers,
                expansion_radii=expansion_radii, **kwargs)

# }}}


# {{{

class LayerPotentialMatrixBlockGenerator(LayerPotentialBase):
    """Generator for a subset of the layer potential matrix entries.

    .. automethod:: __call__
    """

    default_name = "qbx_block"

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    @memoize_method
    def get_kernel(self):
        loopy_insns, result_names = self.get_loopy_insns_and_result_names()
        kernel_exprs = self.get_kernel_exprs(result_names)
        arguments = (
            self.get_default_src_tgt_arguments()
            + [
                lp.GlobalArg("srcindices", None, shape="nresult"),
                lp.GlobalArg("tgtindices", None, shape="nresult"),
                lp.ValueArg("nresult", None)
            ]
            + [lp.GlobalArg("result_%d" % i, dtype, shape="nresult")
             for i, dtype in enumerate(self.value_dtypes)])

        loopy_knl = lp.make_kernel([
            "{[imat, idim]: 0 <= imat < nresult and 0 <= idim < dim}"
            ],
            self.get_kernel_scaling_assignments()
            # NOTE: itgt, isrc need to always be defined in case a statement
            # in loopy_insns or kernel_exprs needs them (e.g. hardcoded in
            # places like get_kernel_exprs)
            + ["""
                for imat
                    <> itgt = tgtindices[imat]
                    <> isrc = srcindices[imat]

                    <> a[idim] = center[idim, itgt] - src[idim, isrc] {dup=idim}
                    <> b[idim] = tgt[idim, itgt] - center[idim, itgt] {dup=idim}
                    <> rscale = expansion_radii[itgt]
            """]
            + loopy_insns + kernel_exprs
            + ["""
                    result_{i}[imat] = knl_{i}_scaling * pair_result_{i} \
                            {{id_prefix=write_lpot}}
                """.format(i=iknl)
                for iknl in range(len(self.out_kernels))]
            + ["end"],
            arguments,
            name=self.name,
            assumptions="nresult>=1",
            default_offset=lp.auto,
            silenced_warnings="write_race(write_lpot*)",
            fixed_parameters=dict(dim=self.dim),
            lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
        loopy_knl = lp.add_dtypes(loopy_knl,
            dict(nsources=np.int32, ntargets=np.int32))

        for expn in self.in_kernels + self.out_kernels:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self,
            targets_is_obj_array, sources_is_obj_array, centers_is_obj_array):
        loopy_knl = self.get_kernel()

        if targets_is_obj_array:
            loopy_knl = lp.tag_array_axes(loopy_knl, "tgt", "sep,C")
        if sources_is_obj_array:
            loopy_knl = lp.tag_array_axes(loopy_knl, "src", "sep,C")
        if centers_is_obj_array:
            loopy_knl = lp.tag_array_axes(loopy_knl, "center", "sep,C")

        loopy_knl = lp.split_iname(loopy_knl, "imat", 1024, outer_tag="g.0")
        return loopy_knl

    def __call__(self, queue, targets, sources, centers, expansion_radii,
                 index_set, **kwargs):
        """
        :arg targets: target point coordinates.
        :arg sources: source point coordinates.
        :arg centers: QBX target expansion centers.
        :arg expansion_radii: radii for each expansion center.
        :arg index_set: a :class:`sumpy.tools.MatrixBlockIndexRanges` used
            to define the blocks.
        :return: a tuple of one-dimensional arrays of kernel evaluations at
            target-source pairs described by `index_set`.
        """
        knl = self.get_cached_optimized_kernel(
                targets_is_obj_array=is_obj_array_like(targets),
                sources_is_obj_array=is_obj_array_like(sources),
                centers_is_obj_array=is_obj_array_like(centers))

        return knl(queue,
                   src=sources,
                   tgt=targets,
                   center=centers,
                   expansion_radii=expansion_radii,
                   tgtindices=index_set.linear_row_indices,
                   srcindices=index_set.linear_col_indices, **kwargs)

# }}}

# }}}


# {{{ jump term handling

def find_jump_term(kernel, arg_provider):
    from sumpy.kernel import (
            AxisTargetDerivative,
            DirectionalSourceDerivative,
            DirectionalTargetDerivative,
            DerivativeBase)

    tgt_derivatives = []
    src_derivatives = []

    while isinstance(kernel, DerivativeBase):
        if isinstance(kernel, AxisTargetDerivative):
            tgt_derivatives.append(kernel.axis)
            kernel = kernel.kernel
        elif isinstance(kernel, DirectionalTargetDerivative):
            tgt_derivatives.append(kernel.dir_vec_name)
            kernel = kernel.kernel
        elif isinstance(kernel, DirectionalSourceDerivative):
            src_derivatives.append(kernel.dir_vec_name)
            kernel = kernel.kernel
        else:
            raise RuntimeError("derivative type '%s' not understood"
                    % type(kernel))

    tgt_count = len(tgt_derivatives)
    src_count = len(src_derivatives)

    info = arg_provider

    if src_count == 0:
        if tgt_count == 0:
            return 0
        elif tgt_count == 1:
            tgt_derivative, = tgt_derivatives
            if isinstance(tgt_derivative, int):
                # axis derivative
                return info.side/2 * info.normal[tgt_derivative] * info.density
            else:
                # directional derivative
                return (info.side/2
                        * np.dot(info.normal, getattr(info, tgt_derivative))
                        * info.density)

        elif tgt_count == 2:
            i, j = tgt_derivatives

            assert isinstance(i, int)
            assert isinstance(j, int)

            from pytools import delta
            return (
                    - info.side * info.mean_curvature / 2
                    * (-delta(i, j) + 2*info.normal[i]*info.normal[j])
                    * info.density

                    + info.side / 2
                    * (info.normal[i]*info.tangent[j]
                        + info.normal[j]*info.tangent[i])
                    * info.density_prime)

    elif src_count == 1:
        src_derivative_name, = src_derivatives

        if tgt_count == 0:
            return (
                    - info.side/2
                    * np.dot(info.normal, getattr(info, src_derivative_name))
                    * info.density)
        elif tgt_count == 1:
            from warnings import warn
            warn("jump terms for mixed derivatives (1 src+1 tgt) only available "
                    "for the double-layer potential")
            i, = tgt_derivatives
            assert isinstance(i, int)
            return (
                    - info.side/2
                    * info.tangent[i]
                    * info.density_prime)

    raise ValueError("don't know jump term for %d "
            "target and %d source derivatives" % (tgt_count, src_count))


# {{{ symbolic argument provider

class _JumpTermSymbolicArgumentProvider:
    """This class answers requests by :func:`find_jump_term` for symbolic values
    of quantities needed for the computation of the jump terms and tracks what
    data was requested. This tracking allows assembling the argument list of the
    resulting computational kernel.
    """

    def __init__(self, data_args, dim, density_var_name,
            density_dtype, geometry_dtype):
        # dictionary of loopy arguments
        self.arguments = data_args
        self.dim = dim
        self.density_var_name = density_var_name
        self.density_dtype = density_dtype
        self.geometry_dtype = geometry_dtype

    @property
    @memoize_method
    def density(self):
        self.arguments[self.density_var_name] = \
                lp.GlobalArg(self.density_var_name, self.density_dtype,
                        shape="ntargets", order="C")
        return parse("%s[itgt]" % self.density_var_name)

    @property
    @memoize_method
    def density_prime(self):
        prime_var_name = self.density_var_name+"_prime"
        self.arguments[prime_var_name] = \
                lp.GlobalArg(prime_var_name, self.density_dtype,
                        shape="ntargets", order="C")
        return parse("%s[itgt]" % prime_var_name)

    @property
    @memoize_method
    def side(self):
        self.arguments["side"] = \
                lp.GlobalArg("side", self.geometry_dtype, shape="ntargets")
        return parse("side[itgt]")

    @property
    @memoize_method
    def normal(self):
        self.arguments["normal"] = \
                lp.GlobalArg("normal", self.geometry_dtype,
                        shape=("ntargets", self.dim), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("normal[itgt, %d]" % i)
            for i in range(self.dim)])

    @property
    @memoize_method
    def tangent(self):
        self.arguments["tangent"] = \
                lp.GlobalArg("tangent", self.geometry_dtype,
                        shape=("ntargets", self.dim), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("tangent[itgt, %d]" % i)
            for i in range(self.dim)])

    @property
    @memoize_method
    def mean_curvature(self):
        self.arguments["mean_curvature"] = \
                lp.GlobalArg("mean_curvature",
                        self.geometry_dtype, shape="ntargets",
                        order="C")
        return parse("mean_curvature[itgt]")

    @property
    @memoize_method
    def src_derivative_dir(self):
        self.arguments["src_derivative_dir"] = \
                lp.GlobalArg("src_derivative_dir",
                        self.geometry_dtype, shape=("ntargets", self.dim),
                        order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("src_derivative_dir[itgt, %d]" % i)
            for i in range(self.dim)])

    @property
    @memoize_method
    def tgt_derivative_dir(self):
        self.arguments["tgt_derivative_dir"] = \
                lp.GlobalArg("tgt_derivative_dir",
                        self.geometry_dtype, shape=("ntargets", self.dim),
                        order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("tgt_derivative_dir[itgt, %d]" % i)
            for i in range(self.dim)])

# }}}

# }}}

# vim: fdm=marker
