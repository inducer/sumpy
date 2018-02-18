from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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


import six
from six.moves import range, zip
import numpy as np
import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
import sumpy.symbolic as sym
from pytools import memoize_method
from pymbolic import parse, var

from sumpy.tools import KernelComputation, KernelCacheWrapper

import logging
logger = logging.getLogger(__name__)


__doc__ = """

QBX for Layer Potentials
------------------------

.. autoclass:: LayerPotentialBase
.. autoclass:: LayerPotential
.. autoclass:: LayerPotentialMatrixGenerator

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


def expand(expansion_nr, sac, expansion, avec, bvec):
    rscale = sym.Symbol("rscale")

    coefficients = expansion.coefficients_from_source(avec, bvec, rscale)

    assigned_coeffs = [
            sym.Symbol(
                    sac.assign_unique("expn%dcoeff%s" % (
                        expansion_nr, stringify_expn_index(i)),
                        coefficients[expansion.get_storage_index(i)]))
            for i in expansion.get_coefficient_identifiers()]

    return sac.assign_unique("expn%d_result" % expansion_nr,
            expansion.evaluate(assigned_coeffs, bvec, rscale))


# {{{ layer potential computation

# {{{ base class

class LayerPotentialBase(KernelComputation, KernelCacheWrapper):
    def __init__(self, ctx, expansions, strength_usage=None,
            value_dtypes=None,
            options=[], name="layerpot", device=None):
        KernelComputation.__init__(self, ctx, expansions, strength_usage,
                value_dtypes,
                name, options, device)

        from pytools import single_valued
        self.dim = single_valued(knl.dim for knl in self.expansions)

    def get_cache_key(self):
        return (type(self).__name__, tuple(self.kernels),
                tuple(self.strength_usage), tuple(self.value_dtypes))

    @property
    def expansions(self):
        return self.kernels

    def get_compute_a_and_b_vecs(self):
        return """
            <> a[idim] = center[idim,itgt] - src[idim,isrc] {dup=idim}
            <> b[idim] = tgt[idim,itgt] - center[idim,itgt] {dup=idim}
            <> rscale = expansion_radii[itgt]
            """

    def get_src_tgt_arguments(self):
        return [
                lp.GlobalArg("src", None,
                    shape=(self.dim, "nsources"), order="C"),
                lp.GlobalArg("tgt", None,
                    shape=(self.dim, "ntargets"), order="C"),
                lp.GlobalArg("center", None,
                    shape=(self.dim, "ntargets"), order="C"),
                lp.GlobalArg("expansion_radii", None, shape="ntargets"),
                lp.ValueArg("nsources", np.int32),
                lp.ValueArg("ntargets", np.int32),
                ]

    def get_domains(self):
        return [
                "{[itgt]: 0 <= itgt < ntargets}",
                "{[isrc]: 0 <= isrc < nsources}",
                "{[idim]: 0 <= idim < dim}"
                ]

    def get_loop_begin(self):
        return ["for itgt, isrc"]

    def get_loop_end(self):
        return ["end"]

    def get_assumptions(self):
        return "nsources>=1 and ntargets>=1"

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector

        avec = make_sym_vector("a", self.dim)
        bvec = make_sym_vector("b", self.dim)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        logger.info("compute expansion expressions: start")

        result_names = [expand(i, sac, expn, avec, bvec)
                for i, expn in enumerate(self.expansions)]

        logger.info("compute expansion expressions: done")

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(
                six.iteritems(sac.assignments),
                vector_names=set(["a", "b"]),
                pymbolic_expr_maps=[
                    expn.kernel.get_code_transformer() for expn in self.expansions],
                retain_names=result_names,
                complex_dtype=np.complex128  # FIXME
                )

        isrc_sym = var("isrc")
        exprs = [
                var(name)
                * self.get_strength_or_not(isrc_sym, i)
                for i, name in enumerate(result_names)]

        from sumpy.tools import gather_loopy_source_arguments
        arguments = (
                self.get_src_tgt_arguments()
                + self.get_input_and_output_arguments()
                + gather_loopy_source_arguments(self.kernels))

        loopy_knl = lp.make_kernel(
                self.get_domains(),
                self.get_kernel_scaling_assignments()
                + self.get_loop_begin()
                + [self.get_compute_a_and_b_vecs()]
                + loopy_insns
                + [
                    lp.Assignment(id=None,
                        assignee="pair_result_%d" % i, expression=expr,
                        temp_var_type=lp.auto)
                    for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]
                + self.get_loop_end()
                + self.get_result_store_instructions(),
                arguments,
                name=self.name,
                assumptions=self.get_assumptions(),
                default_offset=lp.auto,
                silenced_warnings="write_race(write_lpot*)",
                fixed_parameters=dict(dim=self.dim),
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

        loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

        for expn in self.expansions:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_array_axes(loopy_knl, "center", "sep,C")

        return loopy_knl

    @memoize_method
    def get_optimized_kernel(self):
        # FIXME specialize/tune for GPU/CPU
        loopy_knl = self.get_kernel()
        return loopy_knl

        # FIXME: how to tune for blocks?
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
    """Direct applier for the layer potential."""

    default_name = "qbx_apply"

    def get_strength_or_not(self, isrc, kernel_idx):
        return var("strength_%d" % self.strength_usage[kernel_idx]).index(isrc)

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("strength_%d" % i, None, shape="nsources", order="C")
                for i in range(self.strength_count)
                ]+[
                lp.GlobalArg("result_%d" % i, None, shape="ntargets", order="C")
                for i in range(len(self.kernels))
                ]

    def get_result_store_instructions(self):
        return [
                """
                result_{i}[itgt] = \
                        knl_{i}_scaling*simul_reduce(
                            sum, isrc, pair_result_{i}) \
                                    {{id_prefix=write_lpot,inames=itgt}}
                """.format(i=iknl)
                for iknl in range(len(self.expansions))
                ]

    def __call__(self, queue, targets, sources, centers, strengths, expansion_radii,
            **kwargs):
        """
        :arg strengths: are required to have area elements and quadrature weights
            already multiplied in.
        """

        knl = self.get_cached_optimized_kernel()

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

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("result_%d" % i, dtype, shape="ntargets,nsources")
                for i, dtype in enumerate(self.value_dtypes)
                ]

    def get_result_store_instructions(self):
        return [
                """
                result_KNLIDX[itgt, isrc] = \
                        knl_KNLIDX_scaling*pair_result_KNLIDX  {inames=isrc:itgt}
                """.replace("KNLIDX", str(iknl))
                for iknl in range(len(self.expansions))
                ]

    def __call__(self, queue, targets, sources, centers, expansion_radii, **kwargs):
        knl = self.get_optimized_kernel()

        return knl(queue, src=sources, tgt=targets, center=centers,
                expansion_radii=expansion_radii, **kwargs)

# }}}


# {{{

class LayerPotentialMatrixBlockGenerator(LayerPotentialBase):
    default_name = "qbx_block"

    def get_src_tgt_arguments(self):
        return LayerPotentialBase.get_src_tgt_arguments(self) \
            + [
                lp.GlobalArg("srcindices", None, shape="nsrcindices"),
                lp.GlobalArg("tgtindices", None, shape="ntgtindices"),
                lp.GlobalArg("srcranges", None, shape="nranges"),
                lp.GlobalArg("tgtranges", None, shape="nranges"),
                lp.ValueArg("nsrcindices", np.int32),
                lp.ValueArg("ntgtindices", np.int32),
                lp.ValueArg("nranges", None)
            ]

    def get_domains(self):
        # FIXME: this doesn't work when separating j and k
        return [
                "{[i]: 0 <= i < nranges - 1}",
                "{[j, k]: 0 <= j < tgt_length and 0 <= k < src_length}",
                "{[idim]: 0 <= idim < dim}"
                ]

    def get_loop_begin(self):
        return [
                """
                for i
                    <> tgtstart = tgtranges[i]
                    <> tgtend = tgtranges[i + 1]
                    <> tgt_length = tgtend - tgtstart
                    <> srcstart = srcranges[i]
                    <> srcend = srcranges[i + 1]
                    <> src_length = srcend - srcstart
                    for j, k
                        <> itgt = tgtindices[tgtstart + j]
                        <> isrc = srcindices[srcstart + k]
                """
                ]

    def get_loop_end(self):
        return [
                """
                    end
                end
                """
                ]

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("result_%d" % i, dtype, shape="ntgtindices,nsrcindices")
                for i, dtype in enumerate(self.value_dtypes)
                ]

    def get_result_store_instructions(self):
        return [
                """
                result_KNLIDX[tgtstart + j, srcstart + k] = \
                        knl_KNLIDX_scaling*pair_result_KNLIDX  {inames=i}
                """.replace("KNLIDX", str(iknl))
                for iknl in range(len(self.expansions))
                ]

    def get_assumptions(self):
        return "nranges>=2"

    def __call__(self, queue, targets, sources, centers, expansion_radii,
            tgtindices, srcindices, tgtranges, srcranges, **kwargs):
        knl = self.get_optimized_kernel()

        return knl(queue, src=sources, tgt=targets, center=centers,
                expansion_radii=expansion_radii,
                tgtindices=tgtindices, srcindices=srcindices,
                tgtranges=tgtranges, srcranges=srcranges, **kwargs)

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

class _JumpTermSymbolicArgumentProvider(object):
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
