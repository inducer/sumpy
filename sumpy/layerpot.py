from __future__ import division

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


import numpy as np
import loopy as lp
import sympy as sp
from pytools import memoize_method
from pymbolic import parse, var

from sumpy.tools import KernelComputation

import logging
logger = logging.getLogger(__name__)


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
    coefficients = expansion.coefficients_from_source(avec, bvec)

    assigned_coeffs = [
            sp.Symbol(
                    sac.assign_unique("expn%dcoeff%s" % (
                        expansion_nr, stringify_expn_index(i)),
                        coefficients[expansion.get_storage_index(i)]))
            for i in expansion.get_coefficient_indices()]

    return sac.assign_unique("expn%d_result" % expansion_nr,
            expansion.evaluate(assigned_coeffs, bvec))


# {{{ layer potential computation

# {{{ base class

class LayerPotentialBase(KernelComputation):
    def __init__(self, ctx, expansions, strength_usage=None,
            value_dtypes=None, strength_dtypes=None,
            geometry_dtype=None,
            options=[], name="layerpot", device=None):
        KernelComputation.__init__(self, ctx, expansions, strength_usage,
                value_dtypes, strength_dtypes, geometry_dtype,
                name, options, device)

        from pytools import single_valued
        self.dim = single_valued(knl.dim for knl in self.expansions)

    @property
    def expansions(self):
        return self.kernels

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

        from sumpy.symbolic import kill_trivial_assignments
        assignments = kill_trivial_assignments([
                (name, expr.subs("tau", 0))
                for name, expr in sac.assignments.iteritems()],
                retain_names=result_names)

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(assignments,
                vector_names=set(["a", "b"]),
                pymbolic_expr_maps=[
                    expn.kernel.transform_to_code for expn in self.expansions],
                complex_dtype=np.complex128  # FIXME
                )

        isrc_sym = var("isrc")
        exprs = [
                var(name)
                * self.get_strength_or_not(isrc_sym, i)
                for i, name in enumerate(result_names)]

        geo_dtype = self.geometry_dtype
        arguments = (
                [
                    lp.GlobalArg("src", geo_dtype,
                        shape=(self.dim, "nsrc"), order="C"),
                    lp.GlobalArg("tgt", geo_dtype,
                        shape=(self.dim, "ntgt"), order="C"),
                    lp.GlobalArg("center", geo_dtype,
                        shape=(self.dim, "ntgt"), order="C"),
                    lp.ValueArg("nsrc", np.int32),
                    lp.ValueArg("ntgt", np.int32),
                ] + self.get_input_and_output_arguments()
                + self.gather_kernel_arguments())

        loopy_knl = lp.make_kernel(self.device,
                "{[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
                "and 0<=idim<%d}" % self.dim,
                [
                "<> a[idim] = center[idim,itgt] - src[idim,isrc] {id=compute_a}",
                "<> b[idim] = tgt[idim,itgt] - center[idim,itgt] {id=compute_b}",
                ]+self.get_kernel_scaling_assignments()+loopy_insns+[
                    lp.Instruction(id=None,
                        assignee="pair_result_%d" % i, expression=expr,
                        temp_var_type=dtype)
                    for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]+self.get_result_store_instructions(),
                arguments,
                defines=dict(KNLIDX=range(len(exprs))),
                name=self.name, assumptions="nsrc>=1 and ntgt>=1",
                preambles=self.gather_kernel_preambles()
                )

        for where in ["compute_a", "compute_b"]:
            loopy_knl = lp.duplicate_inames(loopy_knl, "idim", where,
                    tags={"idim": "unr"})

        for expn in self.expansions:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        loopy_knl = lp.tag_data_axis(loopy_knl, "center", 0, "sep")

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME specialize/tune for GPU/CPU
        loopy_knl = self.get_kernel()

        import pyopencl as cl
        dev = self.context.devices[0]
        if dev.type == cl.device_type.CPU:
            loopy_knl = lp.split_iname(loopy_knl, "itgt", 16, outer_tag="g.0",
                    inner_tag="l.0")
            loopy_knl = lp.split_iname(loopy_knl, "isrc", 256)
            loopy_knl = lp.generate_loop_schedules(loopy_knl, [
                "isrc_outer", "itgt_inner"])
        else:
            from warnings import warn
            warn("don't know how to tune layer potential computation for '%s'" % dev)
            loopy_knl = lp.split_iname(loopy_knl, "itgt", 128, outer_tag="g.0")

        return loopy_knl

    @memoize_method
    def get_compiled_kernel(self):
        kernel = self.get_optimized_kernel()
        return lp.CompiledKernel(self.context, kernel)

# }}}


# {{{ direct applier

class LayerPotential(LayerPotentialBase):
    """Direct applier for the layer potential."""

    def get_strength_or_not(self, isrc, kernel_idx):
        return var("strength_%d" % self.strength_usage[kernel_idx])[isrc]

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("strength_%d" % i, dtype, shape="nsrc", order="C")
                for i, dtype in enumerate(self.strength_dtypes)
            ]+[
                lp.GlobalArg("result_%d" % i, dtype, shape="ntgt", order="C")
                for i, dtype in enumerate(self.value_dtypes)
            ]

    def get_result_store_instructions(self):
        return [
                "result_${KNLIDX}[itgt] = \
                        knl_${KNLIDX}_scaling*sum(isrc, pair_result_${KNLIDX})"
                ]

    def __call__(self, queue, targets, sources, centers, strengths, **kwargs):
        """
        :arg strengths: are required to have area elements and quadrature weights
            already multiplied in.
        """

        cknl = self.get_compiled_kernel()
        #print cknl.code

        for i, dens in enumerate(strengths):
            kwargs["strength_%d" % i] = dens

        return cknl(queue, src=sources, tgt=targets, center=centers, **kwargs)

# }}}


# {{{ matrix writer

class LayerPotentialMatrixGenerator(LayerPotentialBase):
    """Generator for layer potential matrix entries."""

    def get_strength_or_not(self, isrc, kernel_idx):
        return 1

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("result_%d" % i, dtype, shape="ntgt,nsrc", order="C")
                for i, dtype in enumerate(self.value_dtypes)
            ]

    def get_result_store_instructions(self):
        return [
                "result_${KNLIDX}[itgt, isrc] = \
                        knl_${KNLIDX}_scaling*pair_result_${KNLIDX}"
                ]

    def __call__(self, queue, targets, sources, centers, **kwargs):
        cknl = self.get_compiled_kernel()

        return cknl(queue, src=sources, tgt=targets, center=centers,
                **kwargs)

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
                        shape="ntgt", order="C")
        return parse("%s[itgt]" % self.density_var_name)

    @property
    @memoize_method
    def density_prime(self):
        prime_var_name = self.density_var_name+"_prime"
        self.arguments[prime_var_name] = \
                lp.GlobalArg(prime_var_name, self.density_dtype,
                        shape="ntgt", order="C")
        return parse("%s[itgt]" % prime_var_name)

    @property
    @memoize_method
    def side(self):
        self.arguments["side"] = \
                lp.GlobalArg("side", self.geometry_dtype, shape="ntgt", order="C")
        return parse("side[itgt]")

    @property
    @memoize_method
    def normal(self):
        self.arguments["normal"] = \
                lp.GlobalArg("normal", self.geometry_dtype,
                        shape=("ntgt", self.dim), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("normal[itgt, %d]" % i)
            for i in range(self.dim)])

    @property
    @memoize_method
    def tangent(self):
        self.arguments["tangent"] = \
                lp.GlobalArg("tangent", self.geometry_dtype,
                        shape=("ntgt", self.dim), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("tangent[itgt, %d]" % i)
            for i in range(self.dim)])

    @property
    @memoize_method
    def mean_curvature(self):
        self.arguments["mean_curvature"] = \
                lp.GlobalArg("mean_curvature",
                        self.geometry_dtype, shape="ntgt",
                        order="C")
        return parse("mean_curvature[itgt]")

    @property
    @memoize_method
    def src_derivative_dir(self):
        self.arguments["src_derivative_dir"] = \
                lp.GlobalArg("src_derivative_dir",
                        self.geometry_dtype, shape=("ntgt", self.dim),
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
                        self.geometry_dtype, shape=("ntgt", self.dim),
                        order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("tgt_derivative_dir[itgt, %d]" % i)
            for i in range(self.dim)])

# }}}

# }}}

# vim: fdm=marker
