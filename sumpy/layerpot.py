from __future__ import division

import numpy as np
import loopy as lp
import sympy as sp
from pytools import memoize_method
from pymbolic import parse, var

from sumpy.tools import KernelComputation




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
    def __init__(self, ctx, expansions, density_usage=None,
            value_dtypes=None, density_dtypes=None,
            geometry_dtype=None,
            options=[], name="layerpot", device=None,
            optimize=False):
        KernelComputation.__init__(self, ctx, expansions, density_usage,
                value_dtypes, density_dtypes, geometry_dtype,
                name, options, device)

        from pytools import single_valued
        self.dimensions = single_valued(knl.dimensions for knl in self.expansions)
        self.optimize = optimize

    @property
    def expansions(self):
        return self.kernels

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector

        avec = make_sym_vector("a", self.dimensions)
        bvec = make_sym_vector("b", self.dimensions)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        result_names = [expand(i, sac, expn, avec, bvec)
                for i, expn in enumerate(self.expansions)]
        if self.optimize:
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
                    expn.kernel.transform_to_code
                    for expn in self.expansions])

        isrc_sym = var("isrc")
        exprs = [
                var(name)
                * self.get_density_or_not(isrc_sym, i)
                * var("speed")[isrc_sym]
                * var("weights")[isrc_sym]
                for i, name in enumerate(result_names)]

        geo_dtype = self.geometry_dtype
        arguments = (
                [
                   lp.GlobalArg("src", geo_dtype, shape=("nsrc", self.dimensions), order="C"),
                   lp.GlobalArg("tgt", geo_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.GlobalArg("center", geo_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.GlobalArg("speed", geo_dtype, shape="nsrc", order="C"),
                   lp.GlobalArg("weights", geo_dtype, shape="nsrc", order="C"),
                   lp.ScalarArg("nsrc", np.int32),
                   lp.ScalarArg("ntgt", np.int32),
                ] + self.get_input_and_output_arguments()
                + self.gather_kernel_arguments())

        loopy_knl = lp.make_kernel(self.device,
                "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
                "and 0<=idim<%d}" % self.dimensions,
                [
                "[|idim] <> a[idim] = center[itgt,idim] - src[isrc,idim]",
                "[|idim] <> b[idim] = tgt[itgt,idim] - center[itgt,idim]",
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
        for expn in self.expansions:
            loopy_knl = expn.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME specialize/tune for GPU/CPU
        loopy_knl = self.get_kernel()

        import pyopencl as cl
        dev = self.context.devices[0]
        if dev.type == cl.device_type.CPU:
            loopy_knl = lp.split_dimension(loopy_knl, "itgt", 16, outer_tag="g.0",
                    inner_tag="l.0")
            loopy_knl = lp.split_dimension(loopy_knl, "isrc", 256)
            loopy_knl = lp.generate_loop_schedules(loopy_knl, [
                "isrc_outer", "itgt_inner"])
        else:
            from warnings import warn
            warn("don't know how to tune layer potential computation for '%s'" % dev)
            loopy_knl = lp.split_dimension(loopy_knl, "itgt", 128, outer_tag="g.0")

        return loopy_knl

    @memoize_method
    def get_compiled_kernel(self):
        kernel = self.get_optimized_kernel()
        return lp.CompiledKernel(self.context, kernel)

# }}}

# {{{ direct applier

class LayerPotential(LayerPotentialBase):
    """Direct applier for the layer potential."""

    def get_density_or_not(self, isrc, kernel_idx):
        return var("density_%d" % self.strength_usage[kernel_idx])[isrc]

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("density_%d" % i, dtype, shape="nsrc", order="C")
                for i, dtype in enumerate(self.strength_dtypes)
            ]+[
                lp.GlobalArg("result_%d" % i, dtype, shape="ntgt", order="C")
                for i, dtype in enumerate(self.value_dtypes)
            ]

    def get_result_store_instructions(self):
        return [
                "result_{KNLIDX}[itgt] = knl_{KNLIDX}_scaling*sum(isrc, pair_result_{KNLIDX})"
                ]

    def __call__(self, queue, targets, sources, centers, densities,
            speed, weights, **kwargs):
        cknl = self.get_compiled_kernel()
        #print cknl.code

        for i, dens in enumerate(densities):
            kwargs["density_%d" % i] = dens

        return cknl(queue, src=sources, tgt=targets, center=centers,
                speed=speed, weights=weights, nsrc=len(sources), ntgt=len(targets),
                **kwargs)

# }}}

# {{{ matrix writer

class LayerPotentialMatrixGenerator(LayerPotentialBase):
    """Generator for layer potential matrix entries."""

    def get_density_or_not(self, isrc, kernel_idx):
        return 1

    def get_input_and_output_arguments(self):
        return [
                lp.GlobalArg("result_%d" % i, dtype, shape="ntgt,nsrc", order="C")
                for i, dtype in enumerate(self.value_dtypes)
            ]

    def get_result_store_instructions(self):
        return [
                "result_{KNLIDX}[itgt, isrc] = knl_{KNLIDX}_scaling*pair_result_{KNLIDX}"
                ]

    def __call__(self, queue, targets, sources, centers, speed, weights, **kwargs):
        cknl = self.get_compiled_kernel()

        return cknl(queue, src=sources, tgt=targets, center=centers,
                speed=speed, weights=weights, nsrc=len(sources), ntgt=len(targets),
                **kwargs)

# }}}

# }}}

# {{{ jump term handling

def find_jump_term(kernel, arg_provider):
    from sumpy.kernel import TargetDerivative, SourceDerivative

    tgt_derivatives = []
    source_diff_vectors = []

    while isinstance(kernel, (TargetDerivative, SourceDerivative)):
        if isinstance(kernel, TargetDerivative):
            tgt_derivatives.append(kernel.axis)
            kernel = kernel.kernel
        elif isinstance(kernel, SourceDerivative):
            source_diff_vectors.append(
                    (kernel.dir_vec_name, kernel.dir_vec_dtype))
            kernel = kernel.kernel

    tgt_count = len(tgt_derivatives)
    src_count = len(source_diff_vectors)

    info = arg_provider

    if src_count == 0:
        if tgt_count == 0:
            return 0
        elif tgt_count == 1:
            i, = tgt_derivatives
            return info.side/2 * info.normal[i] * info.density
        elif tgt_count == 2:
            i, j = tgt_derivatives

            from pytools import delta
            return (
                    - info.side * info.mean_curvature / 2
                    * (-delta(i, j) + 2*info.normal[i]*info.normal[j])
                    * info.density

                    + info.side / 2
                    * (info.normal[i]*info.tangent[j] + info.normal[j]*info.tangent[i])
                    * info.density_prime)

    elif src_count == 1:
        if tgt_count == 0:
            return (
                    - info.side/2
                    * np.dot(info.normal, info.src_derivative_dir)
                    * info.density)
        elif tgt_count == 1:
            from warnings import warn
            warn("jump terms for mixed derivatives (1 src+1 tgt) only available "
                    "for the double-layer potential")
            i, = tgt_derivatives
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

    def __init__(self, data_args, dimensions, density_var_name,
            density_dtype, geometry_dtype):
        # list of loopy arguments
        self.arguments = data_args
        self.dimensions = dimensions
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
                        shape=("ntgt", self.dimensions), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("normal[itgt, %d]" % i)
            for i in range(self.dimensions)])

    @property
    @memoize_method
    def tangent(self):
        self.arguments["tangent"] = \
                lp.GlobalArg("tangent", self.geometry_dtype,
                        shape=("ntgt", self.dimensions), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("tangent[itgt, %d]" % i)
            for i in range(self.dimensions)])

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
                        self.geometry_dtype, shape=("ntgt", self.dimensions), order="C")
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            parse("src_derivative_dir[itgt, %d]" % i)
            for i in range(self.dimensions)])

# }}}




class JumpTermApplier(KernelComputation):
    def __init__(self, ctx, kernels, density_usage=None,
            value_dtypes=None, density_dtypes=None,
            geometry_dtype=None,
            options=[], name="jump_term", device=None):
        KernelComputation.__init__(self, ctx, kernels, density_usage,
                value_dtypes, density_dtypes, geometry_dtype,
                name, options, device)

        from pytools import single_valued
        self.dimensions = single_valued(knl.dimensions for knl in self.kernels)

    def get_kernel(self):
        data_args = {}

        exprs = [find_jump_term(
                    k,
                    _JumpTermSymbolicArgumentProvider(data_args,
                        self.dimensions,
                        "density_%d" % self.strength_usage[i],
                        self.strength_dtypes[self.strength_usage[i]],
                        self.geometry_dtype))
                    for i, k in enumerate(self.kernels)]

        data_args = list(data_args.itervalues())

        operand_args = [
            lp.GlobalArg("result_%d" % i, dtype, shape="ntgt", order="C")
            for i, dtype in enumerate(self.value_dtypes)
            ]+[
            lp.GlobalArg("limit_%d" % i, dtype, shape="ntgt", order="C")
            for i, dtype in enumerate(self.value_dtypes)]

        # FIXME (fast) special case for jump term == 0
        loopy_knl = lp.make_kernel(self.device,
                "[ntgt] -> {[itgt]: 0<=itgt<ntgt}",
                [
                lp.Instruction(id=None,
                    assignee="temp_result_%d" % i, expression=expr,
                    temp_var_type=dtype)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]+[
                "result_%d[itgt] = limit_%d[itgt] + temp_result_%d" % (i, i, i)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))],
                [
                   lp.ScalarArg("ntgt", np.int32),
                ] + operand_args + data_args,
                name=self.name, assumptions="ntgt>=1")

        return loopy_knl, data_args

    def get_optimized_kernel(self):
        # FIXME specialize/tune for GPU/CPU
        loopy_knl, data_args = self.get_kernel()
        loopy_knl = lp.split_dimension(loopy_knl, "itgt", 128, outer_tag="g.0")
        return loopy_knl, data_args

    @memoize_method
    def get_compiled_kernel(self):
        opt_knl, data_args = self.get_optimized_kernel()
        return lp.CompiledKernel(self.context, opt_knl), data_args

    def __call__(self, queue, limits, argument_provider):
        cknl, data_args = self.get_compiled_kernel()

        kwargs = {}
        for i, limit in enumerate(limits):
            kwargs["limit_%d" % i] = limit

        for arg in data_args:
            kwargs[arg.name] = getattr(argument_provider, arg.name)

        return cknl(queue, ntgt=len(limits[0]), **kwargs)

# }}}

# vim: fdm=marker
