from __future__ import division

import numpy as np
import loopy as lp
from pytools import memoize_method
from pymbolic import parse
import sympy as sp

from sumpy.tools import KernelComputation




def expand_line(kernel, order, avec, bvec):
    from pymbolic.sympy_conv import make_cse
    from pytools import factorial

    tau = sp.Symbol("tau")
    avec_line = avec + tau*bvec

    kernel_expr = kernel.get_expression(avec)

    from sumpy.symbolic import vector_subs
    line_kernel = vector_subs(kernel_expr, avec, avec_line)

    from sumpy.tools import DerivativeCache
    dcache = DerivativeCache(line_kernel)

    return sum(
            sp.sympify(1) # tau^n for tau = 1
            / factorial(i)
            * make_cse(dcache.diff_scalar(tau, i).subs(tau, 0), "taylor_%d" % i)
            for i in range(order+1))




def expand_volume(kernel, order, avec, bvec):
    dimensions = len(avec)

    tgt_derivative_multi_index = [0] * dimensions
    from sumpy.kernel import TargetDerivative
    while isinstance(kernel, TargetDerivative):
        tgt_derivative_multi_index[kernel.axis] += 1
        kernel = kernel.kernel

    kernel_expr = kernel.get_expression(avec)

    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most
            as gnitstam)

    multi_indices = sorted(gnitstam(order, dimensions), key=sum)

    from sumpy.tools import mi_factorial, mi_power, mi_derivative
    from pymbolic.sympy_conv import make_cse

    from sumpy.tools import DerivativeCache
    dcache = DerivativeCache(kernel_expr)

    return sum(
            mi_derivative(mi_power(bvec, mi), bvec, tgt_derivative_multi_index)
            / mi_factorial(mi)
            * make_cse(dcache.diff_vector(avec, mi),
                "taylor_" + "_".join(str(mi_i) for mi_i in mi))
            for mi in multi_indices)




# {{{ layer potential applier

class LayerPotential(KernelComputation):
    def __init__(self, ctx, kernels, order, density_usage=None,
            value_dtypes=None, density_dtypes=None,
            geometry_dtype=None,
            options=[], name="layerpot", device=None):
        KernelComputation.__init__(self, ctx, kernels, density_usage,
                value_dtypes, density_dtypes, geometry_dtype,
                name, options, device)

        from pytools import single_valued
        self.dimensions = single_valued(knl.dimensions for knl in self.kernels)

        self.order = order

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector

        avec = make_sym_vector("a", self.dimensions)
        bvec = make_sym_vector("b", self.dimensions)

        from sumpy.codegen import sympy_to_pymbolic_for_code
        exprs = sympy_to_pymbolic_for_code(
                [expand_line(k, self.order, avec, bvec)
                    for i, k in enumerate(self.kernels)])
        exprs = [knl.transform_to_code(expr) for knl, expr in zip(
            self.kernels, exprs)]
        from pymbolic import var
        isrc_sym = var("isrc")
        exprs = [
                expr
                * var("density_%d" % self.strength_usage[i])[isrc_sym]
                * var("speed")[isrc_sym]
                * var("weights")[isrc_sym]
                for i, expr in enumerate(exprs)]

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
                ]+[
                   lp.GlobalArg("density_%d" % i, dtype, shape="nsrc", order="C")
                   for i, dtype in enumerate(self.strength_dtypes)
                ]+[
                   lp.GlobalArg("result_%d" % i, dtype, shape="ntgt", order="C")
                   for i, dtype in enumerate(self.value_dtypes)
                   ]
                + self.gather_kernel_arguments())

        knl = lp.make_kernel(self.device,
                "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
                "and 0<=idim<%d}" % self.dimensions,
                [
                "[|idim] <%s> a[idim] = center[itgt,idim] - src[isrc,idim]" % geo_dtype.name,
                "[|idim] <%s> b[idim] = tgt[itgt,idim] - center[itgt,idim]" % geo_dtype.name,
                ]+self.get_kernel_scaling_assignments()+[
                lp.Instruction(id=None,
                    assignee="pair_result_%d" % i, expression=expr,
                    temp_var_type=dtype)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]+[
                "result_%d[itgt] = knl_%d_scaling*sum_%s(isrc, pair_result_%d)"
                % (i, i, dtype.name, i)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))],
                arguments,
                name=self.name, assumptions="nsrc>=1 and ntgt>=1",
                preambles=self.gather_kernel_preambles())

        return knl

    def get_optimized_kernel(self):
        # FIXME specialize/tune for GPU/CPU
        knl = self.get_kernel()
        knl = lp.split_dimension(knl, "itgt", 128, outer_tag="g.0")
        return knl

    @memoize_method
    def get_compiled_kernel(self):
        return lp.CompiledKernel(self.context, self.get_optimized_kernel())

    def __call__(self, queue, targets, sources, centers, densities,
            speed, weights, **kwargs):
        cknl = self.get_compiled_kernel()

        for i, dens in enumerate(densities):
            kwargs["density_%d" % i] = dens

        return cknl(queue, src=sources, tgt=targets, center=centers,
                speed=speed, weights=weights, nsrc=len(sources), ntgt=len(targets),
                **kwargs)

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
        knl = lp.make_kernel(self.device,
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

        return knl, data_args

    def get_optimized_kernel(self):
        # FIXME specialize/tune for GPU/CPU
        knl, data_args = self.get_kernel()
        knl = lp.split_dimension(knl, "itgt", 128, outer_tag="g.0")
        return knl, data_args

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
