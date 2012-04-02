from __future__ import division

import numpy as np
import loopy as lp
from pytools import memoize_method

from sumpy.tools import KernelComputation




def pop_expand(kernel, order, avec, bvec):
    dimensions = len(avec)
    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most
            as gnitstam)

    multi_indices = sorted(gnitstam(order, dimensions), key=sum)

    from sumpy.tools import mi_factorial, mi_power, mi_derivative
    from pymbolic.sympy_conv import make_cse
    return sum(
            mi_power(bvec, mi)/mi_factorial(mi) 
            #* (-1)**sum(mi) # we're expanding K(-a)
            * make_cse(mi_derivative(kernel, avec, mi),
                "taylor_" + "_".join(str(mi_i) for mi_i in mi))
            for mi in multi_indices)




class LayerPotential(KernelComputation):
    def __init__(self, ctx, kernels, order, density_usage=None,
            value_dtypes=None, strength_dtypes=None,
            geometry_dtype=None,
            options=[], name="layerpot", device=None):
        """
        :arg kernels: list of :class:`sumpy.kernel.Kernel` instances
        :arg density_usage: A list of integers indicating which expression
          uses which density. This implicitly specifies the
          number of density arrays that need to be passed.
          Default: all kernels use the same density.
        """
        KernelComputation.__init__(self, ctx, kernels, density_usage,
                value_dtypes, strength_dtypes, geometry_dtype,
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
                [pop_expand(k.get_expression(avec), self.order, avec, bvec)
                    for i, k in enumerate(self.kernels)])
        from pymbolic import var
        isrc_sym = var("isrc")
        exprs = [
                expr
                * var("density_%d" % i)[isrc_sym]
                * var("speed")[isrc_sym]
                * var("weights")[isrc_sym]
                for i, expr in enumerate(exprs)]

        geo_dtype = self.geometry_dtype
        arguments = (
                [
                   lp.ArrayArg("src", geo_dtype, shape=("nsrc", self.dimensions), order="C"),
                   lp.ArrayArg("tgt", geo_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.ArrayArg("center", geo_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.ArrayArg("speed", geo_dtype, shape="nsrc", order="C"),
                   lp.ArrayArg("weights", geo_dtype, shape="nsrc", order="C"),
                   lp.ScalarArg("nsrc", np.int32),
                   lp.ScalarArg("ntgt", np.int32),
                ]+[
                   lp.ArrayArg("density_%d" % i, dtype, shape="nsrc", order="C")
                   for i, dtype in enumerate(self.strength_dtypes)
                ]+[
                   lp.ArrayArg("result_%d" % i, dtype, shape="ntgt", order="C")
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
                name="layerpot", assumptions="nsrc>=1 and ntgt>=1",
                preamble=self.gather_kernel_preambles())

        return knl

    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")
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
