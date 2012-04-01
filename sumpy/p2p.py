from __future__ import division

import numpy as np
import pyopencl as cl
import loopy as lp
from pytools import memoize_method

from sumpy.tools import KernelComputation




# LATER:
# - Optimization for source == target (postpone)




class P2P(KernelComputation):
    def __init__(self, ctx, kernels,  exclude_self, strength_usage=None,
            value_dtypes=None, strength_dtypes=None,
            geometry_dtype=None, options=[], name="p2p", device=None):
        """
        :arg kernels: list of :class:`sumpy.kernel.Kernel` instances
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """
        KernelComputation.__init__(self, ctx, kernels, strength_usage,
                value_dtypes, strength_dtypes, geometry_dtype,
                name, options, device)

        self.exclude_self = exclude_self

        from pytools import single_valued
        self.dimensions = single_valued(knl.dimensions for knl in self.kernels)

    @memoize_method
    def get_kernel(self):
        from sumpy.symbolic import make_sym_vector
        avec = make_sym_vector("d", self.dimensions)

        from sumpy.codegen import sympy_to_pymbolic_for_code
        exprs = sympy_to_pymbolic_for_code(
                [k.get_expression(avec) for  k in self.kernels])
        from pymbolic import var
        exprs = [var("strength_%d" % i)[var("isrc")]*expr
                for i, expr in enumerate(exprs)]

        arguments = (
                [
                   lp.ArrayArg("src", self.geometry_dtype, shape=("nsrc", self.dimensions), order="C"),
                   lp.ArrayArg("tgt", self.geometry_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.ScalarArg("nsrc", np.int32),
                   lp.ScalarArg("ntgt", np.int32),
                ]+[
                   lp.ArrayArg("strength_%d" % i, dtype, shape="nsrc", order="C")
                   for i, dtype in enumerate(self.strength_dtypes)
                ]+[
                   lp.ArrayArg("result_%d" % i, dtype, shape="ntgt", order="C")
                   for i, dtype in enumerate(self.value_dtypes)
                   ]
                + self.gather_arguments())

        if self.exclude_self:
            from pymbolic.primitives import If, ComparisonOperator, Variable
            exprs = [
                    If(
                        ComparisonOperator(Variable("i"), "!=", Variable("j")),
                        expr, 0)
                    for expr in exprs]

        from pymbolic import parse
        return lp.make_kernel(self.device,
                "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
                "and 0<=idim<%d}" % self.dimensions,
                [
                "[|idim] <%s> d[idim] = tgt[itgt,idim] - src[isrc,idim]" 
                % self.geometry_dtype.name,
                ]+[
                lp.Instruction(id=None,
                    assignee=parse("pair_result_%d" % i), expression=expr,
                    temp_var_type=dtype)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]+[
                "result_%d[itgt] = sum_%s(isrc, pair_result_%d)" % (i, dtype.name, i)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))],
                arguments,
                name=self.name, assumptions="nsrc>=1 and ntgt>=1",
                preamble=self.gather_preambles())

    def get_optimized_kernel(self):
        knl = self.get_kernel()
        knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")
        return knl

    def __call__(self, queue, targets, sources, src_strengths, **kwargs):
        cknl = self.get_compiled_kernel()

        for i, sstr in enumerate(src_strengths):
            kwargs["strength_%d" % i] = sstr

        return cknl(queue, src=sources, tgt=targets,
                nsrc=len(sources), ntgt=len(targets), **kwargs)
