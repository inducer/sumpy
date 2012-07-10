from __future__ import division

import numpy as np
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
        dvec = make_sym_vector("d", self.dimensions)

        from sumpy.assignment_collection import SymbolicAssignmentCollection
        sac = SymbolicAssignmentCollection()

        result_names = [
                sac.assign_unique("knl%d" % i,
                    knl.postprocess_at_target(
                        knl.postprocess_at_source(
                            knl.get_expression(dvec), dvec),
                        dvec))
                        for i, knl in enumerate(self.kernels)]

        sac.run_global_cse()

        from sumpy.codegen import to_loopy_insns
        loopy_insns = to_loopy_insns(sac.assignments.iteritems(),
                vector_names=set(["d"]),
                pymbolic_expr_maps=[knl.transform_to_code for knl in self.kernels])

        from pymbolic import var
        exprs = [
                var(name)
                    * var("strength_%d" % self.strength_usage[i])[var("isrc")]
                for i, name in enumerate(result_names)]

        arguments = (
                [
                   lp.GlobalArg("src", self.geometry_dtype, shape=("nsrc", self.dimensions), order="C"),
                   lp.GlobalArg("tgt", self.geometry_dtype, shape=("ntgt", self.dimensions), order="C"),
                   lp.ScalarArg("nsrc", np.int32),
                   lp.ScalarArg("ntgt", np.int32),
                ]+[
                   lp.GlobalArg("strength_%d" % i, dtype, shape="nsrc", order="C")
                   for i, dtype in enumerate(self.strength_dtypes)
                ]+[
                   lp.GlobalArg("result_%d" % i, dtype, shape="ntgt", order="C")
                   for i, dtype in enumerate(self.value_dtypes)
                   ]
                + self.gather_kernel_arguments())

        if self.exclude_self:
            from pymbolic.primitives import If, ComparisonOperator, Variable
            exprs = [
                    If(
                        ComparisonOperator(Variable("isrc"), "!=", Variable("itgt")),
                        expr, 0)
                    for expr in exprs]

        loopy_knl = lp.make_kernel(self.device,
                "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
                "and 0<=idim<%d}" % self.dimensions,
                self.get_kernel_scaling_assignments()
                + loopy_insns
                + [
                "[|idim] <> d[idim] = tgt[itgt,idim] - src[isrc,idim]",
                ]+[
                lp.Instruction(id=None,
                    assignee="pair_result_%d" % i, expression=expr,
                    temp_var_type=dtype)
                for i, (expr, dtype) in enumerate(zip(exprs, self.value_dtypes))
                ]+[
                "result_{KNLIDX}[itgt] = knl_{KNLIDX}_scaling*sum(isrc, pair_result_{KNLIDX})"
                ],
                arguments,
                name=self.name, assumptions="nsrc>=1 and ntgt>=1",
                preambles=self.gather_kernel_preambles(),
                defines=dict(KNLIDX=range(len(exprs))))

        for knl in self.kernels:
            loopy_knl = knl.prepare_loopy_kernel(loopy_knl)

        return loopy_knl

    def get_optimized_kernel(self):
        # FIXME
        knl = self.get_kernel()
        knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")
        return knl

    @memoize_method
    def get_compiled_kernel(self):
        return lp.CompiledKernel(self.context, self.get_optimized_kernel())

    def __call__(self, queue, targets, sources, src_strengths, **kwargs):
        cknl = self.get_compiled_kernel()
        #print cknl.code
        #1/0

        for i, sstr in enumerate(src_strengths):
            kwargs["strength_%d" % i] = sstr

        return cknl(queue, src=sources, tgt=targets,
                nsrc=len(sources), ntgt=len(targets), **kwargs)
