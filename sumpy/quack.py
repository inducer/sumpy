from __future__ import division

import numpy as np
import loopy as lp
import pyopencl as cl
import sympy as sp

from pymbolic.mapper import IdentityMapper
import pymbolic.primitives as prim

from pytools import memoize_method




def quack_expand(kernel, order, avec, bvec):
    dimensions = len(avec)
    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most
            as gnitstam)

    multi_indices = sorted(gnitstam(order, dimensions), key=sum)

    from sumpy.tools import mi_factorial, mi_power, mi_derivative
    return sum(
            mi_power(bvec, mi)/mi_factorial(mi) 
            * (-1)**sum(mi) # we're expanding K(-a)
            * mi_derivative(kernel, avec, mi)
            for mi in multi_indices)




# {{{ hankel handling

class HankelGetter(object):
    @memoize_method
    def hank1_01(self, arg):
        return prim.CommonSubexpression(
                prim.Variable("hank1_01")(arg))

    @memoize_method
    def hank1(self, order, arg):
        if order == 0:
            return prim.Lookup(self.hank1_01(arg), "order0")
        elif order == 1:
            return prim.Lookup(self.hank1_01(arg), "order1")
        elif order < 0:
            # AS (9.1.6)
            nu = -order
            return prim.wrap_in_cse(
                    (-1)**nu * self.hank1(nu, arg),
                    "hank1_neg%d" % nu)
        elif order > 1:
            # AS (9.1.27)
            nu = order-1
            return prim.CommonSubexpression(
                    2*nu/arg*self.hank1(nu, arg)
                    - self.hank1(nu-1, arg),
                    "hank1_%d" % order)
        else:
            assert False

    @memoize_method
    def hank1_deriv(self, order, arg, n_derivs):
        # AS (9.1.31)
        k = n_derivs
        nu = order
        return prim.CommonSubexpression(
                2**(-k)*sum(
                    (-1)**idx*int(sp.binomial(k, idx)) * self.hank1(i, arg)
                    for idx, i in enumerate(range(nu-k, nu+k+1, 2))),
                "d%d_hank1_%d" % (n_derivs, order))





class HankelSubstitutor(IdentityMapper):
    def __init__(self, hank_getter):
        self.hank_getter = hank_getter

    def map_call(self, expr):
        if isinstance(expr.function, prim.Variable) and expr.function.name == "H1_0":
            hank_arg, = expr.parameters
            return self.hank_getter.hank1(0, hank_arg)
        else:
            return IdentityMapper.map_call(self, expr)


    def map_substitution(self, expr):
        assert isinstance(expr.child, prim.Derivative)
        call = expr.child.child

        if isinstance(call.function, prim.Variable) and call.function.name == "H1_0":
            hank_arg, = expr.values
            return self.hank_getter.hank1_deriv(0, hank_arg,
                    len(expr.child.variables))
        else:
            return IdentityMapper.map_substitution(self, expr)

# }}}




class PowerRewriter(IdentityMapper):
    def map_power(self, expr):
        exp = expr.exponent
        if isinstance(exp, int):
            new_base = prim.wrap_in_cse(expr.base)

            if exp > 1:
                return self.rec(prim.wrap_in_cse(new_base**(exp-1))*new_base)
            elif exp == 1:
                return new_base
            elif exp < 0:
                return self.rec((1/new_base)**(-exp))

        if (isinstance(expr.exponent, prim.Quotient)
                and isinstance(expr.exponent.numerator, int)
                and isinstance(expr.exponent.denominator, int)):

            p, q = expr.exponent.numerator, expr.exponent.denominator
            if q < 0:
                q *= -1
                p *= -1

            if q == 1:
                return self.rec(new_base**p)

            if q == 2:
                assert p != 0

                if p > 0:
                    orig_base = prim.wrap_in_cse(expr.base)
                    new_base = prim.wrap_in_cse(
                            prim.Variable("rsqrt")(orig_base) * orig_base)
                else:
                    new_base = prim.wrap_in_cse(prim.Variable("rsqrt")(expr.base))
                    p *= -1

                return self.rec(new_base**p)

        return IdentityMapper.map_power(self, expr)





class FractionKiller(IdentityMapper):
    def map_quotient(self, expr):
        num = expr.numerator
        denom = expr.denominator

        if isinstance(num, int) and isinstance(denom, int):
            if num % denom == 0:
                return num // denom
            return int(expr.numerator) / int(expr.denominator)

        return IdentityMapper.map_quotient(self, expr)




def test_quack(ctx_factory):
    geo_dtype = np.float64
    value_dtype = np.complex128
    ctx = ctx_factory()

    dimensions = 2

    from sumpy.symbolic import make_sym_vector, make_helmholtz_kernel
    avec = make_sym_vector("a", dimensions)
    bvec = make_sym_vector("b", dimensions)
    kernel = make_helmholtz_kernel(avec)

    texp = quack_expand(kernel, 4, avec, bvec)

    from pymbolic.sympy_conv import ToPymbolicMapper
    texp = ToPymbolicMapper()(texp)

    texp = PowerRewriter()(texp)
    texp = FractionKiller()(texp)

    texp = HankelSubstitutor(HankelGetter())(texp)

    from pymbolic.cse import tag_common_subexpressions
    texp = tag_common_subexpressions([texp])

    from pymbolic.mapper.c_code import CCodeMapper
    ccm = CCodeMapper()
    expr = ccm(texp)
    for name, rhs in ccm.cse_name_list:
        print "%s = %s" % (name, rhs)
    print expr
    1/0



    from sumpy.symbolic.codegen import generate_cl_statements_from_assignments
    stmts = generate_cl_statements_from_assignments([("result", texp)])
    for lhs, rhs in stmts:
        print "%s = %s" % (lhs, rhs)
    1/0



    knl = lp.make_kernel(ctx.devices[0],
            "[nsrc,ntgt] -> {[isrc,itgt,]: 0<=itgt<ntgt and 0<=isrc<nsrc}",
           exprs, [
               lp.ArrayArg("density", geo_dtype, shape="nsrc", order="C"),
               lp.ArrayArg("src", geo_dtype, shape=("nsrc", dimensions), order="C"),
               lp.ArrayArg("tgt", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ArrayArg("center", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ArrayArg("pot", value_dtype, shape="ntgt", order="C"),
               lp.ScalarArg("nsrc", np.int32),
               lp.ScalarArg("ntgt", np.int32),
               ],
           name="quack", assumptions="nsrc>=1 and ntgt>=1")

    seq_knl = knl

    def variant_1(knl):
        knl = lp.split_dimension(knl, "i", 256,
                outer_tag="g.0", inner_tag="l.0",
                slabs=(0,1))
        knl = lp.split_dimension(knl, "j", 256, slabs=(0,1))
        return knl, []

    def variant_cpu(knl):
        knl = lp.expand_subst(knl)
        knl = lp.split_dimension(knl, "i", 1024,
                outer_tag="g.0", slabs=(0,1))
        knl = lp.add_prefetch(knl, "x[i,k]", ["k"], default_tag=None)
        return knl, []

    def variant_gpu(knl):
        knl = lp.expand_subst(knl)
        knl = lp.split_dimension(knl, "i", 256,
                outer_tag="g.0", inner_tag="l.0", slabs=(0,1))
        knl = lp.split_dimension(knl, "j", 256, slabs=(0,1))
        knl = lp.add_prefetch(knl, "x[i,k]", ["k"], default_tag=None)
        knl = lp.add_prefetch(knl, "x[j,k]", ["j_inner", "k"],
                ["x_fetch_j", "x_fetch_k"])
        knl = lp.tag_dimensions(knl, dict(x_fetch_k="unr"))
        return knl, ["j_outer", "j_inner"]

    n = 3000

    for variant in [variant_1, variant_cpu, variant_gpu]:
        variant_knl, loop_prio = variant(knl)
        kernel_gen = lp.generate_loop_schedules(variant_knl,
                loop_priority=loop_prio)
        kernel_gen = lp.check_kernels(kernel_gen, dict(N=n))

        lp.auto_test_vs_ref(seq_knl, ctx, kernel_gen,
                op_count=n**2*1e-6, op_label="M particle pairs",
                parameters={"N": n}, print_ref_code=True)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
