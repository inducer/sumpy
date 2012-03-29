from __future__ import division

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.tools
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

HANKEL_PREAMBLE = """//CL//

#include "hank103.cl"

typedef struct hank1_01_result_str
{
    cdouble_t order0, order1;
} hank1_01_result;

hank1_01_result hank1_01(cdouble_t z)
{
    int ifexpon ;
    hank1_01_result result;
    hank103(z, &result.order0, &result.order1, /*ifexpon*/ 1);
    return result;
}
"""

hank1_01_result_dtype = np.dtype([
    ("order0", np.complex128),
    ("order1", np.complex128),
    ])
cl.tools.register_dtype(hank1_01_result_dtype,
        "hank1_01_result")



class HankelGetter(object):
    @memoize_method
    def hank1_01(self, arg):
        from loopy.symbolic import TypedCSE
        return TypedCSE(
                prim.Variable("hank1_01")(arg),
                dtype=hank1_01_result_dtype)

    @memoize_method
    def hank1(self, order, arg):
        from loopy.symbolic import TypedCSE
        if order == 0:
            return TypedCSE(
                    prim.Lookup(self.hank1_01(arg), "order0"),
                    dtype=np.complex128)
        elif order == 1:
            return TypedCSE(
                    prim.Lookup(self.hank1_01(arg), "order1"),
                    dtype=np.complex128)
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

    from sumpy.symbolic import (
            make_sym_vector,
            make_laplace_kernel,
            make_helmholtz_kernel)

    avec = make_sym_vector("a", dimensions)
    bvec = make_sym_vector("b", dimensions)
    kernel = make_helmholtz_kernel(avec)

    texp = quack_expand(kernel, 6, avec, bvec)

    from pymbolic.sympy_conv import ToPymbolicMapper
    texp = ToPymbolicMapper()(texp)

    texp = PowerRewriter()(texp)
    texp = FractionKiller()(texp)

    texp = HankelSubstitutor(HankelGetter())(texp)

    from pymbolic.cse import tag_common_subexpressions
    texp, = tag_common_subexpressions([texp])

    from pymbolic import parse
    knl = lp.make_kernel(ctx.devices[0],
            "[nsrc,ntgt] -> {[isrc,itgt]: 0<=itgt<ntgt and 0<=isrc<nsrc}",
            [
            "<float64> a0 = center[itgt,0] - src[isrc,0]",
            "<float64> a1 = center[itgt,1] - src[isrc,1]",
            "<float64> b0 = tgt[itgt,0] - center[itgt,0]",
            "<float64> b1 = tgt[itgt,1] - center[itgt,1]",
            lp.Instruction(id=None,
                assignee=parse("pairpot"), expression=texp,
                temp_var_type=np.complex128),
            "pot[itgt] = sum_complex128(isrc, pairpot)"
            ], [
               lp.ArrayArg("density", geo_dtype, shape="nsrc", order="C"),
               lp.ArrayArg("src", geo_dtype, shape=("nsrc", dimensions), order="C"),
               lp.ArrayArg("tgt", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ArrayArg("center", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ArrayArg("pot", value_dtype, shape="ntgt", order="C"),
               lp.ScalarArg("nsrc", np.int32),
               lp.ScalarArg("ntgt", np.int32),
               lp.ScalarArg("k", np.complex128),
               ],
           name="quack", assumptions="nsrc>=1 and ntgt>=1",
           preamble=["""
           #define PYOPENCL_DEFINE_CDOUBLE
           #include "pyopencl-complex.h"
           //#include "hank103.h"
           """, HANKEL_PREAMBLE])

    ref_knl = knl

    knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")

    nsrc = 3000
    ntgt = 3000

    par_values = dict(nsrc=nsrc, ntgt=ntgt, k=1)

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, par_values)

    lp.auto_test_vs_ref(ref_knl, ctx, kernel_gen,
            op_count=[nsrc*ntgt], op_label=["point pairs"],
            parameters=par_values, print_ref_code=True,
            codegen_kwargs=dict(allow_complex=True))




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
