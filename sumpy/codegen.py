from __future__ import division

import numpy as np
import sympy as sp
import pyopencl as cl
import pyopencl.tools

from pymbolic.mapper import IdentityMapper
import pymbolic.primitives as prim

from pytools import memoize_method


# {{{ hankel handling

import sumpy.hank103

HANKEL_PREAMBLE = sumpy.hank103.CODE+"""//CL//
typedef struct hank1_01_result_str
{
    cdouble_t order0, order1;
} hank1_01_result;

hank1_01_result hank1_01(cdouble_t z)
{
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




class VectorComponentRewriter(IdentityMapper):
    def __init__(self, vec_names=["a", "b", "d"]):
        self.vec_names = vec_names

    def map_variable(self, expr):
        for vn in self.vec_names:
            if expr.name.startswith(vn):
                try:
                    subscript = int(expr.name[len(vn):])
                except TypeError:
                    continue
                else:
                    return prim.Variable(vn)[subscript]

        return IdentityMapper.map_variable(self, expr)




def sympy_to_pymbolic_for_code(sympy_exprs):
    from pymbolic.sympy_conv import SympyToPymbolicMapper
    exprs = [SympyToPymbolicMapper()(se) for se in sympy_exprs]

    exprs = [VectorComponentRewriter()(expr) for expr in exprs]
    exprs = [PowerRewriter()(expr) for expr in exprs]
    exprs = [FractionKiller()(expr) for expr in exprs]

    exprs = [HankelSubstitutor(HankelGetter())(expr) for expr in exprs]

    from pymbolic.cse import tag_common_subexpressions
    return tag_common_subexpressions(exprs)
