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


import re

import numpy as np
import loopy as lp
from loopy.kernel.instruction import make_assignment

from pymbolic.mapper import IdentityMapper, CSECachingMapperMixin
import pymbolic.primitives as prim

from pytools import memoize_method

from sumpy.symbolic import (SympyToPymbolicMapper as SympyToPymbolicMapperBase)

import logging
logger = logging.getLogger(__name__)


__doc__ = """

Conversion of :mod:`sympy` expressions to :mod:`loopy`
------------------------------------------------------

.. autoclass:: SympyToPymbolicMapper
.. autofunction:: to_loopy_insns

"""


# {{{ sympy -> pymbolic mapper

import sumpy.symbolic as sym
_SPECIAL_FUNCTION_NAMES = frozenset(dir(sym.functions))


class SympyToPymbolicMapper(SympyToPymbolicMapperBase):

    def not_supported(self, expr):
        if isinstance(expr, int):
            return expr
        elif getattr(expr, "is_Function", False):
            func_name = SympyToPymbolicMapperBase.function_name(self, expr)
            return prim.Variable(func_name)(
                    *tuple(self.rec(arg) for arg in expr.args))
        else:
            return SympyToPymbolicMapperBase.not_supported(self, expr)

# }}}


# {{{ bessel -> loopy codegen

BESSEL_PREAMBLE = """//CL//
#include <pyopencl-bessel-j.cl>
#include <pyopencl-bessel-y.cl>
#include <pyopencl-bessel-j-complex.cl>

double bessel_jv_two(int v, double z, double *jvp1)
{
    *jvp1 = bessel_jv(v+1, z);
    return bessel_jv(v, z);
}

cdouble_t bessel_jv_two_complex(int v, cdouble_t z, cdouble_t *jvp1)
{
    cdouble_t jv;
    bessel_j_complex(v, z, &jv, jvp1);
    return jv;
}
"""

HANKEL_PREAMBLE = """//CL//
#include <pyopencl-bessel-j.cl>
#include <pyopencl-bessel-y.cl>
#include <pyopencl-hankel-complex.cl>

cdouble_t hank1_01(double z, cdouble_t *order1)
{
    *order1 = cdouble_new(bessel_j1(z), bessel_y1(z));
    return cdouble_new(bessel_j0(z), bessel_y0(z));
}

cdouble_t hank1_01_complex(cdouble_t z, cdouble_t *order1)
{
    cdouble_t order0;
    hankel_01_complex(z, &order0, order1, 1);
    return order0;
}
"""


class BesselJvvp1(lp.ScalarCallable):
    def with_types(self, arg_id_to_dtype, clbl_inf_ctx):
        from loopy.types import NumpyType

        for i in arg_id_to_dtype:
            if not (-2 <= i <= 1):
                raise TypeError(f"{self.name} can only take 2 arguments.")

        if (arg_id_to_dtype.get(0) is None) or (arg_id_to_dtype.get(1) is None):
            # not enough info about input types
            return self, clbl_inf_ctx

        n_dtype = arg_id_to_dtype[0]
        z_dtype = arg_id_to_dtype[1]

        # *technically* takes a float, but let's not worry about that.
        if n_dtype.numpy_dtype.kind != "i":
            raise TypeError(f"{self.name} expects an integer first argument")

        if z_dtype.numpy_dtype.kind == "c":
            return (self.copy(name_in_target="bessel_jv_two_complex",
                              arg_id_to_dtype={
                                  -2: NumpyType(np.complex128),
                                  -1: NumpyType(np.complex128),
                                  0: NumpyType(np.int32),
                                  1: NumpyType(np.complex128),
                                  }),
                    clbl_inf_ctx)
        else:
            return (self.copy(name_in_target="bessel_jv_two",
                              arg_id_to_dtype={
                                  -2: NumpyType(np.float64),
                                  -1: NumpyType(np.float64),
                                  0: NumpyType(np.int32),
                                  1: NumpyType(np.float64),
                                  }),
                    clbl_inf_ctx)

    def generate_preambles(self, target):
        from loopy import PyOpenCLTarget
        if not isinstance(target, PyOpenCLTarget):
            raise NotImplementedError("Only the PyOpenCLTarget is supported as"
                                      "of now.")

        yield ("40-sumpy-bessel", BESSEL_PREAMBLE)


class Hankel1_01(lp.ScalarCallable):  # noqa: N801
    def with_types(self, arg_id_to_dtype, clbl_inf_ctx):
        from loopy.types import NumpyType

        for i in arg_id_to_dtype:
            if not (-2 <= i <= 0):
                raise TypeError(f"{self.name} can only take one argument.")

        if arg_id_to_dtype.get(0) is None:
            # not enough info about input types
            return self, clbl_inf_ctx

        z_dtype = arg_id_to_dtype[0]

        if z_dtype.numpy_dtype.kind == "c":
            return (self.copy(name_in_target="hank1_01_complex",
                              arg_id_to_dtype={
                                  -2: NumpyType(np.complex128),
                                  -1: NumpyType(np.complex128),
                                  0: NumpyType(np.complex128),
                                  }),
                    clbl_inf_ctx)
        else:
            return (self.copy(name_in_target="hank1_01",
                              arg_id_to_dtype={
                                  -2: NumpyType(np.complex128),
                                  -1: NumpyType(np.complex128),
                                  0: NumpyType(np.float64),
                                  }),
                    clbl_inf_ctx)

    def generate_preambles(self, target):
        from loopy import PyOpenCLTarget
        if not isinstance(target, PyOpenCLTarget):
            raise NotImplementedError("Only the PyOpenCLTarget is supported as"
                                      "of now.")

        yield ("50-sumpy-hankel", HANKEL_PREAMBLE)


def register_bessel_callables(loopy_knl):
    from sumpy.codegen import BesselJvvp1, Hankel1_01
    if "bessel_jvvp1" not in loopy_knl.callables_table:
        loopy_knl = lp.register_callable(loopy_knl, "bessel_jvvp1",
            BesselJvvp1("bessel_jvvp1"))
    if "hank1_01" not in loopy_knl.callables_table:
        loopy_knl = lp.register_callable(loopy_knl, "hank1_01",
            Hankel1_01("hank1_01"))
    return loopy_knl


def _fp_contract_fast_preamble(preamble_info):
    yield ("fp_contract_fast_pocl", "#pragma clang fp contract(fast)")


def register_optimization_preambles(loopy_knl, device):
    if isinstance(loopy_knl.target, lp.PyOpenCLTarget):
        import pyopencl as cl
        if device.platform.name == "Portable Computing Language" and \
                (device.type & cl.device_type.GPU):
            loopy_knl = lp.register_preamble_generators(loopy_knl,
                [_fp_contract_fast_preamble])
    return loopy_knl

# }}}


# {{{ custom mapper base classes

class CSECachingIdentityMapper(IdentityMapper, CSECachingMapperMixin):
    pass


class CallExternalRecMapper(IdentityMapper):
    def rec(self, expr, rec_self=None, *args, **kwargs):
        if rec_self:
            return rec_self.rec(expr, *args, **kwargs)
        else:
            return super().rec(expr, *args, **kwargs)

# }}}


# {{{ bessel handling

class BesselTopOrderGatherer(CSECachingIdentityMapper, CallExternalRecMapper):
    """This mapper walks the expression tree to find the highest-order
    Bessel J being used, so that all other Js can be computed by the
    (stable) downward recurrence.
    """
    def __init__(self):
        self.bessel_j_arg_to_top_order = {}

    def map_call(self, expr, rec_self=None, *args):
        if isinstance(expr.function, prim.Variable) \
                and expr.function.name == "bessel_j":
            order, arg = expr.parameters
            self.rec(arg)
            assert isinstance(order, int)
            self.bessel_j_arg_to_top_order[arg] = max(
                    self.bessel_j_arg_to_top_order.get(arg, 0),
                    abs(order))
        return CSECachingIdentityMapper.map_call(rec_self or self,
                expr, rec_self, *args)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression


class BesselDerivativeReplacer(CSECachingIdentityMapper, CallExternalRecMapper):
    def map_call(self, expr, rec_self=None, *args):
        call = expr

        if (isinstance(call.function, prim.Variable)
                and call.function.name in ["Hankel1", "BesselJ"]):
            if call.function.name == "Hankel1":
                function = prim.Variable("hankel_1")
            else:
                function = prim.Variable("bessel_j")
            order, arg, n_derivs = call.parameters
            import sympy as sym

            # AS (9.1.31)
            # https://dlmf.nist.gov/10.6.7
            if order >= 0:
                order_str = str(order)
            else:
                order_str = "m"+str(-order)
            k = n_derivs
            return prim.CommonSubexpression(
                    2**(-k)*sum(
                        (-1)**idx*int(sym.binomial(k, idx)) * function(i, arg)
                        for idx, i in enumerate(range(order-k, order+k+1, 2))),
                    f"d{n_derivs}_{function.name}_{order_str}",
                    scope=prim.cse_scope.EVALUATION)
        else:
            return CSECachingIdentityMapper.map_call(
                    rec_self or self, expr, rec_self, *args)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression


class BesselSubstitutor(CSECachingIdentityMapper):
    def __init__(self, name_gen, bessel_j_arg_to_top_order, assignments):
        self.name_gen = name_gen
        self.bessel_j_arg_to_top_order = bessel_j_arg_to_top_order
        self.cse_cache = {}
        self.assignments = assignments

    def map_call(self, expr, *args):
        if isinstance(expr.function, prim.Variable):
            name = expr.function.name
            if name == "bessel_j":
                order, arg = expr.parameters
                return self.bessel_j(order, self.rec(arg, *args))
            elif name == "hankel_1":
                order, arg = expr.parameters
                return self.hankel_1(order, self.rec(arg, *args))

        return super().map_call(expr)

    def wrap_in_cse(self, expr, prefix):
        cse = prim.wrap_in_cse(expr, prefix)
        return self.cse_cache.setdefault(expr, cse)

    # {{{ bessel implementation

    @memoize_method
    def bessel_jv_two(self, order, arg):
        name_om1 = self.name_gen(f"bessel_{order - 1}")
        name_o = self.name_gen(f"bessel_{order}")
        self.assignments.append(
                make_assignment(
                    (prim.Variable(name_om1), prim.Variable(name_o),),
                    prim.Variable("bessel_jvvp1")(order, arg),
                    temp_var_types=(lp.Optional(None),)*2))

        return prim.Variable(name_om1), prim.Variable(name_o)

    @memoize_method
    def bessel_j(self, order, arg):
        top_order = self.bessel_j_arg_to_top_order[arg]
        if order == top_order:
            return self.bessel_jv_two(top_order-1, arg)[1]
        elif order == top_order-1:
            return self.bessel_jv_two(top_order-1, arg)[0]
        elif order < 0:
            return self.wrap_in_cse(
                    (-1)**order*self.bessel_j(-order, arg),
                    f"bessel_j_neg{-order}")
        else:
            assert abs(order) < top_order

            # AS (9.1.27)
            nu = order+1
            return self.wrap_in_cse(
                    2*nu/arg*self.bessel_j(nu, arg) - self.bessel_j(nu+1, arg),
                    f"bessel_j_{order}")

    # }}}

    # {{{ hankel implementation

    @memoize_method
    def hank1_01(self, arg):
        name_0 = self.name_gen("hank1_0")
        name_1 = self.name_gen("hank1_1")
        self.assignments.append(
                make_assignment(
                    (prim.Variable(name_0), prim.Variable(name_1),),
                    prim.Variable("hank1_01")(arg),
                    temp_var_types=(lp.Optional(None),)*2))
        return prim.Variable(name_0), prim.Variable(name_1)

    @memoize_method
    def hankel_1(self, order, arg):
        if order == 0:
            return self.hank1_01(arg)[0]
        elif order == 1:
            return self.hank1_01(arg)[1]
        elif order < 0:
            # AS (9.1.6)
            nu = -order
            return self.wrap_in_cse(
                    (-1) ** nu * self.hankel_1(nu, arg),
                    f"hank1_neg{nu}")
        elif order > 1:
            # AS (9.1.27)
            nu = order-1
            return self.wrap_in_cse(
                    2*nu/arg*self.hankel_1(nu, arg) - self.hankel_1(nu-1, arg),
                    f"hank1_{order}")
        else:
            raise AssertionError()

    # }}}

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ power rewriter

class PowerRewriter(CSECachingIdentityMapper, CallExternalRecMapper):
    def map_power(self, expr, rec_self=None, *args):
        exp = expr.exponent
        if isinstance(exp, int):
            new_base = prim.wrap_in_cse(expr.base)

            if exp > 2 and exp % 2 == 0:
                square = prim.wrap_in_cse(new_base*new_base)
                return self.rec(prim.wrap_in_cse(square**(exp//2)),
                        rec_self, *args)
            elif exp == 2:
                return new_base * new_base
            elif exp > 1 and exp % 2 == 1:
                square = prim.wrap_in_cse(new_base*new_base)
                return self.rec(prim.wrap_in_cse(square**((exp-1)//2))*new_base,
                        rec_self, *args)
            elif exp == 1:
                return new_base
            elif exp < 0:
                return self.rec((1/new_base)**(-exp), rec_self, *args)

        if (isinstance(expr.exponent, prim.Quotient)
                and isinstance(expr.exponent.numerator, int)
                and isinstance(expr.exponent.denominator, int)):

            p, q = expr.exponent.numerator, expr.exponent.denominator
            if q < 0:
                q *= -1
                p *= -1

            if q == 1:
                return self.rec(new_base**p, rec_self, *args)

            if q == 2:
                assert p != 0

                if p > 0:
                    orig_base = prim.wrap_in_cse(expr.base)
                    new_base = prim.wrap_in_cse(prim.Variable("sqrt")(orig_base))
                else:
                    new_base = prim.wrap_in_cse(prim.Variable("rsqrt")(expr.base))
                    p *= -1

                return self.rec(new_base**p, rec_self, *args)

        return CSECachingIdentityMapper.map_power(rec_self or self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ convert big integers into floats

from loopy.tools import is_integer


class BigIntegerKiller(CSECachingIdentityMapper, CallExternalRecMapper):

    def __init__(self, warn_on_digit_loss=True, int_type=np.int64,
            float_type=np.float64):
        super().__init__()
        self.warn = warn_on_digit_loss
        self.float_type = float_type
        self.iinfo = np.iinfo(int_type)

    def map_constant(self, expr, *args):
        """Convert integer values not within the range of `self.int_type` to float.
        """
        if not is_integer(expr):
            return expr

        if self.iinfo.min <= expr <= self.iinfo.max:
            return expr

        if self.warn:
            expr_as_float = self.float_type(expr)
            if int(expr_as_float) != int(expr):
                from warnings import warn
                warn(f"Converting '{expr}' to "
                     f"'{self.float_type.__name__}' loses digits", stacklevel=1)

            # Suppress further warnings.
            self.warn = False
            return expr_as_float

        return self.float_type(expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ convert complex to np.complex

class ComplexRewriter(CSECachingIdentityMapper, CallExternalRecMapper):

    def __init__(self, complex_dtype=None):
        super().__init__()
        self.complex_dtype = complex_dtype

    def map_constant(self, expr, rec_self=None):
        """Convert complex values to numpy types
        """
        if not isinstance(expr, (complex, np.complex64, np.complex128)):
            return IdentityMapper.map_constant(rec_self or self, expr,
                    rec_self=rec_self)

        complex_dtype = self.complex_dtype
        if complex_dtype is None:
            if complex(np.complex64(expr)) == expr:
                return np.complex64(expr)
            complex_dtype = np.complex128

        if isinstance(complex_dtype, np.dtype):
            return complex_dtype.type(expr)
        else:
            return complex_dtype(expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ vector component rewriter

INDEXED_VAR_RE = re.compile("^([a-zA-Z_]+)([0-9]+)$")


class VectorComponentRewriter(CSECachingIdentityMapper, CallExternalRecMapper):
    """For names in name_whitelist, turn ``a3`` into ``a[3]``."""

    def __init__(self, name_whitelist=frozenset()):
        self.name_whitelist = name_whitelist

    def map_variable(self, expr, *args):
        match_obj = INDEXED_VAR_RE.match(expr.name)
        if match_obj is not None:
            name = match_obj.group(1)
            subscript = int(match_obj.group(2))
            if name in self.name_whitelist:
                return prim.Variable(name).index(subscript)
            else:
                return expr
        else:
            return expr

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ sum sign grouper

class SumSignGrouper(CSECachingIdentityMapper, CallExternalRecMapper):
    """Anti-cancellation cargo-cultism."""

    def map_sum(self, expr, *args):
        first_group = []
        second_group = []

        for orig_child in expr.children:
            child = self.rec(orig_child, *args)
            tchild = child
            if isinstance(tchild, prim.CommonSubexpression):
                tchild = tchild.child

            if isinstance(tchild, prim.Product):
                neg_int_count = 0
                for subchild in tchild.children:
                    if isinstance(subchild, int) and subchild < 0:
                        neg_int_count += 1

                if neg_int_count % 2 == 1:
                    second_group.append(child)
                else:
                    first_group.append(child)
            else:
                first_group.append(child)

        new_children = tuple(first_group + second_group)
        if len(new_children) == len(expr.children) and \
                all(child is orig_child for child, orig_child in
                    zip(new_children, expr.children)):
            return expr
        return prim.Sum(tuple(first_group+second_group))

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


class MathConstantRewriter(CSECachingIdentityMapper, CallExternalRecMapper):
    def map_variable(self, expr, *args):
        if expr.name == "pi":
            return prim.Variable("M_PI")
        else:
            return expr

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression


# {{{ combine mappers

def combine_mappers(*mappers):
    """Returns a mapper that combines the work of several other mappers.  For
    this to work, the mappers need to be instances of
    :class:`sumpy.codegen.CallExternalRecMapper`.  When calling parent class
    methods, the mappers need to use the (first) argument *rec_self* as the
    instance passed to the *map_* method. *rec_self* is a (custom-generated)
    *CombinedMapper* instance which dispatches the object to all the mappers
    given. The mappers need to commute and be idempotent.
    """
    from collections import defaultdict
    all_methods = defaultdict(list)
    base_classes = [CSECachingMapperMixin, IdentityMapper]
    for mapper in mappers:
        assert isinstance(mapper, CallExternalRecMapper)
        for method_name in dir(type(mapper)):
            if not method_name.startswith("map_"):
                continue
            if method_name == "map_common_subexpression_uncached":
                continue
            method = getattr(type(mapper), method_name)
            method_equals_base_class_method = False
            for base_class in base_classes:
                base_class_method = getattr(base_class, method_name, None)
                if base_class_method is not None:
                    method_equals_base_class_method = (base_class_method == method)
                    break
            else:
                raise RuntimeError(f"Unknown mapping method {method_name}")

            if method_equals_base_class_method:
                continue
            all_methods[method_name].append((mapper, method))

    class CombinedMapper(CSECachingIdentityMapper):
        def __init__(self, all_methods):
            self.all_methods = all_methods
        map_common_subexpression_uncached = \
                IdentityMapper.map_common_subexpression

    def _map(method_name, self, expr, rec_self=None, *args):
        if method_name not in self.all_methods:
            return getattr(IdentityMapper, method_name)(self, expr)
        for mapper, method in self.all_methods[method_name]:
            new_expr = method(mapper, expr, self)
            if new_expr is not expr:
                # Re-traverse the whole thing from the get-go.
                return self.rec(new_expr)
        return expr

    from functools import partial
    import types
    combine_mapper = CombinedMapper(all_methods)
    for method_name in all_methods.keys():
        setattr(combine_mapper, method_name,
                types.MethodType(partial(_map, method_name), combine_mapper))
    return combine_mapper

# }}}


# {{{ to-loopy conversion

def to_loopy_insns(assignments, vector_names=frozenset(), pymbolic_expr_maps=(),
                   complex_dtype=None, retain_names=frozenset()):
    logger.info("loopy instruction generation: start")
    assignments = list(assignments)

    # convert from sympy
    sympy_conv = SympyToPymbolicMapper()
    assignments = [(name, sympy_conv(expr)) for name, expr in assignments]

    bdr = BesselDerivativeReplacer()
    btog = BesselTopOrderGatherer()
    vcr = VectorComponentRewriter(vector_names)
    pwr = PowerRewriter()
    ssg = SumSignGrouper()
    bik = BigIntegerKiller()
    cmr = ComplexRewriter(complex_dtype)

    if 0:
        # https://github.com/inducer/sumpy/pull/40#issuecomment-852635444
        cmb_mapper = combine_mappers(bdr, btog, vcr, pwr, ssg, bik, cmr)
    else:
        def cmb_mapper(expr):
            expr = bdr(expr)
            expr = vcr(expr)
            expr = pwr(expr)
            expr = ssg(expr)
            expr = bik(expr)
            expr = cmr(expr)
            expr = btog(expr)
            return expr

    def convert_expr(name, expr):
        logger.debug("generate expression for: %s", name)
        expr = cmb_mapper(expr)
        for m in pymbolic_expr_maps:
            expr = m(expr)
        return expr

    assignments = [(name, convert_expr(name, expr)) for name, expr in assignments]
    from pytools import UniqueNameGenerator
    name_gen = UniqueNameGenerator({name for name, expr in assignments})

    result = []
    bessel_sub = BesselSubstitutor(
            name_gen, btog.bessel_j_arg_to_top_order,
            result)

    import loopy as lp
    from pytools import MinRecursionLimit
    with MinRecursionLimit(3000):
        for name, expr in assignments:
            result.append(lp.Assignment(id=None,
                    assignee=name, expression=bessel_sub(expr),
                    temp_var_type=lp.Optional(None)))

    logger.info("loopy instruction generation: done")
    return result

# }}}

# vim: fdm=marker
