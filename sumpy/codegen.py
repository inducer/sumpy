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
import pyopencl as cl
import pyopencl.tools  # noqa
import loopy as lp

import re

from pymbolic.mapper import IdentityMapper, WalkMapper, CSECachingMapperMixin
import pymbolic.primitives as prim

from loopy.types import NumpyType

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


# {{{ trivial assignment elimination

def make_one_step_subst(assignments):
    assignments = dict(assignments)
    unwanted_vars = set(assignments.keys())

    # Ensure no re-assignments.
    assert len(unwanted_vars) == len(assignments)

    from loopy.symbolic import get_dependencies
    unwanted_deps = {
        name: get_dependencies(value) & unwanted_vars
        for name, value in assignments.items()}

    # {{{ compute substitution order

    toposort = []
    visited = set()
    visiting = set()

    while unwanted_vars:
        stack = [unwanted_vars.pop()]

        while stack:
            top = stack[-1]

            if top in visiting:
                visiting.remove(top)
                toposort.append(top)

            if top in visited:
                stack.pop()
                continue

            visited.add(top)
            visiting.add(top)

            for dep in unwanted_deps[top]:
                # Check for no cycles.
                assert dep not in visiting
                stack.append(dep)

    # }}}

    # {{{ make substitution

    from pymbolic import substitute

    result = {}
    used_name_to_var = {}
    from pymbolic import evaluate
    from functools import partial
    simplify = partial(evaluate, context=used_name_to_var)

    for name in toposort:
        value = assignments[name]
        value = substitute(value, result)
        used_name_to_var.update(
            (used_name, prim.Variable(used_name))
            for used_name in get_dependencies(value)
            if used_name not in used_name_to_var)

        result[name] = simplify(value)

    # }}}

    return result


def is_assignment_nontrivial(name, value):
    if prim.is_constant(value):
        return False
    elif isinstance(value, prim.Variable):
        return False
    elif (isinstance(value, prim.Product)
            and len(value.children) == 2
            and sum(1 for arg in value.children if prim.is_constant(arg)) == 1
            and sum(1 for arg in value.children
                    if isinstance(arg, prim.Variable)) == 1):
        # const*var: not good enough
        return False

    return True


def kill_trivial_assignments(assignments, retain_names=set()):
    logger.info("kill trivial assignments (plain): start")
    from time import time
    start = time()
    approved_assignments = []
    rejected_assignments = []

    for name, value in assignments:
        if name in retain_names or is_assignment_nontrivial(name, value):
            approved_assignments.append((name, value))
        else:
            rejected_assignments.append((name, value))

    # un-substitute rejected assignments
    unsubst_rej = make_one_step_subst(rejected_assignments)

    result = []
    from pymbolic import substitute
    for name, expr in approved_assignments:
        r = substitute(expr, unsubst_rej)
        result.append((name, r))

    nrej = len(rejected_assignments)
    total_time = time() - start
    logger.info(f"kill trivial assignments (plain): done, "
                f"{nrej} assignments killed in {total_time:.3g} seconds")

    return result

# }}}


# {{{ bessel handling

BESSEL_PREAMBLE = """//CL//
#include <pyopencl-bessel-j.cl>
#include <pyopencl-bessel-y.cl>
#include <pyopencl-bessel-j-complex.cl>

typedef struct bessel_j_two_result_str
{
    cdouble_t jv, jvp1;
} bessel_j_two_result;

bessel_j_two_result bessel_jv_two(int v, double z)
{
    bessel_j_two_result result;
    result.jv = cdouble_fromreal(bessel_jv(v, z));
    result.jvp1 = cdouble_fromreal(bessel_jv(v+1, z));
    return result;
}

bessel_j_two_result bessel_jv_two_complex(int v, cdouble_t z)
{
    bessel_j_two_result result;
    bessel_j_complex(v, z, &result.jv, &result.jvp1);
    return result;
}
"""

HANKEL_PREAMBLE = """//CL//
#include <pyopencl-hankel-complex.cl>

typedef struct hank1_01_result_str
{
    cdouble_t order0, order1;
} hank1_01_result;

hank1_01_result hank1_01(double z)
{
    hank1_01_result result;
    result.order0 = cdouble_new(bessel_j0(z), bessel_y0(z));
    result.order1 = cdouble_new(bessel_j1(z), bessel_y1(z));
    return result;
}

hank1_01_result hank1_01_complex(cdouble_t z)
{
    hank1_01_result result;
    hankel_01_complex(z, &result.order0, &result.order1, 1);
    return result;
}
"""


def bessel_preamble_generator(preamble_info):
    from loopy.target.pyopencl import PyOpenCLTarget
    if not isinstance(preamble_info.kernel.target, PyOpenCLTarget):
        raise NotImplementedError("Only the PyOpenCLTarget is supported as of now")

    require_bessel = False
    if any(func.name == "hank1_01" for func in preamble_info.seen_functions):
        yield ("50-sumpy-hankel", HANKEL_PREAMBLE)
        require_bessel = True
    if (require_bessel
            or any(func.name == "bessel_jv_two"
                for func in preamble_info.seen_functions)):
        yield ("40-sumpy-bessel", BESSEL_PREAMBLE)


hank1_01_result_dtype = cl.tools.get_or_register_dtype("hank1_01_result",
        NumpyType(np.dtype([
            ("order0", np.complex128),
            ("order1", np.complex128),
            ])),
        )

bessel_j_two_result_dtype = cl.tools.get_or_register_dtype("bessel_j_two_result",
        NumpyType(np.dtype([
            ("jv", np.complex128),
            ("jvp1", np.complex128),
            ])),
        )


def bessel_mangler(kernel, identifier, arg_dtypes):
    """A function "mangler" to make Bessel functions
    digestible for :mod:`loopy`.

    See argument *function_manglers* to :func:`loopy.make_kernel`.
    """

    from loopy.target.pyopencl import PyOpenCLTarget
    if not isinstance(kernel.target, PyOpenCLTarget):
        raise NotImplementedError("Only the PyOpenCLTarget is supported as of now")

    if identifier == "hank1_01":
        if arg_dtypes[0].is_complex():
            identifier = "hank1_01_complex"
            return lp.CallMangleInfo(
                    target_name=identifier,
                    result_dtypes=(NumpyType(np.dtype(hank1_01_result_dtype)),),
                    arg_dtypes=(
                        NumpyType(np.dtype(np.complex128)),
                        ))
        else:
            return lp.CallMangleInfo(
                    target_name=identifier,
                    result_dtypes=(NumpyType(np.dtype(hank1_01_result_dtype)),),
                    arg_dtypes=(
                        NumpyType(np.dtype(np.float64)),
                        ))

    elif identifier == "bessel_jv_two":
        if arg_dtypes[1].is_complex():
            identifier = "bessel_jv_two_complex"
            return lp.CallMangleInfo(
                    target_name=identifier,
                    result_dtypes=(NumpyType(np.dtype(bessel_j_two_result_dtype)),),
                    arg_dtypes=(
                        NumpyType(np.dtype(np.int32)),
                        NumpyType(np.dtype(np.complex128)),))
        else:
            return lp.CallMangleInfo(
                    target_name=identifier,
                    result_dtypes=(NumpyType(np.dtype(bessel_j_two_result_dtype)),),
                    arg_dtypes=(
                        NumpyType(np.dtype(np.int32)),
                        NumpyType(np.dtype(np.float64)),))

    else:
        return None


class BesselGetter:
    def __init__(self, bessel_j_arg_to_top_order):
        self.bessel_j_arg_to_top_order = bessel_j_arg_to_top_order
        self.cse_cache = {}

    @memoize_method
    def hank1_01(self, arg):
        return prim.Variable("hank1_01")(arg)

    @memoize_method
    def bessel_jv_two(self, order, arg):
        return prim.Variable("bessel_jv_two")(order, arg)

    def wrap_in_cse(self, expr, prefix):
        cse = prim.wrap_in_cse(expr, prefix)
        return self.cse_cache.setdefault(expr, cse)

    @memoize_method
    def hankel_1(self, order, arg):
        if order == 0:
            return self.wrap_in_cse(
                    prim.Lookup(self.hank1_01(arg), "order0"),
                    "hank1_01_result")
        elif order == 1:
            return self.wrap_in_cse(
                    prim.Lookup(self.hank1_01(arg), "order1"),
                    "hank1_01_result")
        elif order < 0:
            # AS (9.1.6)
            nu = -order
            return self.wrap_in_cse(
                    (-1) ** nu * self.hankel_1(nu, arg),
                    "hank1_neg%d" % nu)
        elif order > 1:
            # AS (9.1.27)
            nu = order-1
            return self.wrap_in_cse(
                    2*nu/arg*self.hankel_1(nu, arg)
                    - self.hankel_1(nu-1, arg),
                    "hank1_%d" % order)
        else:
            assert False

    @memoize_method
    def bessel_j(self, order, arg):
        top_order = self.bessel_j_arg_to_top_order[arg]

        if order == top_order:
            return self.wrap_in_cse(
                    prim.Lookup(self.bessel_jv_two(top_order-1, arg), "jvp1"),
                    "bessel_jv_two_result")
        elif order == top_order-1:
            return self.wrap_in_cse(
                    prim.Lookup(self.bessel_jv_two(top_order-1, arg), "jv"),
                    "bessel_jv_two_result")
        elif order < 0:
            return self.wrap_in_cse(
                    (-1)**order*self.bessel_j(-order, arg),
                    "bessel_j_neg%d" % -order)
        else:
            assert abs(order) < top_order

            # AS (9.1.27)
            nu = order+1
            return self.wrap_in_cse(
                    2*nu/arg*self.bessel_j(nu, arg)
                    - self.bessel_j(nu+1, arg),
                    "bessel_j_%d" % order)


class BesselTopOrderGatherer(CSECachingMapperMixin, WalkMapper):
    """This mapper walks the expression tree to find the highest-order
    Bessel J being used, so that all other Js can be computed by the
    (stable) downward recurrence.
    """
    def __init__(self):
        self.bessel_j_arg_to_top_order = {}

    def map_call(self, expr):
        if isinstance(expr.function, prim.Variable) \
                and expr.function.name == "bessel_j":
            order, arg = expr.parameters
            self.rec(arg)
            assert isinstance(order, int)
            self.bessel_j_arg_to_top_order[arg] = max(
                    self.bessel_j_arg_to_top_order.get(arg, 0),
                    abs(order))
        else:
            return WalkMapper.map_call(self, expr)

    map_common_subexpression_uncached = WalkMapper.map_common_subexpression


class BesselDerivativeReplacer(CSECachingMapperMixin, IdentityMapper):
    def map_substitution(self, expr):
        assert isinstance(expr.child, prim.Derivative)
        call = expr.child.child

        if (isinstance(call.function, prim.Variable)
                and call.function.name in ["hankel_1", "bessel_j"]):
            function = call.function
            order, _ = call.parameters
            arg, = expr.values

            n_derivs = len(expr.child.variables)
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
                    "d%d_%s_%s" % (n_derivs, function.name, order_str))
        else:
            return IdentityMapper.map_substitution(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression


class BesselSubstitutor(CSECachingMapperMixin, IdentityMapper):
    def __init__(self, bessel_getter):
        self.bessel_getter = bessel_getter

    def map_call(self, expr):
        if isinstance(expr.function, prim.Variable):
            name = expr.function.name
            if name in ["hankel_1", "bessel_j"]:
                order, arg = expr.parameters
                return getattr(self.bessel_getter, name)(order, self.rec(arg))

        return IdentityMapper.map_call(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ power rewriter

class PowerRewriter(CSECachingMapperMixin, IdentityMapper):
    def map_power(self, expr):
        exp = expr.exponent
        if isinstance(exp, int):
            new_base = prim.wrap_in_cse(expr.base)

            if exp > 1 and exp % 2 == 0:
                square = prim.wrap_in_cse(new_base*new_base)
                return self.rec(prim.wrap_in_cse(square**(exp//2)))
            elif exp > 1 and exp % 2 == 1:
                square = prim.wrap_in_cse(new_base*new_base)
                return self.rec(prim.wrap_in_cse(square**((exp-1)//2))*new_base)
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
                    new_base = prim.wrap_in_cse(prim.Variable("sqrt")(orig_base))
                else:
                    new_base = prim.wrap_in_cse(prim.Variable("rsqrt")(expr.base))
                    p *= -1

                return self.rec(new_base**p)

        return IdentityMapper.map_power(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ fraction killer

class FractionKiller(CSECachingMapperMixin, IdentityMapper):
    """Kills fractions where the numerator is evenly divisible by the
    denominator.

    (Why does :mod:`sympy` even produce these?)
    """
    def map_quotient(self, expr):
        num = expr.numerator
        denom = expr.denominator

        if isinstance(num, int) and isinstance(denom, int):
            if num % denom == 0:
                return num // denom
            return int(expr.numerator) / int(expr.denominator)

        return IdentityMapper.map_quotient(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ convert big integers into floats

from loopy.tools import is_integer


class BigIntegerKiller(CSECachingMapperMixin, IdentityMapper):

    def __init__(self, warn_on_digit_loss=True, int_type=np.int64,
            float_type=np.float64):
        IdentityMapper.__init__(self)
        self.warn = warn_on_digit_loss
        self.float_type = float_type
        self.iinfo = np.iinfo(int_type)

    def map_constant(self, expr):
        """Convert integer values not within the range of `self.int_type` to float.
        """
        if not is_integer(expr):
            return IdentityMapper.map_constant(self, expr)

        if self.iinfo.min <= expr <= self.iinfo.max:
            return expr

        if self.warn:
            expr_as_float = self.float_type(expr)
            if int(expr_as_float) != int(expr):
                from warnings import warn
                warn("Converting '%d' to '%s' loses digits"
                        % (expr, self.float_type.__name__))

            # Suppress further warnings.
            self.warn = False
            return expr_as_float

        return self.float_type(expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ convert 123000000j to 123000000 * 1j

class ComplexRewriter(CSECachingMapperMixin, IdentityMapper):

    def __init__(self, float_type=np.float32):
        IdentityMapper.__init__(self)
        self.float_type = float_type

    def map_constant(self, expr):
        """Convert complex values not within complex64 to a product for loopy
        """
        if not isinstance(expr, complex):
            return IdentityMapper.map_constant(self, expr)

        if complex(self.float_type(expr.imag)) == expr.imag:
            return IdentityMapper.map_constant(self, expr)

        return expr.real + prim.Product((expr.imag, 1j))

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ vector component rewriter

INDEXED_VAR_RE = re.compile("^([a-zA-Z_]+)([0-9]+)$")


class VectorComponentRewriter(CSECachingMapperMixin, IdentityMapper):
    """For names in name_whitelist, turn ``a3`` into ``a[3]``."""

    def __init__(self, name_whitelist=set()):
        self.name_whitelist = name_whitelist

    def map_variable(self, expr):
        match_obj = INDEXED_VAR_RE.match(expr.name)
        if match_obj is not None:
            name = match_obj.group(1)
            subscript = int(match_obj.group(2))
            if name in self.name_whitelist:
                return prim.Variable(name).index(subscript)
            else:
                return IdentityMapper.map_variable(self, expr)
        else:
            return IdentityMapper.map_variable(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


# {{{ sum sign grouper

class SumSignGrouper(CSECachingMapperMixin, IdentityMapper):
    """Anti-cancellation cargo-cultism."""

    def map_sum(self, expr):
        first_group = []
        second_group = []

        for child in expr.children:
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

        return prim.Sum(tuple(first_group+second_group))

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

# }}}


class MathConstantRewriter(CSECachingMapperMixin, IdentityMapper):
    def map_variable(self, expr):
        if expr.name == "pi":
            return prim.Variable("M_PI")
        else:
            return IdentityMapper.map_variable(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression


def to_loopy_insns(assignments, vector_names=set(), pymbolic_expr_maps=[],
                   complex_dtype=None, retain_names=set()):
    logger.info("loopy instruction generation: start")
    assignments = list(assignments)

    # convert from sympy
    sympy_conv = SympyToPymbolicMapper()
    assignments = [(name, sympy_conv(expr)) for name, expr in assignments]

    assignments = kill_trivial_assignments(assignments, retain_names)

    bdr = BesselDerivativeReplacer()
    assignments = [(name, bdr(expr)) for name, expr in assignments]

    btog = BesselTopOrderGatherer()
    for name, expr in assignments:
        btog(expr)

    #from pymbolic.mapper.cse_tagger import CSEWalkMapper, CSETagMapper
    #cse_walk = CSEWalkMapper()
    #for name, expr in assignments:
    #    cse_walk(expr)
    #cse_tag = CSETagMapper(cse_walk)

    # do the rest of the conversion
    bessel_sub = BesselSubstitutor(BesselGetter(btog.bessel_j_arg_to_top_order))
    vcr = VectorComponentRewriter(vector_names)
    pwr = PowerRewriter()
    ssg = SumSignGrouper()
    fck = FractionKiller()
    bik = BigIntegerKiller()
    cmr = ComplexRewriter()

    def convert_expr(name, expr):
        logger.debug("generate expression for: %s" % name)
        expr = bdr(expr)
        expr = bessel_sub(expr)
        expr = vcr(expr)
        expr = pwr(expr)
        expr = fck(expr)
        expr = ssg(expr)
        expr = bik(expr)
        expr = cmr(expr)
        #expr = cse_tag(expr)
        for m in pymbolic_expr_maps:
            expr = m(expr)
        return expr

    import loopy as lp
    from pytools import MinRecursionLimit
    with MinRecursionLimit(3000):
        result = [
                lp.Assignment(id=None,
                    assignee=name, expression=convert_expr(name, expr),
                    temp_var_type=lp.Optional(None))
                for name, expr in assignments]

    logger.info("loopy instruction generation: done")
    return result

# vim: fdm=marker
