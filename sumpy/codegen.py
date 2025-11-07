from __future__ import annotations


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

import logging
import re
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from constantdict import constantdict
from typing_extensions import override

import loopy as lp
import pymbolic.primitives as prim
from loopy.kernel.instruction import Assignment, CallInstruction, make_assignment
from pymbolic.mapper import CSECachingMapperMixin, IdentityMapper, P
from pymbolic.typing import ArithmeticExpression, Expression
from pytools import memoize_method

import sumpy.symbolic as sym


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set

    from numpy.typing import DTypeLike

    import pyopencl as cl
    from loopy.codegen import PreambleInfo
    from loopy.target import TargetBase
    from loopy.translation_unit import CallablesInferenceContext
    from loopy.types import LoopyType


logger = logging.getLogger(__name__)


__doc__ = """

Conversion of :mod:`sympy` expressions to :mod:`loopy`
------------------------------------------------------

.. autoclass:: SympyToPymbolicMapper
.. autofunction:: to_loopy_insns
"""


def wrap_in_cse(expr: Expression,
                prefix: str | None = None) -> prim.CommonSubexpression:
    return prim.make_common_subexpression(expr, prefix, wrap_vars=False)


# {{{ sympy -> pymbolic mapper

_SPECIAL_FUNCTION_NAMES = frozenset(dir(sym.functions))


class SympyToPymbolicMapper(sym.SympyToPymbolicMapper):
    @override
    def not_supported(self, expr: object) -> Expression:
        if isinstance(expr, int):
            return expr
        elif getattr(expr, "is_Function", False):
            func_name = sym.SympyToPymbolicMapper.function_name(self, expr)
            return prim.Variable(func_name)(
                    *tuple(self.rec(arg) for arg in expr.args))
        else:
            return sym.SympyToPymbolicMapper.not_supported(self, expr)

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
    @override
    def with_types(self,
                   arg_id_to_dtype: Mapping[int | str, LoopyType],
                   clbl_inf_ctx: CallablesInferenceContext,
               ) -> tuple[BesselJvvp1, CallablesInferenceContext]:
        from loopy.types import NumpyType

        for i in arg_id_to_dtype:
            if isinstance(i, str):
                raise TypeError(f"{self.name} cannot handle keyword arguments")

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
                              arg_id_to_dtype=constantdict({
                                  -2: NumpyType(np.complex128),
                                  -1: NumpyType(np.complex128),
                                  0: NumpyType(np.int32),
                                  1: NumpyType(np.complex128),
                                  })),
                    clbl_inf_ctx)
        else:
            return (self.copy(name_in_target="bessel_jv_two",
                              arg_id_to_dtype=constantdict({
                                  -2: NumpyType(np.float64),
                                  -1: NumpyType(np.float64),
                                  0: NumpyType(np.int32),
                                  1: NumpyType(np.float64),
                                  })),
                    clbl_inf_ctx)

    @override
    def generate_preambles(self, target: TargetBase) -> Iterator[tuple[str, str]]:
        if not isinstance(target, lp.PyOpenCLTarget):
            raise NotImplementedError("Only the PyOpenCLTarget is supported as"
                                      "of now.")

        yield ("40-sumpy-bessel", BESSEL_PREAMBLE)


class Hankel1_01(lp.ScalarCallable):  # noqa: N801
    @override
    def with_types(self,
                   arg_id_to_dtype: Mapping[int | str, LoopyType],
                   clbl_inf_ctx: CallablesInferenceContext,
               ) -> tuple[Hankel1_01, CallablesInferenceContext]:
        from loopy.types import NumpyType

        for i in arg_id_to_dtype:
            if isinstance(i, str):
                raise TypeError(f"{self.name} cannot handle keyword arguments")

            if not (-2 <= i <= 0):
                raise TypeError(f"{self.name} can only take one argument.")

        if arg_id_to_dtype.get(0) is None:
            # not enough info about input types
            return self, clbl_inf_ctx

        z_dtype = arg_id_to_dtype[0]

        if z_dtype.numpy_dtype.kind == "c":
            return (self.copy(name_in_target="hank1_01_complex",
                              arg_id_to_dtype=constantdict({
                                  -2: NumpyType(np.complex128),
                                  -1: NumpyType(np.complex128),
                                  0: NumpyType(np.complex128),
                                  })),
                    clbl_inf_ctx)
        else:
            return (self.copy(name_in_target="hank1_01",
                              arg_id_to_dtype=constantdict({
                                  -2: NumpyType(np.complex128),
                                  -1: NumpyType(np.complex128),
                                  0: NumpyType(np.float64),
                                  })),
                    clbl_inf_ctx)

    @override
    def generate_preambles(self, target: TargetBase) -> Iterator[tuple[str, str]]:
        if not isinstance(target, lp.PyOpenCLTarget):
            raise NotImplementedError("Only the PyOpenCLTarget is supported as"
                                      "of now.")

        yield ("50-sumpy-hankel", HANKEL_PREAMBLE)


def register_bessel_callables(loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
    if "bessel_jvvp1" not in loopy_knl.callables_table:
        loopy_knl = lp.register_callable(
            loopy_knl,
            "bessel_jvvp1",
            BesselJvvp1("bessel_jvvp1"))

    if "hank1_01" not in loopy_knl.callables_table:
        loopy_knl = lp.register_callable(
            loopy_knl,
            "hank1_01",
            Hankel1_01("hank1_01"))

    return loopy_knl


def _fp_contract_fast_preamble(
        preamble_info: PreambleInfo
    ) -> Iterator[tuple[str, str]]:
    yield ("fp_contract_fast_pocl", "#pragma clang fp contract(fast)")


def register_optimization_preambles(
        loopy_knl: lp.TranslationUnit, device: cl.Device
    ) -> lp.TranslationUnit:
    if isinstance(loopy_knl.target, lp.PyOpenCLTarget):
        import pyopencl as cl
        if (device.platform.name == "Portable Computing Language"
                and (device.type & cl.device_type.GPU)):
            loopy_knl = lp.register_preamble_generators(
                loopy_knl,
                [_fp_contract_fast_preamble])

    return loopy_knl

# }}}


# {{{ custom mapper base classes

class CSECachingIdentityMapper(IdentityMapper[P],
                               CSECachingMapperMixin[Expression, P],
                               ABC):
    pass


# }}}


# {{{ bessel handling

class BesselTopOrderGatherer(CSECachingIdentityMapper[P]):
    """This mapper walks the expression tree to find the highest-order
    Bessel J being used, so that all other Js can be computed by the
    (stable) downward recurrence.
    """

    bessel_j_arg_to_top_order: dict[Expression, int]

    def __init__(self) -> None:
        self.bessel_j_arg_to_top_order = {}

    @override
    def map_call(self,
                 expr: prim.Call,
                 *args: P.args, **kwargs: P.kwargs) -> Expression:
        function = expr.function
        if isinstance(function, prim.Variable) and function.name == "bessel_j":
            order, arg = expr.parameters
            self.rec(arg, *args, **kwargs)

            assert isinstance(order, int)
            self.bessel_j_arg_to_top_order[arg] = max(
                    self.bessel_j_arg_to_top_order.get(arg, 0),
                    abs(order))

        return super().map_call(expr, *args, **kwargs)

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)


class BesselDerivativeReplacer(CSECachingIdentityMapper[P]):
    @override
    def map_call(self,
                 expr: prim.Call,
                 *args: P.args, **kwargs: P.kwargs) -> Expression:
        call = expr

        if (isinstance(call.function, prim.Variable)
                and call.function.name in ["Hankel1", "BesselJ"]):
            if call.function.name == "Hankel1":
                function = prim.Variable("hankel_1")
            else:
                function = prim.Variable("bessel_j")
            order, arg, k = call.parameters
            assert isinstance(order, int)
            assert isinstance(k, int)

            # AS (9.1.31)
            # https://dlmf.nist.gov/10.6.7
            if order >= 0:  # noqa: SIM108
                order_str = f"{order}"
            else:
                order_str = f"m{-order}"

            from math import comb
            return prim.CommonSubexpression(
                    2.0**(-k) * sum(
                        (-1)**idx * comb(k, idx) * function(i, arg)
                        for idx, i in enumerate(range(order-k, order+k+1, 2))),
                    f"d{k}_{function.name}_{order_str}",
                    scope=prim.cse_scope.EVALUATION)
        else:
            return super().map_call(expr, *args, **kwargs)

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)


class BesselSubstitutor(CSECachingIdentityMapper[P]):
    name_gen: Callable[[str], str]
    bessel_j_arg_to_top_order: dict[Expression, int]
    assignments: list[Assignment | CallInstruction]

    cse_cache: dict[Expression, prim.CommonSubexpression]

    def __init__(self,
                 name_gen: Callable[[str], str],
                 bessel_j_arg_to_top_order: dict[Expression, int],
                 assignments: Sequence[Assignment | CallInstruction]) -> None:
        self.name_gen = name_gen
        self.bessel_j_arg_to_top_order = bessel_j_arg_to_top_order
        self.cse_cache = {}
        self.assignments = list(assignments)

    @override
    def map_call(self, expr: prim.Call,
                 *args: P.args, **kwargs: P.kwargs) -> Expression:
        if isinstance(expr.function, prim.Variable):
            name = expr.function.name
            if name == "bessel_j":
                order, arg = expr.parameters
                assert isinstance(order, int)
                assert prim.is_arithmetic_expression(arg)

                return self.bessel_j(order, self.rec_arith(arg, *args, **kwargs))
            elif name == "hankel_1":
                order, arg = expr.parameters
                assert isinstance(order, int)
                assert prim.is_arithmetic_expression(arg)

                return self.hankel_1(order, self.rec_arith(arg, *args, **kwargs))

        return super().map_call(expr, *args, **kwargs)

    def wrap_in_cse(self, expr: Expression, prefix: str) -> prim.CommonSubexpression:
        cse = wrap_in_cse(expr, prefix)
        return self.cse_cache.setdefault(expr, cse)

    # {{{ bessel implementation

    @memoize_method
    def bessel_jv_two(
            self, order: int, arg: Expression
        ) -> tuple[prim.Variable, prim.Variable]:
        om0 = prim.Variable(self.name_gen(f"bessel_{order}"))
        om1 = prim.Variable(self.name_gen(f"bessel_{order - 1}"))

        self.assignments.append(
                make_assignment(
                    (om1, om0),
                    prim.Variable("bessel_jvvp1")(order, arg),
                    temp_var_types=(lp.Optional(None),)*2))

        return om1, om0

    @memoize_method
    def bessel_j(
            self, order: int, arg: ArithmeticExpression
        ) -> ArithmeticExpression:
        top_order = self.bessel_j_arg_to_top_order[arg]
        if order == top_order:
            return self.bessel_jv_two(top_order-1, arg)[1]
        elif order == top_order-1:
            return self.bessel_jv_two(top_order-1, arg)[0]
        elif order < 0:
            return self.wrap_in_cse(
                    (-1.0)**order*self.bessel_j(-order, arg),
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
    def hank1_01(self, arg: Expression) -> tuple[prim.Variable, prim.Variable]:
        hank1_0 = prim.Variable(self.name_gen("hank1_0"))
        hank1_1 = prim.Variable(self.name_gen("hank1_1"))

        self.assignments.append(
                make_assignment(
                    (hank1_0, hank1_1),
                    prim.Variable("hank1_01")(arg),
                    temp_var_types=(lp.Optional(None),)*2))

        return hank1_0, hank1_1

    @memoize_method
    def hankel_1(self, order: int, arg: ArithmeticExpression) -> ArithmeticExpression:
        if order == 0:
            return self.hank1_01(arg)[0]
        elif order == 1:
            return self.hank1_01(arg)[1]
        elif order < 0:
            # AS (9.1.6)
            nu = -order
            return self.wrap_in_cse(
                    (-1.0) ** nu * self.hankel_1(nu, arg),
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

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)

# }}}


# {{{ power rewriter

class PowerRewriter(CSECachingIdentityMapper[P]):
    @override
    def map_power(self,
                  expr: prim.Power,
                  *args: P.args, **kwargs: P.kwargs) -> Expression:
        exp = expr.exponent
        new_base = wrap_in_cse(expr.base)

        if isinstance(exp, int):
            if exp > 2 and exp % 2 == 0:
                square = wrap_in_cse(new_base*new_base)
                return self.rec(wrap_in_cse(square**(exp//2)), *args, **kwargs)
            elif exp == 2:
                return new_base * new_base
            elif exp > 1 and exp % 2 == 1:
                square = wrap_in_cse(new_base*new_base)
                return self.rec(wrap_in_cse(square**((exp-1)//2))*new_base,
                                *args, **kwargs)
            elif exp == 1:
                return new_base
            elif exp < 0:
                return self.rec((1/new_base)**(-exp), *args, **kwargs)

        if (isinstance(exp, prim.Quotient)
                and isinstance(exp.numerator, int)
                and isinstance(exp.denominator, int)):
            p, q = exp.numerator, exp.denominator
            if q < 0:
                q *= -1
                p *= -1

            if q == 1:
                return self.rec(new_base**p, *args, **kwargs)

            if q == 2:
                assert p != 0

                if p > 0:
                    orig_base = wrap_in_cse(expr.base)
                    new_base = wrap_in_cse(prim.Variable("sqrt")(orig_base))
                else:
                    new_base = wrap_in_cse(prim.Variable("rsqrt")(expr.base))
                    p *= -1

                return self.rec(new_base**p, *args, **kwargs)

        return super().map_power(expr, *args, **kwargs)

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)

# }}}


# {{{ convert big integers into floats


class BigIntegerKiller(CSECachingIdentityMapper[P]):
    warn: bool
    float_type: type[np.floating]
    iinfo: np.iinfo

    def __init__(self,
                 warn_on_digit_loss: bool = True,
                 int_type: type[np.integer] = np.int64,
                 float_type: type[np.floating] = np.float64) -> None:
        super().__init__()
        self.warn = warn_on_digit_loss
        self.float_type = float_type
        self.iinfo = np.iinfo(int_type)

    @override
    def map_constant(self, expr: object,
                     *args: P.args, **kwargs: P.kwargs) -> Expression:
        """Convert integer values not within the range of `self.int_type` to float.
        """
        from loopy.typing import is_integer

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

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)

# }}}


# {{{ convert complex to np.complex

class ComplexRewriter(CSECachingIdentityMapper[[]]):
    complex_dtype: np.dtype[np.complexfloating] | None

    def __init__(self,
                 complex_dtype: np.dtype[np.complexfloating] | None = None) -> None:
        super().__init__()
        self.complex_dtype = complex_dtype

    @override
    def map_constant(self, expr: object) -> Expression:
        """Convert complex values to numpy types
        """
        if not isinstance(expr, (complex, np.complex64, np.complex128)):
            return super().map_constant(expr)

        complex_dtype = self.complex_dtype
        if complex_dtype is None:
            if complex(np.complex64(expr)) == expr:
                return np.complex64(expr)

            complex_dtype = np.complex128

        if isinstance(complex_dtype, np.dtype):
            return complex_dtype.type(expr)
        else:
            return complex_dtype(expr)

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr)

# }}}


# {{{ vector component rewriter

INDEXED_VAR_RE = re.compile(r"^([a-zA-Z_]+)([0-9]+)$")


class VectorComponentRewriter(CSECachingIdentityMapper[P]):
    """For names in name_whitelist, turn ``a3`` into ``a[3]``."""

    name_whitelist: frozenset[str]

    def __init__(self, name_whitelist: frozenset[str] | None = None) -> None:
        if name_whitelist is None:
            name_whitelist = frozenset()

        self.name_whitelist = name_whitelist

    @override
    def map_variable(self, expr: prim.Variable,
                     *args: P.args, **kwargs: P.kwargs) -> Expression:
        match_obj = INDEXED_VAR_RE.match(expr.name)
        if match_obj is not None:
            name = match_obj.group(1)
            subscript = int(match_obj.group(2))

            if name in self.name_whitelist:
                return prim.Variable(name)[subscript]
            else:
                return expr
        else:
            return expr

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)

# }}}


# {{{ sum sign grouper

class SumSignGrouper(CSECachingIdentityMapper[P]):
    """Anti-cancellation cargo-cultism."""

    @override
    def map_sum(self, expr: prim.Sum,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        first_group: list[ArithmeticExpression] = []
        second_group: list[ArithmeticExpression] = []

        for orig_child in expr.children:
            child = self.rec_arith(orig_child, *args, **kwargs)
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
        if (len(new_children) == len(expr.children)
                and all(child is orig_child for child, orig_child
                        in zip(new_children, expr.children, strict=True))):
            return expr

        return prim.Sum(tuple(first_group + second_group))

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)

# }}}


class MathConstantRewriter(CSECachingIdentityMapper[P]):
    @override
    def map_variable(self, expr: prim.Variable,
                     *args: P.args, **kwargs: P.kwargs) -> Expression:
        if expr.name == "pi":
            return prim.Variable("M_PI")
        else:
            return expr

    @override
    def map_common_subexpression_uncached(
                self,
                expr: prim.CommonSubexpression,
                *args: P.args, **kwargs: P.kwargs) -> Expression:
        return IdentityMapper.map_common_subexpression(self, expr, *args, **kwargs)


# {{{ to-loopy conversion

def to_loopy_insns(
        assignments: Iterable[tuple[str, sym.Expr]],
        vector_names: Set[str] | None = None,
        pymbolic_expr_maps: Sequence[Callable[[Expression], Expression]] = (),
        complex_dtype: DTypeLike = None,
        retain_names: Set[str] | None = None,
    ) -> Sequence[Assignment | CallInstruction]:
    if vector_names is None:
        vector_names = frozenset()
    vector_names = frozenset(vector_names)

    if retain_names is None:
        retain_names = frozenset()
    retain_names = frozenset(retain_names)

    if complex_dtype is None:
        complex_dtype = np.dtype(np.complex128)
    complex_dtype = np.dtype(complex_dtype)

    logger.info("loopy instruction generation: start")
    assignments = list(assignments)

    # convert from sympy
    sympy_conv = SympyToPymbolicMapper()
    pymbolic_assignments = [(name, sympy_conv(expr)) for name, expr in assignments]

    bdr = BesselDerivativeReplacer()
    btog = BesselTopOrderGatherer()
    vcr = VectorComponentRewriter(vector_names)
    pwr = PowerRewriter()
    ssg = SumSignGrouper()
    bik = BigIntegerKiller()
    cmr = ComplexRewriter(complex_dtype)

    def cmb_mapper(expr: Expression, /) -> Expression:
        expr = bdr(expr)
        expr = vcr(expr)
        expr = pwr(expr)
        expr = ssg(expr)
        expr = bik(expr)
        expr = cmr(expr)
        expr = btog(expr)
        return expr

    def convert_expr(name: str, expr: Expression) -> Expression:
        logger.debug("generate expression for: %s", name)
        expr = cmb_mapper(expr)
        for m in pymbolic_expr_maps:
            expr = m(expr)

        return expr

    pymbolic_assignments = [
        (name, convert_expr(name, expr)) for name, expr in pymbolic_assignments
    ]

    from pytools import UniqueNameGenerator
    name_gen = UniqueNameGenerator({name for name, _expr in pymbolic_assignments})

    result: list[Assignment | CallInstruction] = []
    bessel_sub = BesselSubstitutor(
            name_gen, btog.bessel_j_arg_to_top_order,
            result)

    import loopy as lp
    from pytools import MinRecursionLimit

    with MinRecursionLimit(3000):
        for name, expr in pymbolic_assignments:
            result.append(lp.Assignment(id=None,
                    assignee=name, expression=bessel_sub(expr),
                    temp_var_type=lp.Optional(None)))

    logger.info("loopy instruction generation: done")
    return result

# }}}

# vim: fdm=marker
