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

__doc__ = """

 Symbolic Tools
 ==============

 .. class:: Basic

    The expression base class for the "heavy-duty" computer algebra toolkit
    in use. Either :class:`sympy.core.basic.Basic` or :class:`symengine.Basic`.

 .. autoclass:: SpatialConstant
"""


import logging
import math
from typing import TYPE_CHECKING, Any, ClassVar, cast

from typing_extensions import override

import pymbolic.primitives as prim
from pymbolic.mapper import IdentityMapper as IdentityMapperBase


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pymbolic.typing import ArithmeticExpression, Expression
    from pytools import T
    from pytools.obj_array import ObjectArray1D

logger = logging.getLogger(__name__)

USE_SYMENGINE = False


# {{{ symbolic backend

def _find_symbolic_backend():
    global USE_SYMENGINE

    try:
        import symengine  # noqa: F401
        symengine_found = True
        symengine_error = None
    except ImportError as import_error:
        symengine_found = False
        symengine_error = import_error

    allowed_backends = ("sympy", "symengine")
    backend_env_var = "SUMPY_FORCE_SYMBOLIC_BACKEND"

    import os
    backend = os.environ.get(backend_env_var)
    if backend is not None:
        if backend not in allowed_backends:
            raise RuntimeError(
                f"{backend_env_var} value is unrecognized: '{backend}' "
                "(allowed values are {})".format(
                    ", ".join(f"'{val}'" for val in allowed_backends))
                )

        if backend == "symengine" and not symengine_found:
            raise RuntimeError(f"could not find SymEngine: {symengine_error}")

        USE_SYMENGINE = (backend == "symengine")  # pyright: ignore[reportConstantRedefinition]
    else:
        USE_SYMENGINE = symengine_found  # pyright: ignore[reportConstantRedefinition]


_find_symbolic_backend()

# }}}


# {{{ symbolic API classes

if TYPE_CHECKING or not USE_SYMENGINE:
    import sympy as sym

    from pymbolic.interop.sympy import (
        PymbolicToSympyMapper as PymbolicToSympyMapperBase,
        SympyToPymbolicMapper as SympyToPymbolicMapperBase,
    )
else:
    import symengine as sym

    from pymbolic.interop.symengine import (
        PymbolicToSymEngineMapper as PymbolicToSympyMapperBase,
        SymEngineToPymbolicMapper as SympyToPymbolicMapperBase,
    )

# Symbolic API common to SymEngine and sympy.
# Before adding a function here, make sure it's present in both modules.
Add = sym.Add
Basic = sym.Basic
Expr = sym.Expr
Mul = sym.Mul
Pow = sym.Pow
exp = sym.exp
sqrt = sym.sqrt
log = sym.log
symbols = sym.symbols
sympify = sym.sympify
cos = sym.cos
sin = sym.sin
atan2 = sym.atan2
Function = sym.Function
Symbol = sym.Symbol
Derivative = sym.Derivative
Integer = sym.Integer
Rational = sym.Rational
Matrix = sym.Matrix
Subs = sym.Subs
I = cast("Expr", sym.I)  # noqa: E741
pi = cast("Expr", sym.pi)
functions = sym.functions
Number = sym.Number
Float = sym.Float


def _coeff_isneg(a: Basic) -> bool:
    if a.is_Mul:
        a = a.args[0]

    return a.is_Number and bool(a.is_negative)


if TYPE_CHECKING or USE_SYMENGINE:
    def UnevaluatedExpr(x: T) -> T:  # noqa: N802
        return x
else:
    try:
        from sympy import UnevaluatedExpr
    except ImportError:
        def UnevaluatedExpr(x):  # noqa: N802
            return x


if USE_SYMENGINE:
    def doit(expr: Expr) -> Expr:
        return expr

    def unevaluated_pow(a: Expr, b: complex | Expr) -> Expr:
        return Pow(a, b)
else:
    def doit(expr: Expr) -> Expr:
        return expr.doit()

    def unevaluated_pow(a: Expr, b: complex | Expr) -> Expr:
        return Pow(a, b, evaluate=False)

# }}}


# {{{ debugging of sympy CSE via Maxima

class _DerivativeKiller(IdentityMapperBase[[]]):
    @override
    def map_derivative(self, expr: prim.Derivative) -> Expression:
        return prim.Variable("d_{}".format("_".join(expr.variables)))(expr.child)

    @override
    def map_substitution(self, expr: prim.Substitution) -> Expression:
        return self.rec(expr.child)


def _get_assignments_in_maxima(
            assignments: dict[str, Basic],
            prefix: str = "",
        ) -> str:
    variable_names = set(assignments.keys())
    written_assignments: set[str] = set()
    prefix_subst_dict = {vn: f"{prefix}{vn}" for vn in variable_names}

    from pymbolic.interop.maxima import MaximaStringifyMapper

    mstr = MaximaStringifyMapper()
    s2p = SympyToPymbolicMapper()
    dkill = _DerivativeKiller()

    result: list[str] = []

    def write_assignment(name: str) -> None:
        symbols = [atm for atm in assignments[name].atoms()
                if isinstance(atm, Symbol)
                and atm.name in variable_names]

        for symb in symbols:
            if symb.name not in written_assignments:
                write_assignment(symb.name)

        result.append("{}{} : {};".format(
            prefix, name, mstr(dkill(s2p(
                assignments[name].subs(prefix_subst_dict))))))
        written_assignments.add(name)

    for name in assignments:
        if name not in written_assignments:
            write_assignment(name)

    return "\n".join(result)


def checked_cse(exprs, symbols=None):
    kwargs = {}
    if symbols is not None:
        kwargs["symbols"] = symbols

    new_assignments, new_exprs = sym.cse(exprs, **kwargs)

    max_old = _get_assignments_in_maxima({
            f"old_expr{i}": expr
            for i, expr in enumerate(exprs)})
    new_assign_dict = {
            f"new_expr{i}": expr
            for i, expr in enumerate(new_exprs)}
    for name, val in new_assignments:
        new_assign_dict[name.name] = val
    max_new = _get_assignments_in_maxima(new_assign_dict)

    with open("check.mac", "w") as outf:
        outf.write("ratprint:false;\n")
        outf.write(f"{max_old}\n\n")
        outf.write(f"{max_new}\n")
        for i in range(len(exprs)):
            outf.write(f'print("diff in expr {i}:\n");\n')
            outf.write(f"print(ratsimp(old_expr{i} - new_expr{i}));\n")

    from subprocess import check_call
    check_call(["maxima", "--very-quiet", "-r", 'load("check.mac");'])

    return new_assignments, new_exprs

# }}}


# {{{ solve_lu

def forward_substitution(
        L: Matrix,  # noqa: N803
        b: Matrix,
        postprocess: Callable[[Basic], Basic],
    ) -> Matrix:
    """Given a lower triangular matrix *L* and a column vector *b*,
    solve the system ``Lx = b`` and apply the callable *postprocess_division*
    on each expression at the end of division calls.
    """
    n = len(b)
    x = Matrix(b)

    for i in range(n):
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] = postprocess(x[i] / L[i, i])

    return x


def backward_substitution(
        U: Matrix,  # noqa: N803
        b: Matrix,
        postprocess: Callable[[Basic], Basic],
    ) -> Matrix:
    """Given an upper triangular matrix *U* and a column vector *b*,
    solve the system ``Ux = b`` and apply the callable *postprocess_division*
    on each expression at the end of division calls.
    """
    n = len(b)
    x = sym.Matrix(b)

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            x[i] -= U[i, j] * x[j]
        x[i] = postprocess(x[i] / U[i, i])

    return x


def solve_lu(
        L: Matrix,  # noqa: N803
        U: Matrix,  # noqa: N803
        permutation: Sequence[tuple[int, int]],
        b: Matrix, *,
        postprocess: Callable[[Basic], Basic] | None = None,
    ) -> Matrix:
    if postprocess is None:
        def default_postprocess(x: Basic) -> Basic:
            return x

        postprocess = default_postprocess

    # Permute first
    b = sym.Matrix(b)
    for p, q in permutation:
        b[p], b[q] = b[q], b[p]

    return backward_substitution(
        U,
        forward_substitution(L, b, postprocess),
        postprocess,
    )

# }}}


# {{{ helpers


if USE_SYMENGINE:
    def evalf(expr: Expr, prec: int = 100) -> Expr:
        """Evaluate an expression numerically using ``prec`` number of bits."""
        return expr.n(prec=prec)

    def simplify(expr: Expr) -> Expr:
        return expr.simplify()
else:
    def evalf(expr: Expr, prec: int = 100) -> Expr:
        """Evaluate an expression numerically using ``prec`` number of bits."""
        dps = int(sym.log(2**prec, 10))
        return expr.n(n=dps)

    def simplify(expr: Expr) -> Expr:
        return sym.simplify(expr)


def chop(expr: Basic, tol: float = 1.0e-8) -> Basic:
    """Replace numbers in *expr* with their closest integer.

    Given a symbolic expression, this removes all occurrences of numbers
    with absolute value less than the given tolerance and replaces floating
    point numbers that are close to an integer up to the given relative
    tolerance by the integer.
    """
    nums = expr.atoms(Number)
    replace_dict: dict[Any, float] = {}
    for num in nums:
        if float(abs(num)) < tol:
            replace_dict[num] = 0
        else:
            new_num = float(num)
            if abs(new_num) < tol or abs(int(new_num) - new_num) < tol * abs(new_num):
                new_num = int(new_num)

            replace_dict[num] = new_num

    return expr.xreplace(replace_dict)


def sym_real_norm_2(x: Matrix) -> Expr:
    return sqrt((x.T*x)[0, 0])


def pymbolic_real_norm_2(
        x: ObjectArray1D[ArithmeticExpression]
    ) -> ArithmeticExpression:
    return prim.Variable("sqrt")(x @ x)


def make_sym_vector(name: str, components: int) -> Matrix:
    return Matrix([Symbol(f"{name}{i}") for i in range(components)])


@prim.expr_dataclass()
class SpatialConstant(prim.Variable):
    """A symbolic constant to represent a symbolic variable that is spatially constant.

    For example the wave-number :math:`k` in the setting of a constant-coefficient
    Helmholtz problem. For use in :attr:`sumpy.kernel.ExpressionKernel.expression`.
    Any variable occurring there that is not a :class:`~sumpy.symbolic.SpatialConstant`
    is assumed to have a spatial dependency.

    .. autoattribute:: prefix
    .. automethod:: as_sympy
    .. automethod:: from_sympy
    """

    prefix: ClassVar[str] = "_spatial_constant_"
    """Prefix used in code generation for variables of this type."""

    def as_sympy(self) -> Symbol:
        """Convert this variable to a :mod:`sympy` expression."""
        return Symbol(f"{self.prefix}{self.name}")

    @classmethod
    def from_sympy(cls, expr: Symbol) -> SpatialConstant:
        """Convert a :mod:`sympy` expression to a constant."""
        if isinstance(expr, Symbol) and expr.name.startswith(cls.prefix):
            return cls(expr.name[len(cls.prefix):])

        raise ValueError(f"expression is not a spatial constant: {expr!r}")

# }}}


# {{{ PymbolicToSym mappers

class PymbolicToSympyMapper(PymbolicToSympyMapperBase):
    def map_spatial_constant(self, expr: SpatialConstant) -> Basic:
        return expr.as_sympy()


class SympyToPymbolicMapper(SympyToPymbolicMapperBase):
    @override
    def map_Symbol(self, expr: Symbol) -> Expression:
        try:
            return SpatialConstant.from_sympy(expr)
        except ValueError:
            return SympyToPymbolicMapperBase.map_Symbol(self, expr)

    @override
    def map_Pow(self, expr: Pow) -> Expression:
        if expr.exp == -1:
            return 1 / self.rec_arith(expr.base)
        else:
            return SympyToPymbolicMapperBase.map_Pow(self, expr)

    @override
    def map_Mul(self, expr: Mul) -> Expression:
        num_args: list[ArithmeticExpression] = []
        den_args: list[ArithmeticExpression] = []
        for child in expr.args:
            if (isinstance(child, Pow)
                    and isinstance(child.exp, Integer)
                    and child.exp < 0):
                den_args.append(self.rec_arith(child.base)**(-self.rec_arith(child.exp)))
            else:
                num_args.append(self.rec_arith(child))

        return math.prod(num_args) / math.prod(den_args)


class PymbolicToSympyMapperWithSymbols(PymbolicToSympyMapper):
    @override
    def map_variable(self, expr: prim.Variable) -> Basic:
        if expr.name == "I":
            return I
        elif expr.name == "pi":
            return pi
        else:
            return PymbolicToSympyMapper.map_variable(self, expr)

    @override
    def map_subscript(self, expr: prim.Subscript) -> sym.Basic:
        if isinstance(expr.aggregate, prim.Variable) and isinstance(expr.index, int):
            return Symbol(f"{expr.aggregate.name}{expr.index}")
        else:
            self.raise_conversion_error(expr)

    @override
    def map_call(self, expr: prim.Call) -> sym.Basic:
        function = expr.function
        if isinstance(function, prim.Variable):
            if function.name == "hankel_1":
                args = [self.rec(param) for param in expr.parameters]
                args.append(sympify(0))
                return Hankel1(*args)
            elif function.name == "bessel_j":
                args = [self.rec(param) for param in expr.parameters]
                args.append(sympify(0))
                return BesselJ(*args)

        return PymbolicToSympyMapper.map_call(self, expr)


def to_sympy(expr: Expression) -> Basic:
    return PymbolicToSympyMapperWithSymbols().to_expr(expr)


def to_pymbolic(expr: Basic) -> Expression:
    return SympyToPymbolicMapper()(expr)

# }}}


# {{{ Bessel and Hankel functions

from sympy import Function as SympyFunction


class _BesselOrHankel(SympyFunction):
    """A symbolic function for BesselJ or Hankel1 functions
    that keeps track of the derivatives taken of the function.
    Arguments are ``(order, z, nderivs)``.
    """
    nargs: ClassVar[tuple[int, ...]] = (3,)

    @override
    def fdiff(self, argindex: int = 1) -> Basic:
        if argindex in (1, 3):
            # we are not differentiating w.r.t order or nderivs
            raise ValueError(f"invalid argindex: {argindex}")

        order, z, nderivs = self.args
        return self.func(order, z, nderivs + 1)


class BesselJ(_BesselOrHankel):
    pass


class Hankel1(_BesselOrHankel):
    pass


_SympyBesselJ = BesselJ
_SympyHankel1 = Hankel1

if not TYPE_CHECKING and USE_SYMENGINE:
    def BesselJ(*args):   # noqa: N802
        return sympify(_SympyBesselJ(*args))

    def Hankel1(*args):   # noqa: N802
        return sympify(_SympyHankel1(*args))

# }}}

# vim: fdm=marker
