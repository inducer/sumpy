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
HAS_SYMENGINE = False


# {{{ symbolic backend

def _find_symbolic_backend():
    global USE_SYMENGINE
    global HAS_SYMENGINE

    try:
        import symengine  # ruff:ignore[unused-import]
        HAS_SYMENGINE = True
        symengine_error = None
    except ImportError as import_error:
        HAS_SYMENGINE = False
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

        if backend == "symengine" and not HAS_SYMENGINE:
            raise RuntimeError(f"could not find SymEngine: {symengine_error}")

        USE_SYMENGINE = (backend == "symengine")  # pyright: ignore[reportConstantRedefinition]
    else:
        USE_SYMENGINE = HAS_SYMENGINE  # pyright: ignore[reportConstantRedefinition]


_find_symbolic_backend()

# }}}


# {{{ symbolic expressions


if TYPE_CHECKING or not USE_SYMENGINE:
    import sympy as sym
else:
    import symengine as sym


from pymbolic.interop.sympy import (
    PymbolicToSympyMapper as PymbolicToSympyMapperBase,
    SympyToPymbolicMapper as SympyToPymbolicMapperBase,
)


if HAS_SYMENGINE:
    from pymbolic.interop.symengine import (
        PymbolicToSymEngineMapper as PymbolicToSymEngineMapperBase,
        SymEngineToPymbolicMapper as SymEngineToPymbolicMapperBase,
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
I = cast("Expr", sym.I)  # ruff:ignore[ambiguous-variable-name]
pi = cast("Expr", sym.pi)
functions = sym.functions
Number = sym.Number
Float = sym.Float


def _coeff_isneg(a: Basic) -> bool:
    if a.is_Mul:
        a = a.args[0]

    return a.is_Number and bool(a.is_negative)


if TYPE_CHECKING or USE_SYMENGINE:
    def UnevaluatedExpr(x: T) -> T:  # ruff:ignore[invalid-function-name]
        return x
else:
    try:
        from sympy import UnevaluatedExpr
    except ImportError:
        def UnevaluatedExpr(x):  # ruff:ignore[invalid-function-name]
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
            prefix, name, mstr(dkill(to_pymbolic(
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

def _solve_lu_forward_substitution(
        L: Matrix,  # ruff: ignore[invalid-argument-name]
        b: Matrix,
        postprocess: Callable[[Basic], Basic],
    ) -> Matrix:
    """Solve the lower triangular system :math:`L x = b`.

    :arg postprocess: a callable that is applied on each expression at the end
        of division calls.
    """
    n = len(b)
    x = b

    for i in range(n):
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] = postprocess(x[i] / L[i, i])

    return x


def _solve_lu_backward_substitution(
        U: Matrix,  # ruff: ignore[invalid-argument-name]
        b: Matrix,
        postprocess: Callable[[Basic], Basic],
    ) -> Matrix:
    """Solve the upper triangular system :math:`U x = b`.

    :arg postprocess: a callable that is applied on each expression at the end
        of division calls.
    """
    n = len(b)
    x = b

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            x[i] -= U[i, j] * x[j]
        x[i] = postprocess(x[i] / U[i, i])

    return x


def solve_lu(
        L: Matrix,  # ruff: ignore[invalid-argument-name]
        U: Matrix,  # ruff: ignore[invalid-argument-name]
        permutation: Sequence[tuple[int, int]],
        b: Matrix, *,
        postprocess: Callable[[Basic], Basic] | None = None,
    ) -> Matrix:
    """Solve the system :math:`L U x = P b`.

    This function uses standard forward and backward substitution to solve the
    system. The matrix *L* is assumed to be unit lower triangular, *U* is
    assumed to be upper triangular and *permutation* is a sequence of row swaps
    for the LU decomposition.

    :arg postprocess: a callable that is applied on each expression at the end
        of division calls in both forward and backward substitution.
    """

    if postprocess is None:
        def default_postprocess(x: Basic) -> Basic:
            return x

        postprocess = default_postprocess

    if L.shape[0] != U.shape[1]:
        raise ValueError(
            f"system matrix is not square: L is {L.shape} and U is {U.shape}"
        )

    if L.shape[1] != U.shape[0]:
        raise ValueError(
            f"incorrect LU decomposition shapes: L is {L.shape} and U is {U.shape}"
        )

    if b.shape != (U.shape[1], 1):
        raise ValueError(f"'b' is not a column vector matching U: {b.shape}")

    b = Matrix(b)

    # Permute first
    for p, q in permutation:
        b[p], b[q] = b[q], b[p]

    y = _solve_lu_forward_substitution(L, b, postprocess=postprocess)
    x = _solve_lu_backward_substitution(U, y, postprocess=postprocess)

    return x

# }}}


# {{{ pymbolic expressions

def sym_real_norm_2(x: Matrix) -> Expr:
    return sqrt((x.T*x)[0, 0])


def pymbolic_real_norm_2(
            x: ObjectArray1D[ArithmeticExpression]) -> ArithmeticExpression:
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


# {{{ sympy <-> pymbolic interop


class PymbolicToSympyLikeMixin:
    def map_variable(self, expr: prim.Variable) -> Basic:
        if not self.symbols:
            return super().map_variable(expr)

        if expr.name == "I":
            return self.sym.I
        elif expr.name == "pi":
            return self.sym.pi
        else:
            return super().map_variable(expr)

    def map_subscript(self, expr: prim.Subscript) -> sym.Basic:
        if not self.symbols:
            return super().map_subscript(expr)

        if isinstance(expr.aggregate, prim.Variable) and isinstance(expr.index, int):
            return self.sym.Symbol(f"{expr.aggregate.name}{expr.index}")
        else:
            self.raise_conversion_error(expr)

    def map_call(self, expr: prim.Call) -> sym.Basic:
        if not self.symbols:
            return super().map_call(expr)

        function = expr.function
        if isinstance(function, prim.Variable):
            if function.name == "hankel_1":
                args = [self.rec(param) for param in expr.parameters]
                args.append(self.sym.sympify(0))
                return Hankel1(*args)
            elif function.name == "bessel_j":
                args = [self.rec(param) for param in expr.parameters]
                args.append(self.sym.sympify(0))
                return BesselJ(*args)

        return super().map_call(expr)


class PymbolicToSympyMapper(PymbolicToSympyLikeMixin, PymbolicToSympyMapperBase):
    symbols: bool

    def __init__(self, *, symbols: bool = False) -> None:
        self.symbols = symbols

    def map_spatial_constant(self, expr: SpatialConstant) -> Basic:
        return self.sym.Symbol(f"{expr.prefix}{expr.name}")


if HAS_SYMENGINE:
    class PymbolicToSymEngineMapper(PymbolicToSympyLikeMixin,
                                    PymbolicToSymEngineMapperBase):
        symbols: bool

        def __init__(self, *, symbols: bool = False) -> None:
            self.symbols = symbols

        def map_spatial_constant(self, expr: SpatialConstant) -> Basic:
            return self.sym.Symbol(f"{expr.prefix}{expr.name}")


class SympyLikeToPymbolicMixin:
    def map_Symbol(self, expr: Symbol) -> Expression:  # ruff: ignore[invalid-function-name]
        if (isinstance(expr, self.sym.Symbol)
            and expr.name.startswith(SpatialConstant.prefix)):
            return SpatialConstant(expr.name[len(SpatialConstant.prefix):])

        return super().map_Symbol(expr)

    def map_Pow(self, expr: Pow) -> Expression:  # ruff: ignore[invalid-function-name]
        if expr.exp == -1:
            return 1 / self.rec_arith(expr.base)
        else:
            return super().map_Pow(expr)

    def map_Mul(self, expr: Mul) -> Expression:  # ruff: ignore[invalid-function-name]
        num_args: list[ArithmeticExpression] = []
        den_args: list[ArithmeticExpression] = []
        for child in expr.args:
            if (isinstance(child, self.sym.Pow)
                    and isinstance(child.exp, self.sym.Integer)
                    and child.exp < 0):
                den_args.append(self.rec_arith(child.base)**(-self.rec_arith(child.exp)))
            else:
                num_args.append(self.rec_arith(child))

        return math.prod(num_args) / math.prod(den_args)

    def not_supported(self, expr: object) -> Expression:
        if self.symbols and getattr(expr, "is_Function", False):
            function_name = self.function_name(expr)
            if function_name in {"Hankel1", "BesselJ"}:
                order, arg, nderivs = expr.args
                if nderivs == 0:
                    return prim.Variable({
                        "Hankel1": "hankel_1",
                        "BesselJ": "bessel_j",
                    }[function_name])(self.rec(order), self.rec(arg))

        return super().not_supported(expr)


class SympyToPymbolicMapper(SympyLikeToPymbolicMixin,
                            SympyToPymbolicMapperBase):
    symbols: bool

    def __init__(self, *, symbols: bool = False) -> None:
        self.symbols = symbols

    @property
    def sym(self) -> Any:
        import sympy as sp
        return sp

    @override
    def map_NumberSymbol(self, expr: sym.NumberSymbol) -> Expression:
        if not self.symbols:
            return super().map_NumberSymbol(expr)

        if expr is self.sym.pi:
            return prim.Variable("pi")
        elif expr is self.sym.I:
            return prim.Variable("I")
        else:
            return super().map_NumberSymbol(expr)


if HAS_SYMENGINE:
    class SymEngineToPymbolicMapper(SympyLikeToPymbolicMixin,
                                    SymEngineToPymbolicMapperBase):
        symbols: bool

        def __init__(self, *, symbols: bool = False) -> None:
            self.symbols = symbols

        @property
        def sym(self) -> Any:
            import symengine as sp
            return sp

        @override
        def map_Constant(self, expr: object) -> Expression:
            if not self.symbols:
                return super().map_Constant(expr)

            if expr is self.sym.pi:
                return prim.Variable("pi")
            elif expr is self.sym.I:
                return prim.Variable("I")
            else:
                return super().map_Constant(expr)


def to_pymbolic(expr: Basic, *, symbols: bool = False) -> Expression:
    r"""Convert the symbolic expression *expr* to :mod:`pymbolic`.

    :arg symbols: if *True*, this will also convert known symbols (e.g. :math:`\pi`)
        to variables.
    """
    if USE_SYMENGINE:
        p2s = SymEngineToPymbolicMapper(symbols=symbols)
    else:
        p2s = SympyToPymbolicMapper(symbols=symbols)

    return p2s(expr)


def to_symbolic(expr: Expression, *, symbols: bool = False) -> Basic:
    r"""Convert from :mod:`pymbolic` to a symbolic object.

    :arg symbols: if *True*, this will also convert known symbols (e.g. :math:`\pi`)
        to their respective constants in the symbolic backend.
    """
    if USE_SYMENGINE:
        s2p = PymbolicToSymEngineMapper(symbols=symbols)
    else:
        s2p = PymbolicToSympyMapper(symbols=symbols)

    return s2p(expr)


def to_symbolic_expr(expr: Expression, *, symbols: bool = False) -> Expr:
    r"""Convert from :mod:`pymbolic` to a symbolic expression.

    This is similar to :func:`to_symbolic`, but will raise an exception if the
    result is not an algebraic expression.
    """
    result = to_symbolic(expr, symbols=symbols)
    if not isinstance(result, Expr):
        raise TypeError(f"result is not an expression: {type(result)}")

    return result

# }}}


# {{{ symbolic functions

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
    def BesselJ(*args):   # ruff:ignore[invalid-function-name]
        return sympify(_SympyBesselJ(*args))

    def Hankel1(*args):   # ruff:ignore[invalid-function-name]
        return sympify(_SympyHankel1(*args))

# }}}

# vim: fdm=marker
