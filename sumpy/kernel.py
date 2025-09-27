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

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from typing_extensions import override

import loopy as lp
import pymbolic.primitives as prim
from pymbolic import Expression, var
from pymbolic.mapper import CSECachingMapperMixin, IdentityMapper
from pymbolic.primitives import make_sym_vector
from pytools import memoize_method

import sumpy.symbolic as sym
from sumpy.derivative_taker import (
    DerivativeCoeffDict,
    DifferentiatedExprDerivativeTaker,
    ExprDerivativeTaker,
    diff_derivative_coeff_dict,
)
from sumpy.symbolic import SpatialConstant, pymbolic_real_norm_2


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import sympy as sp

    from sumpy.assignment_collection import SymbolicAssignmentCollection
    from sumpy.expansion.diff_op import LinearPDESystemOperator

__doc__ = """
Kernel interface
----------------

.. autoclass:: ArithmeticExpr

.. autoclass:: KernelArgument
.. autoclass:: Kernel
    :show-inheritance:

Symbolic kernels
----------------

.. autoclass:: ExpressionKernel
    :show-inheritance:
    :members: mapper_method

PDE kernels
-----------

.. autoclass:: LaplaceKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: BiharmonicKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: HelmholtzKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: YukawaKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: StokesletKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: StressletKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: ElasticityKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: LineOfCompressionKernel
    :show-inheritance:
    :members: mapper_method

Derivatives
-----------

These objects *wrap* other kernels and take derivatives
of them in the process.

.. autoclass:: DerivativeBase
.. autoclass:: AxisTargetDerivative
    :show-inheritance:
    :undoc-members:
    :members: mapper_method,target_array_name
.. autoclass:: AxisSourceDerivative
    :show-inheritance:
    :members: mapper_method
.. autoclass:: DirectionalDerivative
    :show-inheritance:
    :members: directional_kind
.. autoclass:: DirectionalSourceDerivative
    :show-inheritance:
    :members: mapper_method,directional_kind

Transforming kernels
--------------------

.. autoclass:: TargetPointMultiplier
    :undoc-members:
    :members: mapper_method,target_array_name

.. autoclass:: ResultT

.. autoclass:: KernelMapper
    :show-inheritance:
.. autoclass:: KernelCombineMapper
    :show-inheritance:
.. autoclass:: KernelIdentityMapper
    :show-inheritance:
.. autoclass:: AxisSourceDerivativeRemover
    :show-inheritance:
.. autoclass:: AxisTargetDerivativeRemover
    :show-inheritance:
.. autoclass:: SourceDerivativeRemover
    :show-inheritance:
.. autoclass:: TargetDerivativeRemover
    :show-inheritance:
.. autoclass:: TargetTransformationRemover
    :show-inheritance:
.. autoclass:: DerivativeCounter
    :show-inheritance:
"""

ArithmeticExpr: TypeAlias = int | float | complex | sym.Basic


@dataclass(frozen=True)
class KernelArgument:
    """
    .. autoattribute:: loopy_arg
    """

    loopy_arg: lp.KernelArgument
    """A :class:`loopy.KernelArgument` instance describing the type, name, and
    other features of this kernel argument when passed to a generated piece of
    code.
    """

    @property
    def name(self) -> str:
        return self.loopy_arg.name


# {{{ basic kernel interface

@dataclass(frozen=True)
class Kernel(ABC):
    """Basic kernel interface.

    .. autoattribute:: mapper_method
    .. autoattribute:: is_translation_invariant

    .. autoattribute:: dim
    .. autoproperty:: is_complex_valued

    .. automethod:: get_base_kernel
    .. automethod:: replace_base_kernel
    .. automethod:: prepare_loopy_kernel
    .. automethod:: get_code_transformer
    .. automethod:: get_expression
    .. automethod:: postprocess_at_source
    .. automethod:: postprocess_at_target
    .. automethod:: get_global_scaling_const
    .. automethod:: get_args
    .. automethod:: get_source_args
    .. automethod:: get_pde_as_diff_op
    """

    dim: int
    """Dimension of the space the kernel is defined in."""

    # TODO: Allow kernels that are not translation invariant
    is_translation_invariant: ClassVar[bool] = True
    """A boolean flag indicating whether the kernel is translation invariant."""
    mapper_method: ClassVar[str]
    """The name of the mapper method called for the kernel."""

    @property
    @abstractmethod
    def is_complex_valued(self) -> bool:
        """A boolean flag indicating whether this kernel is complex valued."""

    @override
    def __repr__(self) -> str:
        from dataclasses import fields

        args: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, prim.ExpressionNode):
                args.append(f"{f.name}={value}")
            else:
                args.append(f"{f.name}={value!r}")

        return f"{type(self).__name__}({', '.join(args)})"

    def get_base_kernel(self) -> Kernel:
        """
        :returns: the kernel being wrapped by this one, or else *self*.
        """
        return self

    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        """
        :returns: the base kernel being wrapped by this one, or else
            *new_base_kernel*.
        """
        return new_base_kernel

    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        """Apply some changes (such as registering function manglers) to the kernel.

        :returns: a new :mod:`loopy` kernel with the applied changes.
        """
        return loopy_knl

    def get_code_transformer(self) -> Callable[[Expression], Expression]:
        """
        :returns: a function to postprocess the :mod:`pymbolic` expression
            generated from the result of :meth:`get_expression` on the way to
            code generation.
        """
        return lambda expr: expr

    @abstractmethod
    def get_expression(self, dist_vec: sp.Matrix) -> ArithmeticExpr:
        """
        :returns: a :mod:`sympy` expression for the kernel.
        """

    @abstractmethod
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        r"""
        :returns: the PDE for the kernel as a
            :class:`sumpy.expansion.diff_op.LinearPDESystemOperator` object
            :math:`\mathcal{L}`, where :math:`\mathcal{L}(u) = 0` is the PDE.
        """

    def _diff(self,
              expr: sym.Expr,
              vec: sp.Matrix,
              mi: tuple[int, ...]) -> sym.Expr:
        """Take the derivative of an expression."""
        for i in range(self.dim):
            if mi[i] == 0:
                continue
            expr = expr.diff(vec[i], mi[i])

        return expr

    @abstractmethod
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        """
        :returns: an :class:`~sumpy.derivative_taker.ExprDerivativeTaker` instance
            that supports taking derivatives of the base kernel with respect to
            *dvec*.
        """

    @overload
    def postprocess_at_source(
            self, expr: sym.Expr, avec: sp.Matrix
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_source(
            self, expr: ExprDerivativeTaker, avec: sp.Matrix
        ) -> DifferentiatedExprDerivativeTaker: ...

    def postprocess_at_source(
            self, expr: sym.Expr | ExprDerivativeTaker, avec: sp.Matrix,
        ) -> sym.Expr | DifferentiatedExprDerivativeTaker:
        """Transform a kernel evaluation or expansion expression in a place
        where the vector :math:`a` (something - source) is known. ("something" may be
        an expansion center or a target)

        The typical use of this function is to apply source-variable
        derivatives to the kernel.
        """
        expr_dict: DerivativeCoeffDict = {(0,)*self.dim: 1}
        expr_dict = self.get_derivative_coeff_dict_at_source(expr_dict)
        if isinstance(expr, ExprDerivativeTaker):
            return DifferentiatedExprDerivativeTaker(expr, expr_dict)

        result = 0
        for mi, coeff in expr_dict.items():
            result += coeff * self._diff(expr, avec, mi)

        assert isinstance(result, sym.Expr)
        return result

    @overload
    def postprocess_at_target(
            self, expr: sym.Expr, bvec: sp.Matrix,
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_target(
            self, expr: ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> DifferentiatedExprDerivativeTaker: ...

    def postprocess_at_target(
            self, expr: sym.Expr | ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> sym.Expr | DifferentiatedExprDerivativeTaker:
        """Transform a kernel evaluation or expansion expression in a place
        where the vector :math:`b` (target - something) is known. ("something" may
        be an expansion center or a target.)

        The typical use of this function is to apply target-variable
        derivatives to the kernel.
        """
        return expr

    def get_derivative_coeff_dict_at_source(
            self, expr_dict: DerivativeCoeffDict,
        ) -> DerivativeCoeffDict:
        r"""Get the derivative transformation of the expression at the source.

        The transformation is represented by the *expr_dict* which maps from a
        multi-index *mi* to a coefficient *coeff*. The Expression represented by
        *expr_dict* is :math:`\sum_{mi} \frac{\partial^mi}{x^mi}G * coeff`.

        This function is meant to be overridden by child classes where necessary.
        """
        return expr_dict

    @abstractmethod
    def get_global_scaling_const(self) -> ArithmeticExpr:
        r"""A global scaling constant of the kernel.

        Typically, this ensures that the kernel is scaled so that
        :math:`\mathcal{L}(G)(x) = C \delta(x)` with a constant of 1, where
        :math:`\mathcal{L}` is the PDE operator associated with the kernel. Not
        to be confused with *rscale*, which keeps expansion coefficients
        benignly scaled.
        """

    def get_args(self) -> Sequence[KernelArgument]:
        """
        :returns: list of :class:`KernelArgument` instances describing extra
            arguments used by the kernel.
        """
        return []

    def get_source_args(self) -> Sequence[KernelArgument]:
        """
        :returns: list of :class:`KernelArgument` instances describing extra
            arguments used by kernel in picking up contributions from point sources.
        """
        return []

# }}}


@dataclass(frozen=True, repr=False)
class ExpressionKernel(Kernel, ABC):
    r"""
    .. autoattribute:: expression
    .. autoattribute:: global_scaling_const
    """

    mapper_method: ClassVar[str] = "map_expression_kernel"

    expression: Expression
    """A :mod:`pymbolic` expression depending on variables *d_1* through *d_N*
    where *N* equals *dim*. These variables match what is returned from
    :func:`pymbolic.primitives.make_sym_vector` with argument `"d"`. Any
    variable that is not *d* or a :class:`~sumpy.symbolic.SpatialConstant` will
    be viewed as potentially spatially varying.
    """

    global_scaling_const: Expression
    r"""A constant :mod:`pymbolic` expression for the global scaling of the
    kernel. Typically, this ensures that the kernel is scaled so that
    :math:`\mathcal{L}(G)(x)=C\delta(x)` with a constant of 1, where
    :math:`\mathcal{L}` is the PDE operator associated with the kernel. Not to
    be confused with *rscale*, which keeps expansion coefficients benignly
    scaled.
    """

    @override
    def __str__(self) -> str:
        return f"ExprKnl{self.dim}D"

    @override
    def get_expression(self, dist_vec: sp.Matrix) -> ArithmeticExpr:
        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        expr = PymbolicToSympyMapperWithSymbols()(self.expression)

        if self.dim != len(dist_vec):
            raise ValueError(
                "'dist_vec' length does not match expected dimension: "
                f"kernel dim is '{self.dim}' and dist_vec has length '{len(dist_vec)}'")

        from sumpy.symbolic import Symbol
        expr = expr.xreplace({
            Symbol(f"d{i}"): dist_vec_i
            for i, dist_vec_i in enumerate(dist_vec)
            })

        return expr

    @override
    def get_global_scaling_const(self) -> ArithmeticExpr:
        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        return PymbolicToSympyMapperWithSymbols()(self.global_scaling_const)

    @override
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        return ExprDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        raise NotImplementedError


class OneKernel(ExpressionKernel):
    def __init__(self, dim: int):
        super().__init__(
            dim=dim,
            expression=1,
            global_scaling_const=1,
        )

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False


one_kernel_2d = OneKernel(2)
one_kernel_3d = OneKernel(3)


# {{{ PDE kernels

class LaplaceKernel(ExpressionKernel):
    mapper_method: ClassVar[str] = "map_laplace_kernel"

    def __init__(self, dim: int) -> None:
        # See (Kress LIE, Thm 6.2) for scaling
        if dim == 2:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("log")(r)
            scaling = 1/(-2*var("pi"))
        elif dim == 3:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = 1/r
            scaling = 1/(4*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @override
    def __str__(self) -> str:
        return f"LapKnl{self.dim}D"

    @override
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        from sumpy.derivative_taker import LaplaceDerivativeTaker
        return LaplaceDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


class BiharmonicKernel(ExpressionKernel):
    mapper_method: ClassVar[str] = "map_biharmonic_kernel"

    def __init__(self, dim: int) -> None:
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        if dim == 2:
            # Ref: Farkas, Peter. Mathematical foundations for fast algorithms
            # for the biharmonic equation. Technical Report 765,
            # Department of Computer Science, Yale University, 1990.
            expr = r**2 * var("log")(r)
            scaling = 1/(8*var("pi"))
        elif dim == 3:
            # Ref: Jiang, Shidong, Bo Ren, Paul Tsuji, and Lexing Ying.
            # "Second kind integral equations for the first kind Dirichlet problem
            #  of the biharmonic equation in three dimensions."
            # Journal of Computational Physics 230, no. 19 (2011): 7488-7501.
            expr = r
            scaling = -1/(8*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @override
    def __str__(self) -> str:
        return f"BiharmKnl{self.dim}D"

    @override
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        from sumpy.derivative_taker import RadialDerivativeTaker
        return RadialDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


@dataclass(frozen=True, repr=False)
class HelmholtzKernel(ExpressionKernel):
    """
    .. autoattribute:: helmholtz_k_name
    .. autoattribute:: allow_evanescent
    """

    mapper_method: ClassVar[str] = "map_helmholtz_kernel"

    helmholtz_k_name: str
    """The argument name to use for the Helmholtz parameter when generating
    functions to evaluate this kernel.
    """
    allow_evanescent: bool

    def __init__(self,
                 dim: int,
                 helmholtz_k_name: str = "k",
                 allow_evanescent: bool = False) -> None:
        k = SpatialConstant(helmholtz_k_name)

        # Guard against code using the old positional interface.
        assert isinstance(allow_evanescent, bool)

        if dim == 2:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("hankel_1")(0, k*r)
            scaling = var("I")/4
        elif dim == 3:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("exp")(var("I")*k*r)/r
            scaling = 1/(4*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "helmholtz_k_name", helmholtz_k_name)
        object.__setattr__(self, "allow_evanescent", allow_evanescent)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return True

    @override
    def __str__(self) -> str:
        return f"HelmKnl{self.dim}D({self.helmholtz_k_name})"

    @override
    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        from sumpy.codegen import register_bessel_callables
        return register_bessel_callables(loopy_knl)

    @override
    def get_args(self) -> Sequence[KernelArgument]:
        k_dtype = np.complex128 if self.allow_evanescent else np.float64
        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.helmholtz_k_name, k_dtype),
                    )]

    @override
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        from sumpy.derivative_taker import HelmholtzDerivativeTaker
        return HelmholtzDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        k = sym.Symbol(self.helmholtz_k_name)
        return laplacian(w) + k**2 * w


@dataclass(frozen=True, repr=False)
class YukawaKernel(ExpressionKernel):
    """
    .. autoattribute:: yukawa_lambda_name
    """

    mapper_method: ClassVar[str] = "map_yukawa_kernel"

    yukawa_lambda_name: str
    """The argument name to use for the Yukawa parameter when generating
    functions to evaluate this kernel.
    """

    def __init__(self, dim: int, yukawa_lambda_name: str = "lam") -> None:
        lam = SpatialConstant(yukawa_lambda_name)

        # NOTE: The Yukawa kernel is given by [1]
        #   -1/(2 pi)**(n/2) * (lam/r)**(n/2-1) * K(n/2-1, lam r)
        # where K is a modified Bessel function of the second kind.
        #
        # [1] https://en.wikipedia.org/wiki/Green%27s_function
        # [2] https://dlmf.nist.gov/10.27#E8
        # [3] https://dlmf.nist.gov/10.47#E2
        # [4] https://dlmf.nist.gov/10.49

        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        if dim == 2:
            # NOTE: transform K(0, lam r) into a Hankel function using [2]
            expr = var("hankel_1")(0, var("I")*lam*r)
            scaling_for_K0 = var("pi")/2*var("I")       # noqa: N806

            scaling = 1/(2*var("pi")) * scaling_for_K0
        elif dim == 3:
            # NOTE: to get the expression, we do the following and simplify
            # 1. express K(1/2, lam r) as a modified spherical Bessel function
            #   k(0, lam r) using [3] and use expression for k(0, lam r) from [4]
            # 2. or use (AS 10.2.17) directly
            expr = var("exp")(-lam*r) / r

            scaling = 1/(4 * var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)
        object.__setattr__(self, "yukawa_lambda_name", yukawa_lambda_name)

    @property
    @override
    def is_complex_valued(self) -> bool:
        # FIXME
        return True

    @override
    def __str__(self) -> str:
        return f"YukKnl{self.dim}D({self.yukawa_lambda_name})"

    @override
    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        from sumpy.codegen import register_bessel_callables
        return register_bessel_callables(loopy_knl)

    @override
    def get_args(self) -> Sequence[KernelArgument]:
        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.yukawa_lambda_name, np.float64),
                    )]

    @override
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        from sumpy.derivative_taker import HelmholtzDerivativeTaker
        return HelmholtzDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        lam = sym.Symbol(self.yukawa_lambda_name)
        return laplacian(w) - lam**2 * w


@dataclass(frozen=True, repr=False)
class ElasticityKernel(ExpressionKernel):
    """
    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    .. autoattribute:: viscosity_mu
    .. autoattribute:: poisson_ratio
    """
    mapper_method: ClassVar[str] = "map_elasticity_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""

    viscosity_mu: float | SpatialConstant
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel. Can also be a numeric value.
    """
    poisson_ratio: float | SpatialConstant
    r"""The argument name to use for Poisson's ratio :math:`\nu` when generating
    functions to evaluate this kernel. Can also be a numeric value.
    """

    def __new__(cls,
                dim: int, icomp: int, jcomp: int,
                viscosity_mu: float | str | SpatialConstant = "mu",
                poisson_ratio: float | str | SpatialConstant = "nu",
        ) -> ElasticityKernel:
        if poisson_ratio == 0.5:
            return super().__new__(StokesletKernel)
        else:
            return super().__new__(cls)

    def __init__(self,
                 dim: int, icomp: int, jcomp: int,
                 viscosity_mu: float | str | SpatialConstant = "mu",
                 poisson_ratio: float | str | SpatialConstant = "nu") -> None:
        if isinstance(viscosity_mu, str):
            mu = SpatialConstant(viscosity_mu)
        else:
            mu = viscosity_mu

        if isinstance(poisson_ratio, str):
            nu = SpatialConstant(poisson_ratio)
        else:
            nu = poisson_ratio

        if dim == 2:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            # See (Berger and Karageorghis 2001)
            expr = (
                -var("log")(r)*((3 - 4 * nu) if icomp == jcomp else 0)
                +
                d[icomp]*d[jcomp]/r**2
                )
            scaling = -1/(8*var("pi")*(1 - nu)*mu)

        elif dim == 3:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            # Kelvin solution
            expr = (
                (1/r)*((3 - 4*nu) if icomp == jcomp else 0)
                +
                d[icomp]*d[jcomp]/r**3
                )
            scaling = -1/(16*var("pi")*(1 - nu)*mu)

        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)
        object.__setattr__(self, "viscosity_mu", mu)
        object.__setattr__(self, "poisson_ratio", nu)

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.icomp, self.jcomp, self.viscosity_mu, self.poisson_ratio))

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @override
    def __str__(self) -> str:
        return (
            f"ElasticityKnl{self.dim}D_{self.icomp}{self.jcomp}"
            f"({self.viscosity_mu}, {self.poisson_ratio})")

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.viscosity_mu)

        args: list[KernelArgument] = []
        for v in variables:
            args.append(KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64)))

        return [*args, *self.get_source_args()]

    @memoize_method
    @override
    def get_source_args(self) -> Sequence[KernelArgument]:
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.poisson_ratio)

        args: list[KernelArgument] = []
        for v in variables:
            args.append(KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64)))

        return args

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


@dataclass(frozen=True, repr=False)
class StokesletKernel(ElasticityKernel):
    """
    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    .. autoattribute:: viscosity_mu
    """

    def __new__(cls,
                dim: int,
                icomp: int,
                jcomp: int,
                viscosity_mu: float | str | SpatialConstant = "mu",
                poisson_ratio: float | str | SpatialConstant | None = None,
            ) -> StokesletKernel:
        return object.__new__(cls)

    def __init__(self,
                dim: int,
                icomp: int,
                jcomp: int,
                viscosity_mu: float | str | SpatialConstant = "mu",
                poisson_ratio: float | str | SpatialConstant | None = None) -> None:
        if poisson_ratio is None:
            poisson_ratio = 0.5

        if poisson_ratio != 0.5:
            raise ValueError(
                "'StokesletKernel' must have a Poisson ratio of 0.5: "
                f"got '{poisson_ratio}'")

        super().__init__(dim, icomp, jcomp, viscosity_mu, poisson_ratio)

    @override
    def __str__(self) -> str:
        return (
            f"StokesletKnl{self.dim}D_{self.icomp}{self.jcomp}"
            f"({self.viscosity_mu}, {self.poisson_ratio})")


@dataclass(frozen=True, repr=False)
class StressletKernel(ExpressionKernel):
    """
    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    .. autoattribute:: kcomp
    .. autoattribute:: viscosity_mu
    """
    mapper_method: ClassVar[str] = "map_stresslet_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""
    kcomp: int
    """Component index for the kernel."""
    viscosity_mu: float | SpatialConstant
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel. Can also be a numeric value.
    """

    def __init__(self,
                 dim: int, icomp: int, jcomp: int, kcomp: int,
                 viscosity_mu: float | str | SpatialConstant = "mu") -> None:
        # mu is unused but kept for consistency with the Stokeslet.
        if isinstance(viscosity_mu, str):
            mu = SpatialConstant(viscosity_mu)
        else:
            mu = viscosity_mu

        if dim == 2:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            expr = (
                d[icomp]*d[jcomp]*d[kcomp]/r**4
                )
            scaling = 1/(var("pi"))

        elif dim == 3:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            expr = (
                d[icomp]*d[jcomp]*d[kcomp]/r**5
                )
            scaling = 3/(4*var("pi"))

        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)
        object.__setattr__(self, "kcomp", kcomp)
        object.__setattr__(self, "viscosity_mu", mu)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @override
    def __str__(self) -> str:
        return (
            f"StressletKnl{self.dim}D_{self.icomp}{self.jcomp}{self.kcomp}"
            f"({self.viscosity_mu})")

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.viscosity_mu)
        return [
                KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64))
                for v in variables]

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


@dataclass(frozen=True, repr=False)
class LineOfCompressionKernel(ExpressionKernel):
    """A kernel for the line of compression or dilatation of constant strength
    along the axis "axis" from zero to negative infinity.

    This is used for the explicit solution to half-space Elasticity problem.
    See [Mindlin1936]_ for details.

    .. [Mindlin1936] R. D. Mindlin (1936).
         *Force at a Point in the Interior of a Semi-Infinite Solid*.
         Physics. 7 (5): 195-202.
         `doi:10.1063/1.1745385 <https://doi.org/10.1063/1.1745385>`__.

    .. autoattribute:: axis
    .. autoattribute:: viscosity_mu
    .. autoattribute:: poisson_ratio
    """

    mapper_method: ClassVar[str] = "map_line_of_compression_kernel"

    axis: int
    """Axis number (defaulting to 2 for the z axis)."""
    viscosity_mu: float | SpatialConstant
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel. Can also be a numeric value.
    """
    poisson_ratio: float | SpatialConstant
    r"""The argument name to use for Poisson's ratio :math:`\nu` when
    generating functions to evaluate this kernel. Can also be a numeric value.
    """

    def __init__(self,
                 dim: int = 3,
                 axis: int = 2,
                 viscosity_mu: float | str | SpatialConstant = "mu",
                 poisson_ratio: float | str | SpatialConstant = "nu"
             ) -> None:
        if isinstance(viscosity_mu, str):
            mu = SpatialConstant(viscosity_mu)
        else:
            mu = viscosity_mu
        if isinstance(poisson_ratio, str):
            nu = SpatialConstant(poisson_ratio)
        else:
            nu = poisson_ratio

        if dim == 3:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            # Kelvin solution
            expr = d[axis] * var("log")(r + d[axis]) - r
            scaling = (1 - 2*nu)/(4*var("pi")*mu)
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "viscosity_mu", mu)
        object.__setattr__(self, "poisson_ratio", nu)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @override
    def __str__(self) -> str:
        return f"LineOfCompressionKnl{self.dim}D_{self.axis}"

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        from sumpy.tools import get_all_variables
        variables = [
            *get_all_variables(self.viscosity_mu),
            *get_all_variables(self.poisson_ratio)]

        args: list[KernelArgument] = []
        for v in variables:
            args.append(KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64)))

        return args

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


# }}}


# {{{ a kernel defined as wrapping another one--e.g., derivatives

@dataclass(frozen=True)
class KernelWrapper(Kernel, ABC):
    inner_kernel: Kernel
    """The kernel that is being wrapped (to take a derivative of, etc.)."""

    def __init__(self, inner_kernel: Kernel) -> None:
        Kernel.__init__(self, inner_kernel.dim)
        object.__setattr__(self, "inner_kernel", inner_kernel)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return self.inner_kernel.is_complex_valued

    @override
    def get_base_kernel(self) -> Kernel:
        return self.inner_kernel.get_base_kernel()

    @override
    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        return self.inner_kernel.prepare_loopy_kernel(loopy_knl)

    @override
    def get_expression(self, dist_vec: sp.Matrix) -> ArithmeticExpr:
        return self.inner_kernel.get_expression(dist_vec)

    @override
    def get_derivative_coeff_dict_at_source(
            self, expr_dict: DerivativeCoeffDict,
        ) -> DerivativeCoeffDict:
        return self.inner_kernel.get_derivative_coeff_dict_at_source(expr_dict)

    @overload
    def postprocess_at_target(
            self, expr: sym.Expr, bvec: sp.Matrix,
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_target(
            self, expr: ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> DifferentiatedExprDerivativeTaker: ...

    @override
    def postprocess_at_target(
            self, expr: sym.Expr | ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> sym.Expr | DifferentiatedExprDerivativeTaker:
        return self.inner_kernel.postprocess_at_target(expr, bvec)

    @override
    def get_global_scaling_const(self) -> ArithmeticExpr:
        return self.inner_kernel.get_global_scaling_const()

    @override
    def get_code_transformer(self) -> Callable[[Expression], Expression]:
        return self.inner_kernel.get_code_transformer()

    @override
    def get_args(self) -> Sequence[KernelArgument]:
        return self.inner_kernel.get_args()

    @override
    def get_source_args(self) -> Sequence[KernelArgument]:
        return self.inner_kernel.get_source_args()

    @override
    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        raise NotImplementedError(
            f"'replace_base_kernel' is not implemented for '{type(self).__name__}'")

    @override
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: ArithmeticExpr,
            sac: SymbolicAssignmentCollection,
        ) -> ExprDerivativeTaker:
        return self.inner_kernel.get_derivative_taker(dvec, rscale, sac)

# }}}


# {{{ derivatives

class DerivativeBase(KernelWrapper, ABC):
    """Bases: :class:`Kernel`

    .. autoattribute:: inner_kernel
    .. automethod:: replace_inner_kernel
    """

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        return self.inner_kernel.get_pde_as_diff_op()

    @abstractmethod
    def replace_inner_kernel(self, new_inner_kernel: Kernel) -> Kernel:
        """Replace the inner kernel of this wrapper.

        This is essentially the same as :meth:`Kernel.replace_base_kernel`, but it does
        not recurse.
        """


@dataclass(frozen=True)
class AxisSourceDerivative(DerivativeBase):
    """
    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_axis_source_derivative"

    axis: int
    """Direction axis for the source derivative."""

    def __init__(self, axis: int, inner_kernel: Kernel) -> None:
        super().__init__(inner_kernel)
        object.__setattr__(self, "axis", axis)

    @override
    def __str__(self) -> str:
        return f"d/dy{self.axis} {self.inner_kernel}"

    @override
    def get_derivative_coeff_dict_at_source(
            self, expr_dict: DerivativeCoeffDict,
        ) -> DerivativeCoeffDict:
        expr_dict = self.inner_kernel.get_derivative_coeff_dict_at_source(
            expr_dict)
        result = {}
        for mi, coeff in expr_dict.items():
            new_mi = list(mi)
            new_mi[self.axis] += 1
            result[tuple(new_mi)] = -coeff
        return result

    @override
    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    @override
    def replace_inner_kernel(self, new_inner_kernel: Kernel) -> Kernel:
        return type(self)(self.axis, new_inner_kernel)


@dataclass(frozen=True)
class AxisTargetDerivative(DerivativeBase):
    """
    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_axis_target_derivative"
    target_array_name: ClassVar[str] = "targets"

    axis: int

    def __init__(self, axis: int, inner_kernel: Kernel) -> None:
        super().__init__(inner_kernel)
        object.__setattr__(self, "axis", axis)

    @override
    def __str__(self) -> str:
        return f"d/dx{self.axis} {self.inner_kernel}"

    @overload
    def postprocess_at_target(
            self, expr: sym.Expr, bvec: sp.Matrix,
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_target(
            self, expr: ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> DifferentiatedExprDerivativeTaker: ...

    @override
    def postprocess_at_target(
            self, expr: sym.Expr | ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> sym.Expr | DifferentiatedExprDerivativeTaker:
        target_vec = sym.make_sym_vector(self.target_array_name, self.dim)

        # bvec = tgt - ctr
        inner_expr = self.inner_kernel.postprocess_at_target(expr, bvec)
        if isinstance(inner_expr, DifferentiatedExprDerivativeTaker):
            transformation = diff_derivative_coeff_dict(
                    inner_expr.derivative_coeff_dict,
                    self.axis, target_vec)
            return DifferentiatedExprDerivativeTaker(inner_expr.taker, transformation)
        else:
            # Since `bvec` and `tgt` are two different symbolic variables
            # need to differentiate by both to get the correct answer
            return (inner_expr.diff(bvec[self.axis])
                    + inner_expr.diff(target_vec[self.axis]))

    @override
    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    @override
    def replace_inner_kernel(self, new_inner_kernel: Kernel) -> Kernel:
        return type(self)(self.axis, new_inner_kernel)


class _VectorIndexAdder(CSECachingMapperMixin[Expression, []], IdentityMapper[[]]):
    vec_name: str
    additional_indices: tuple[Expression, ...]

    def __init__(self,
                 vec_name: str,
                 additional_indices: tuple[Expression, ...]) -> None:
        self.vec_name = vec_name
        self.additional_indices = additional_indices

    @override
    def map_subscript(self, expr: prim.Subscript) -> Expression:
        from pymbolic.primitives import CommonSubexpression, cse_scope
        name = getattr(expr.aggregate, "name", None)

        if name == self.vec_name and isinstance(expr.index, int):
            return CommonSubexpression(
                    expr.aggregate[(expr.index, *self.additional_indices)],
                    prefix=None, scope=cse_scope.EVALUATION)
        else:
            return IdentityMapper.map_subscript(self, expr)

    @override
    def map_common_subexpression_uncached(self,
                expr: prim.CommonSubexpression) -> Expression:
        result = self.rec(expr.child)
        if result is expr.child:
            return expr

        return type(expr)(
                      result, expr.prefix, expr.scope, **expr.get_extra_properties())


@dataclass(frozen=True)
class DirectionalDerivative(DerivativeBase):
    """
    .. autoattribute:: dir_vec_name
    """
    directional_kind: ClassVar[Literal["src", "tgt"]]
    """The kind of this directional derivative (can only be a source or target)."""

    dir_vec_name: str
    """Name of the vector used for the direction."""

    def __init__(self, inner_kernel: Kernel, dir_vec_name: str | None = None) -> None:
        if dir_vec_name is None:
            dir_vec_name = f"{self.directional_kind}_derivative_dir"

        KernelWrapper.__init__(self, inner_kernel)
        object.__setattr__(self, "dir_vec_name", dir_vec_name)

    @override
    def __str__(self) -> str:
        d = "y" if self.directional_kind == "src" else "x"
        return fr"{self.dir_vec_name}·∇_{d} {self.inner_kernel}"

    @override
    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        return type(self)(
            self.inner_kernel.replace_base_kernel(new_base_kernel),
            dir_vec_name=self.dir_vec_name)

    @override
    def replace_inner_kernel(self, new_inner_kernel: Kernel) -> Kernel:
        return type(self)(new_inner_kernel, dir_vec_name=self.dir_vec_name)


class DirectionalSourceDerivative(DirectionalDerivative):
    mapper_method: ClassVar[str] = "map_directional_source_derivative"
    directional_kind: ClassVar[Literal["src", "tgt"]] = "src"

    @override
    def get_code_transformer(self) -> Callable[[Expression], Expression]:
        inner = self.inner_kernel.get_code_transformer()
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter(frozenset([self.dir_vec_name]))
        via = _VectorIndexAdder(self.dir_vec_name, (prim.Variable("isrc"),))

        def transform(expr: Expression) -> Expression:
            return via(vcr(inner(expr)))

        return transform

    @override
    def get_derivative_coeff_dict_at_source(
            self, expr_dict: DerivativeCoeffDict,
        ) -> DerivativeCoeffDict:
        from sumpy.symbolic import make_sym_vector as make_sympy_vector
        dir_vec = make_sympy_vector(self.dir_vec_name, self.dim)

        expr_dict = self.inner_kernel.get_derivative_coeff_dict_at_source(
            expr_dict)

        # avec = center-src -> minus sign from chain rule
        result: DerivativeCoeffDict = defaultdict(lambda: 0)
        for mi, coeff in expr_dict.items():
            for axis in range(self.dim):
                new_mi = list(mi)
                new_mi[axis] += 1
                result[tuple(new_mi)] += -coeff * dir_vec[axis]

        return dict(result)

    @override
    def get_source_args(self) -> Sequence[KernelArgument]:
        return [
                KernelArgument(
                    loopy_arg=lp.GlobalArg(
                        self.dir_vec_name,
                        None,
                        shape=(self.dim, "nsources"),
                        offset=lp.auto),
                    ),
                    *self.inner_kernel.get_source_args()]

    @override
    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        loopy_knl = self.inner_kernel.prepare_loopy_kernel(loopy_knl)
        return lp.tag_array_axes(loopy_knl, self.dir_vec_name, "sep,C")


class TargetPointMultiplier(KernelWrapper):
    """Bases: :class:`Kernel`

    Wraps a kernel :math:`G(x, y)` and outputs :math:`x_j G(x, y)`
    where :math:`x, y` are targets and sources respectively.

    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_target_point_multiplier"
    target_array_name: ClassVar[str] = "targets"

    axis: int
    """Coordinate axis with which to multiply the kernel."""

    def __init__(self, axis: int, inner_kernel: Kernel) -> None:
        KernelWrapper.__init__(self, inner_kernel)
        self.axis = axis

    @override
    def __str__(self) -> str:
        return f"x{self.axis} {self.inner_kernel}"

    @overload
    def postprocess_at_target(
            self, expr: sym.Expr, bvec: sp.Matrix,
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_target(
            self, expr: ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> DifferentiatedExprDerivativeTaker: ...

    @override
    def postprocess_at_target(
            self, expr: sym.Expr | ExprDerivativeTaker, bvec: sp.Matrix,
        ) -> sym.Expr | DifferentiatedExprDerivativeTaker:
        inner_expr = self.inner_kernel.postprocess_at_target(expr, bvec)
        target_vec = sym.make_sym_vector(self.target_array_name, self.dim)

        zeros = tuple([0]*self.dim)
        mult = cast("sym.Symbol", target_vec[self.axis])

        if isinstance(inner_expr, DifferentiatedExprDerivativeTaker):
            transform: DerivativeCoeffDict = {
                mi: coeff * mult for mi, coeff in
                inner_expr.derivative_coeff_dict.items()}

            return DifferentiatedExprDerivativeTaker(inner_expr.taker, transform)
        elif isinstance(inner_expr, ExprDerivativeTaker):
            return DifferentiatedExprDerivativeTaker(expr.orig_expr, {zeros: mult})
        else:
            return mult * inner_expr

    @override
    def get_code_transformer(self) -> Callable[[Expression], Expression]:
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter(frozenset([self.target_array_name]))
        via = _VectorIndexAdder(self.target_array_name, (prim.Variable("itgt"),))

        inner_transform = self.inner_kernel.get_code_transformer()

        def transform(expr: Expression) -> Expression:
            return via(vcr(inner_transform(expr)))

        return transform

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        raise NotImplementedError("no PDE is known")

    @override
    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    def replace_inner_kernel(self, new_inner_kernel: Kernel) -> Kernel:
        return type(self)(self.axis, new_inner_kernel)

# }}}


# {{{ kernel mappers

ResultT = TypeVar("ResultT")


class KernelMapper(Generic[ResultT]):
    """
    .. automethod:: __call__
    """
    def rec(self, kernel: Kernel) -> ResultT:
        try:
            method = cast(
                "Callable[[Kernel], ResultT]",
                getattr(self, kernel.mapper_method))
        except AttributeError as err:
            raise RuntimeError(f"{type(self)} cannot handle {type(kernel)}") from err
        else:
            return method(kernel)

    def __call__(self, kernel: Kernel) -> ResultT:
        return self.rec(kernel)


class KernelCombineMapper(KernelMapper[ResultT], ABC):
    """
    .. automethod:: combine
    """

    @abstractmethod
    def combine(self, values: Iterable[ResultT]) -> ResultT:
        raise NotImplementedError

    def map_axis_target_derivative(
            self, kernel: AxisTargetDerivative) -> ResultT:
        return self.rec(kernel.inner_kernel)

    def map_axis_source_derivative(
            self, kernel: AxisSourceDerivative) -> ResultT:
        return self.rec(kernel.inner_kernel)

    def map_directional_source_derivative(
            self, kernel: DirectionalSourceDerivative) -> ResultT:
        return self.rec(kernel.inner_kernel)

    def map_target_point_multiplier(
            self, kernel: TargetPointMultiplier) -> ResultT:
        return self.rec(kernel.inner_kernel)


class KernelIdentityMapper(KernelMapper[Kernel]):
    def map_expression_kernel(self, kernel: ExpressionKernel) -> Kernel:
        return kernel

    map_laplace_kernel = map_expression_kernel
    map_biharmonic_kernel = map_expression_kernel
    map_helmholtz_kernel = map_expression_kernel
    map_yukawa_kernel = map_expression_kernel
    map_elasticity_kernel = map_expression_kernel
    map_line_of_compression_kernel = map_expression_kernel
    map_stresslet_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel: AxisTargetDerivative) -> Kernel:
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    def map_axis_source_derivative(self, kernel: AxisSourceDerivative) -> Kernel:
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    def map_target_point_multiplier(self, kernel: TargetPointMultiplier) -> Kernel:
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    def map_directional_source_derivative(
            self, kernel: DirectionalSourceDerivative) -> Kernel:
        return type(kernel)(self.rec(kernel.inner_kernel),
                            dir_vec_name=kernel.dir_vec_name)


class AxisSourceDerivativeRemover(KernelIdentityMapper):
    """Removes all axis source derivatives from the kernel."""

    @override
    def map_axis_source_derivative(self, kernel: AxisSourceDerivative) -> Kernel:
        return self.rec(kernel.inner_kernel)


class AxisTargetDerivativeRemover(KernelIdentityMapper):
    """Removes all axis target derivatives from the kernel."""

    @override
    def map_axis_target_derivative(self, kernel: AxisTargetDerivative) -> Kernel:
        return self.rec(kernel.inner_kernel)


class TargetDerivativeRemover(AxisTargetDerivativeRemover):
    """Removes all target derivatives from the kernel."""


class SourceDerivativeRemover(AxisSourceDerivativeRemover):
    """Removes all source derivatives from the kernel."""

    @override
    def map_directional_source_derivative(
            self, kernel: DirectionalSourceDerivative) -> Kernel:
        return self.rec(kernel.inner_kernel)


class TargetTransformationRemover(TargetDerivativeRemover):
    """Removes all target transformations from the kernel."""

    @override
    def map_target_point_multiplier(self, kernel: TargetPointMultiplier) -> Kernel:
        return self.rec(kernel.inner_kernel)


SourceTransformationRemover = SourceDerivativeRemover


class DerivativeCounter(KernelCombineMapper[int]):
    """Counts the number of derivatives in the kernel."""

    @override
    def combine(self, values: Iterable[int]) -> int:
        return sum(values)

    def map_expression_kernel(self, kernel: ExpressionKernel) -> int:
        return 0

    map_laplace_kernel = map_expression_kernel
    map_biharmonic_kernel = map_expression_kernel
    map_helmholtz_kernel = map_expression_kernel
    map_yukawa_kernel = map_expression_kernel
    map_line_of_compression_kernel = map_expression_kernel
    map_stresslet_kernel = map_expression_kernel

    @override
    def map_axis_target_derivative(self, kernel: AxisTargetDerivative) -> int:
        return self.combine([1, self.rec(kernel.inner_kernel)])

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative
    map_axis_source_derivative = map_axis_target_derivative

# }}}


# vim: fdm=marker
