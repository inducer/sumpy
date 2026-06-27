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
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from typing_extensions import Self, override

import loopy as lp
import pymbolic.primitives as prim
from pymbolic import Expression, var
from pymbolic.mapper import CSECachingMapperMixin, IdentityMapper
from pymbolic.primitives import make_sym_vector
from pytools import keyed_memoize_method, memoize_method, obj_array

import sumpy.symbolic as sym
from sumpy.derivative_taker import (
    DerivativeCoeffDict,
    DifferentiatedExprDerivativeTaker,
    ExprDerivativeTaker,
    diff_derivative_coeff_dict,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import sympy as sp

    from sumpy.assignment_collection import SymbolicAssignmentCollection
    from sumpy.expansion.diff_op import LinearPDESystemOperator

__doc__ = """
Kernel interface
----------------

.. autoclass:: KernelArgument
.. autoclass:: ScalarKernel
    :show-inheritance:
.. autoclass:: SystemKernel
    :show-inheritance:

Symbolic kernels
----------------

.. autoclass:: ExpressionKernel
    :show-inheritance:
    :members: mapper_method

.. autoclass:: OneKernel
    :show-inheritance:
    :members: mapper_method

Scalar PDE kernels
------------------

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

.. autoclass:: StokesComponentKernelBase
    :show-inheritance:
.. autoclass:: StokesletComponentKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: StressletComponentKernel
    :show-inheritance:
    :members: mapper_method

.. autoclass:: ElasticityComponentKernelBase
    :show-inheritance:
.. autoclass:: ElasticityComponentKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: ElasticityStressComponentKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: LineOfCompressionKernel
    :show-inheritance:
    :members: mapper_method

.. autoclass:: BrinkmanComponentKernelBase
    :show-inheritance:
.. autoclass:: BrinkmanletComponentKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: BrinkmanStressComponentKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: HeatKernel
    :show-inheritance:
    :members: mapper_method

System PDE Kernels
------------------

.. autoclass:: ElasticitySystemKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: ElasticityStressSystemKernel
    :show-inheritance:
    :members: mapper_method

.. autoclass:: StokesletSystemKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: StressletSystemKernel
    :show-inheritance:
    :members: mapper_method

.. autoclass:: BrinkmanletSystemKernel
    :show-inheritance:
    :members: mapper_method
.. autoclass:: BrinkmanStressSystemKernel
    :show-inheritance:
    :members: mapper_method

.. [Pozrikidis1992] C. Pozrikidis,
    *Boundary Integral and Singularity Methods for Linearized Viscous Flow*,
    Cambridge University Press, 1992.

.. [Hsiao2008] G. C. Hsiao, W. L. Wendland,
    *Boundary Integral Equations*,
    Springer, 2008.

.. [Kress2013] R. Kress,
    *Linear Integral Equations*,
    Springer Science & Business Media, 2013.

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


def _diff(expr: sym.Expr, vec: sp.Matrix, mi: tuple[int, ...]) -> sym.Expr:
    """Take the derivative of an expression."""
    dim = len(mi)
    assert vec.shape == (dim, 1)

    for i in range(dim):
        if mi[i] == 0:
            continue
        expr = expr.diff(vec[i], mi[i])

    return expr


@dataclass(frozen=True, repr=False)
class ScalarKernel(ABC):
    """Scalar kernel interface.

    .. autoattribute:: mapper_method
    .. autoattribute:: is_translation_invariant

    .. autoattribute:: dim
    .. autoproperty:: is_complex_valued

    .. automethod:: get_base_kernel
    .. automethod:: replace_base_kernel
    .. automethod:: get_pde_system_kernel
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

    def get_base_kernel(self) -> ScalarKernel:
        """
        :returns: the kernel being wrapped by this one, or else *self*.
        """
        return self

    def replace_base_kernel(self, new_base_kernel: ScalarKernel) -> ScalarKernel:
        """
        :returns: the base kernel being wrapped by this one, or else
            *new_base_kernel*.
        """
        return new_base_kernel

    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        """
        :returns: if the kernel is a component kernel of a :class:`SystemKernel`,
            this returns the system kernel and the index of the kernel in that
            system.
        """
        raise TypeError(f"kernel {type(self)} is not part of a system")

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
    def get_expression(self, dist_vec: sym.Matrix) -> sym.Expr:
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

    @abstractmethod
    def get_derivative_taker(
            self,
            dvec: sp.Matrix,
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
        ) -> ExprDerivativeTaker:
        """
        :returns: an :class:`~sumpy.derivative_taker.ExprDerivativeTaker` instance
            that supports taking derivatives of the base kernel with respect to
            *dvec*.
        """

    @overload
    def postprocess_at_source(
            self, expr: sym.Expr, avec: sym.Matrix
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_source(
            self, expr: ExprDerivativeTaker, avec: sym.Matrix
        ) -> DifferentiatedExprDerivativeTaker: ...

    def postprocess_at_source(
            self, expr: sym.Expr | ExprDerivativeTaker, avec: sym.Matrix,
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
            result += coeff * _diff(expr, avec, mi)

        assert isinstance(result, sym.Expr)
        return result

    @overload
    def postprocess_at_target(
            self, expr: sym.Expr, bvec: sym.Matrix,
        ) -> sym.Expr: ...

    @overload
    def postprocess_at_target(
            self, expr: ExprDerivativeTaker | DifferentiatedExprDerivativeTaker,
            bvec: sym.Matrix,
        ) -> DifferentiatedExprDerivativeTaker: ...

    def postprocess_at_target(self,
                expr:
                    sym.Expr | ExprDerivativeTaker | DifferentiatedExprDerivativeTaker,
                bvec: sym.Matrix,
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
    def get_global_scaling_const(self) -> sym.Expr:
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


@dataclass(frozen=True, repr=False)
class SystemKernel(ABC):
    """A kernel representing a vector PDE.

    .. autoattribute:: mapper_method

    .. autoattribute:: dim
    .. autoproperty:: ndim
    .. autoproperty:: shape

    .. automethod:: __getitem__
    .. automethod:: get_expression
    .. automethod:: get_pde_as_diff_op
    """

    mapper_method: ClassVar[str]
    """The name of the mapper method called for the kernel."""

    dim: int
    """Dimension of the space the kernel is defined in."""

    @property
    def ndim(self) -> int:
        """The number of indices in the kernel tensor."""
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the kernel tensor."""

    def __getitem__(self, index: tuple[int, ...], /) -> ScalarKernel:
        """
        :returns: the scalar kernel at *index*.
        """
        if len(index) != self.ndim:
            raise IndexError(
                f"incorrect index size for kernel: kernel is {self.ndim}-dimensional: "
                f"{index} given"
            )

        if any(not 0 <= i < n for i, n in zip(index, self.shape, strict=True)):
            raise IndexError(
                f"index {index} is out of bounds for kernel with shape {self.shape}"
            )

        return self.get_scalar_component(*index)

    @abstractmethod
    def get_scalar_component(self, *args: int) -> ScalarKernel:
        """
        :returns: the scalar kernel the indices given by *args*.
        """

    def get_expression(self, dist_vec: sym.Matrix) -> obj_array.ObjectArrayND[sym.Expr]:
        """
        :returns: a :mod:`sympy` expression for each component the kernel.
        """
        from pytools import ndindex

        result = np.empty(self.shape, dtype=object)
        for i in ndindex(result.shape):
            result[i] = self[i].get_expression(dist_vec)

        return result

    @abstractmethod
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        """
        :returns: the PDE that this kernel satisfies
            (see :meth:`ScalarKernel.get_pde_as_diff_op` as the scalar alternative).
        """
        return []

# }}}


# {{{ generic expression kernel

@dataclass(frozen=True, repr=False)
class ExpressionKernel(ScalarKernel, ABC):
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
    def get_expression(self, dist_vec: sym.Matrix) -> sym.Expr:
        expr = sym.PymbolicToSympyMapperWithSymbols().to_expr(self.expression)

        if self.dim != len(dist_vec):
            raise ValueError(
                "'dist_vec' length does not match expected dimension: "
                f"kernel dim is '{self.dim}' and dist_vec has length '{len(dist_vec)}'")

        return expr.xreplace({
            sym.Symbol(f"d{i}"): dist_vec_i
            for i, dist_vec_i in enumerate(dist_vec)
            })

    @override
    def get_global_scaling_const(self) -> sym.Expr:
        return sym.PymbolicToSympyMapperWithSymbols().to_expr(self.global_scaling_const)

    @override
    def get_derivative_taker(
            self,
            dvec: sym.Matrix,
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
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

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.dim,))

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False


one_kernel_2d = OneKernel(2)
one_kernel_3d = OneKernel(3)

# }}}


# {{{ PDE kernels

class LaplaceKernel(ExpressionKernel):
    r"""A kernel for the Laplace equation (see e.g. Theorem 6.2 from [Kress2013]_).

    .. math::

        \Delta K(\mathbf{x}, \mathbf{y}) = \delta(\mathbf{x} - \mathbf{y}).
    """

    mapper_method: ClassVar[str] = "map_laplace_kernel"

    def __init__(self, dim: int) -> None:
        # See (Kress LIE, Thm 6.2) for scaling
        if dim == 2:
            r = sym.pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("log")(r)
            scaling = 1/(-2*var("pi"))
        elif dim == 3:
            r = sym.pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = 1/r
            scaling = 1/(4*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.dim,))

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
            dvec: sym.Matrix,
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
        ) -> ExprDerivativeTaker:
        from sumpy.derivative_taker import LaplaceDerivativeTaker
        return LaplaceDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


class BiharmonicKernel(ExpressionKernel):
    r"""A kernel for the biharmonic equation.

    .. math::

        \Delta^2 K(\mathbf{x}, \mathbf{y}) = \delta(\mathbf{x} - \mathbf{y}).
    """

    mapper_method: ClassVar[str] = "map_biharmonic_kernel"

    def __init__(self, dim: int) -> None:
        r = sym.pymbolic_real_norm_2(make_sym_vector("d", dim))
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

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.dim,))

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
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
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
    r"""A kernel for the Helmholtz equation (see e.g. Example 12.14 in [Kress2013]_).

    .. math::

        \Delta K(\mathbf{x}, \mathbf{y}) + k^2 K(\mathbf{x}, \mathbf{y})
            = \delta(\mathbf{x} - \mathbf{y}).

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
        k = sym.SpatialConstant(helmholtz_k_name)

        # Guard against code using the old positional interface.
        assert isinstance(allow_evanescent, bool)

        if dim == 2:
            r = sym.pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("hankel_1")(0, k*r)
            scaling = var("I")/4
        elif dim == 3:
            r = sym.pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("exp")(var("I")*k*r)/r
            scaling = 1/(4*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "helmholtz_k_name", helmholtz_k_name)
        object.__setattr__(self, "allow_evanescent", allow_evanescent)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (
            self.__class__,
            (self.dim, self.helmholtz_k_name, self.allow_evanescent),
        )

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
            KernelArgument(loopy_arg=lp.ValueArg(self.helmholtz_k_name, k_dtype)),
        ]

    @override
    def get_derivative_taker(
            self,
            dvec: sym.Matrix,
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
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
    r"""A kernel for the Yukawa equation.

    .. math::

        \Delta K(\mathbf{x}, \mathbf{y}) - \lambda^2 K(\mathbf{x}, \mathbf{y})
            = \delta(\mathbf{x} - \mathbf{y}).

    .. autoattribute:: yukawa_lambda_name
    """

    mapper_method: ClassVar[str] = "map_yukawa_kernel"

    yukawa_lambda_name: str
    """The argument name to use for the Yukawa parameter when generating
    functions to evaluate this kernel.
    """

    def __init__(self, dim: int, yukawa_lambda_name: str = "lam") -> None:
        lam = sym.SpatialConstant(yukawa_lambda_name)

        # NOTE: The Yukawa kernel is given by [1]
        #   -1/(2 pi)**(n/2) * (lam/r)**(n/2-1) * K(n/2-1, lam r)
        # where K is a modified Bessel function of the second kind.
        #
        # [1] https://en.wikipedia.org/wiki/Green%27s_function
        # [2] https://dlmf.nist.gov/10.27#E8
        # [3] https://dlmf.nist.gov/10.47#E2
        # [4] https://dlmf.nist.gov/10.49

        r = sym.pymbolic_real_norm_2(make_sym_vector("d", dim))
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

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.dim, self.yukawa_lambda_name))

    @property
    @override
    def is_complex_valued(self) -> bool:
        # FIXME: 2D uses Hankel functions (complex-valued); 3D uses real exp
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
            KernelArgument(loopy_arg=lp.ValueArg(self.yukawa_lambda_name, np.float64)),
        ]

    @override
    def get_derivative_taker(
            self,
            dvec: sym.Matrix,
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
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
class ElasticityComponentKernelBase(ExpressionKernel):
    r"""Base kernel class for the linear elasticity (Navier-Cauchy) equations
    (see e.g. Section 2.2 in [Hsiao2008]_).

    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: poisson_ratio_name
    """

    viscosity_mu_name: str
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """
    poisson_ratio_name: str
    r"""The argument name to use for Poisson's ratio :math:`\nu` when generating
    functions to evaluate this kernel.
    """

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        return [
            KernelArgument(loopy_arg=lp.ValueArg(self.viscosity_mu_name, np.float64)),
            KernelArgument(loopy_arg=lp.ValueArg(self.poisson_ratio_name, np.float64)),
        ]

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


@dataclass(frozen=True, repr=False)
class ElasticityComponentKernel(ElasticityComponentKernelBase):
    r"""The displacement kernel for the linear elasticity (Navier-Cauchy) equations
    (see e.g. Section 2.2 in [Hsiao2008]_).

    .. math::

        \mu \Delta K_{ij}(\mathbf{x}, \mathbf{y})
        + \frac{\mu}{1 - 2 \nu} \nabla (\nabla \cdot K_{ij}(\mathbf{x}, \mathbf{y}))
        = \delta_{ij} \delta(\mathbf{x} - \mathbf{y}).

    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    """

    mapper_method: ClassVar[str] = "map_elasticity_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""

    def __init__(self,
                 dim: int,
                 icomp: int,
                 jcomp: int,
                 viscosity_mu_name: str = "mu",
                 poisson_ratio_name: str = "nu") -> None:
        if not isinstance(viscosity_mu_name, str):
            raise TypeError(
                f"'viscosity_mu_name' is not a str: {type(viscosity_mu_name)}"
            )

        if not isinstance(poisson_ratio_name, str):
            raise TypeError(
                f"'poisson_ratio_name' is not a str: {type(poisson_ratio_name)}"
            )

        mu = sym.SpatialConstant(viscosity_mu_name)
        nu = sym.SpatialConstant(poisson_ratio_name)

        d = make_sym_vector("d", dim)
        r = sym.pymbolic_real_norm_2(d)
        delta_ij = 1 if icomp == jcomp else 0

        if dim == 2:
            # See (Berger and Karageorghis 2001)
            expr = -var("log")(r)*(3 - 4 * nu)*delta_ij + d[icomp]*d[jcomp]/r**2
            scaling = -1/(8*var("pi")*(1 - nu)*mu)
        elif dim == 3:
            # Kelvin solution
            expr = (1/r)*(3 - 4*nu)*delta_ij + d[icomp]*d[jcomp]/r**3
            scaling = -1/(16*var("pi")*(1 - nu)*mu)
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling,
                         viscosity_mu_name=viscosity_mu_name,
                         poisson_ratio_name=poisson_ratio_name)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)

    @override
    def __str__(self) -> str:
        return (
            f"ElasticityKnl{self.dim}D_{self.icomp}{self.jcomp}"
            f"({self.viscosity_mu_name}, {self.poisson_ratio_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.icomp, self.jcomp,
             self.viscosity_mu_name,
             self.poisson_ratio_name))

    @override
    @memoize_method
    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        return ElasticitySystemKernel(
            self.dim,
            viscosity_mu_name=self.viscosity_mu_name,
            poisson_ratio_name=self.poisson_ratio_name
        ), (self.icomp, self.jcomp)


@dataclass(frozen=True, repr=False)
class ElasticityStressComponentKernel(ElasticityComponentKernelBase):
    r"""The stress kernel for the linear elasticity (Navier-Cauchy) equations
    (see e.g. Section 2.2 in [Hsiao2008]_).

    .. math::

        K_{ijk}(\mathbf{x}, \mathbf{y}) =
            \lambda \partial_l K_{kl}(\mathbf{x}, \mathbf{y}) \delta_{ij}
            + \mu (\partial_j K_{ik}(\mathbf{x}, \mathbf{y})
                   + \partial_i K_{jk}(\mathbf{x}, \mathbf{y})),

    where the two-index :math:`K_{ij}` is the
    :class:`~sumpy.kernel.ElasticityComponentKernel`.

    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    .. autoattribute:: kcomp
    """

    mapper_method: ClassVar[str] = "map_elasticity_stress_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""
    kcomp: int
    """Component index for the kernel."""

    def __init__(self,
                 dim: int,
                 icomp: int,
                 jcomp: int,
                 kcomp: int,
                 viscosity_mu_name: str = "mu",
                 poisson_ratio_name: str = "nu") -> None:
        nu = sym.SpatialConstant(poisson_ratio_name)

        d = make_sym_vector("d", dim)
        r = sym.pymbolic_real_norm_2(d)
        delta_ij = 1 if icomp == jcomp else 0
        delta_ik = 1 if icomp == kcomp else 0
        delta_jk = 1 if jcomp == kcomp else 0

        if dim == 2:
            expr = (
                (1 - 2*nu) * (
                    d[icomp] / r**2 * delta_jk
                    + d[jcomp] / r**2 * delta_ik
                    - d[kcomp] / r**2 * delta_ij)
                + 3 * d[icomp] * d[jcomp] * d[kcomp] / r**4
            )
            scaling = -1/(4*var("pi")*(1 - nu))
        elif dim == 3:
            expr = (
                (1 - 2*nu) * (
                    d[icomp] / r**3 * delta_jk
                    + d[jcomp] / r**3 * delta_ik
                    - d[kcomp] / r**3 * delta_ij)
                + 3 * d[icomp] * d[jcomp] * d[kcomp] / r**5
            )
            scaling = -1/(8*var("pi")*(1 - nu))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling,
                         viscosity_mu_name=viscosity_mu_name,
                         poisson_ratio_name=poisson_ratio_name)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)
        object.__setattr__(self, "kcomp", kcomp)

    @override
    def __str__(self) -> str:
        return (
            f"ElasticityStressKnl{self.dim}D_{self.icomp}{self.jcomp}{self.kcomp}"
            f"({self.viscosity_mu_name}, {self.poisson_ratio_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.icomp, self.jcomp, self.kcomp,
             self.viscosity_mu_name,
             self.poisson_ratio_name))

    @override
    @memoize_method
    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        return ElasticityStressSystemKernel(
            self.dim,
            viscosity_mu_name=self.viscosity_mu_name,
            poisson_ratio_name=self.poisson_ratio_name
        ), (self.icomp, self.jcomp, self.kcomp)


@dataclass(frozen=True, repr=False)
class StokesComponentKernelBase(ExpressionKernel):
    """Base class for kernels of the Stokes equations
    (see e.g. Chapter 2 in [Pozrikidis1992]_).

    .. autoattribute:: viscosity_mu_name
    """

    viscosity_mu_name: str
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        return [
            KernelArgument(loopy_arg=lp.ValueArg(self.viscosity_mu_name, np.float64)),
        ]

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


@dataclass(frozen=True, repr=False)
class StokesletComponentKernel(StokesComponentKernelBase):
    r"""The velocity kernel for the Stokes equations (see e.g. Chapter 2 in
    [Pozrikidis1992]_).

    .. math::

        \begin{cases}
        -\mu \Delta K_{ij}(\mathbf{x}, \mathbf{y})
            + \nabla_i P_j(\mathbf{x}, \mathbf{y})
            = \delta_{ij} \delta(\mathbf{x} - \mathbf{y}), \\
        \nabla_i K_{ij}(\mathbf{x}, \mathbf{y}) = 0, \\
        \end{cases}

    where pressure kernel :math:`P_j = \partial_j K` is the derivative of the
    Laplace kernel. This kernel is often called the Stokeslet or the Oseen-Burgers
    tensor and it represents the velocity field.

    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    """

    mapper_method: ClassVar[str] = "map_stokeslet_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""

    def __init__(self,
                 dim: int,
                 icomp: int,
                 jcomp: int,
                 viscosity_mu_name: str = "mu") -> None:
        if not isinstance(viscosity_mu_name, str):
            raise TypeError(
                f"'viscosity_mu_name' is not a str: {type(viscosity_mu_name)}"
            )
        mu = sym.SpatialConstant(viscosity_mu_name)

        d = make_sym_vector("d", dim)
        r = sym.pymbolic_real_norm_2(d)
        delta_ij = 1 if icomp == jcomp else 0

        if dim == 2:
            expr = -var("log")(r)*delta_ij + d[icomp]*d[jcomp]/r**2
            scaling = -1/(4*var("pi")*mu)
        elif dim == 3:
            expr = (1/r)*delta_ij + d[icomp]*d[jcomp]/r**3
            scaling = -1/(8*var("pi")*mu)
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling,
                         viscosity_mu_name=viscosity_mu_name)
        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)

    @override
    def __str__(self) -> str:
        return (
            f"StokesletKnl{self.dim}D_{self.icomp}{self.jcomp}"
            f"({self.viscosity_mu_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.icomp, self.jcomp, self.viscosity_mu_name))

    @override
    @memoize_method
    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        return StokesletSystemKernel(
            self.dim,
            viscosity_mu_name=self.viscosity_mu_name,
        ), (self.icomp, self.jcomp)


@dataclass(frozen=True, repr=False)
class StressletComponentKernel(StokesComponentKernelBase):
    r"""The stress kernel for the Stokes equations (see e.g. Chapter 2 in
    [Pozrikidis1992]_).

    .. math::

        K_{ijk}(\mathbf{x}, \mathbf{y}) =
            -P_j \delta_{ik}
            + \mu (\partial_k K_{ij} + \partial_i K_{kj})

    where the two-index :math:`K_{ij}` is the
    :class:`~sumpy.kernel.StokesletComponentKernel`. This kernel is often
    called the Stresslet and it represents the stress tensor.

    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    .. autoattribute:: kcomp
    """
    mapper_method: ClassVar[str] = "map_stresslet_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""
    kcomp: int
    """Component index for the kernel."""

    def __init__(self,
                 dim: int,
                 icomp: int,
                 jcomp: int,
                 kcomp: int,
                 viscosity_mu_name: str = "mu") -> None:
        # mu is unused but kept for consistency with the Stokeslet.
        if not isinstance(viscosity_mu_name, str):
            raise TypeError(
                f"'viscosity_mu_name' is not a str: {type(viscosity_mu_name)}"
            )

        d = make_sym_vector("d", dim)
        r = sym.pymbolic_real_norm_2(d)

        if dim == 2:
            expr = d[icomp]*d[jcomp]*d[kcomp]/r**4
            scaling = 1/(var("pi"))
        elif dim == 3:
            expr = d[icomp]*d[jcomp]*d[kcomp]/r**5
            scaling = 3/(4*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling,
                         viscosity_mu_name=viscosity_mu_name)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)
        object.__setattr__(self, "kcomp", kcomp)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (
            self.__class__,
            (self.dim, self.icomp, self.jcomp, self.kcomp,
             self.viscosity_mu_name))

    @override
    def __str__(self) -> str:
        return (
            f"StressletKnl{self.dim}D_{self.icomp}{self.jcomp}{self.kcomp}"
            f"({self.viscosity_mu_name})")

    @override
    @memoize_method
    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        return StressletSystemKernel(
            self.dim,
            viscosity_mu_name=self.viscosity_mu_name,
        ), (self.icomp, self.jcomp, self.kcomp)


@dataclass(frozen=True, repr=False)
class LineOfCompressionKernel(ExpressionKernel):
    """A kernel for the line of compression or dilatation of constant strength
    along an axis from zero to negative infinity.

    This is used for the explicit solution to half-space linear elasticity problem.
    See [Mindlin1936]_ for details.

    .. [Mindlin1936] R. D. Mindlin (1936).
         *Force at a Point in the Interior of a Semi-Infinite Solid*.
         Physics. 7 (5): 195-202.
         `doi:10.1063/1.1745385 <https://doi.org/10.1063/1.1745385>`__.

    .. autoattribute:: axis
    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: poisson_ratio_name
    """

    mapper_method: ClassVar[str] = "map_line_of_compression_kernel"

    axis: int
    """Axis number (defaulting to 2 for the z axis)."""

    viscosity_mu_name: str
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """
    poisson_ratio_name: str
    r"""The argument name to use for Poisson's ratio :math:`\nu` when
    generating functions to evaluate this kernel.
    """

    def __init__(self,
                 dim: int = 3,
                 axis: int = 2,
                 viscosity_mu_name: str = "mu",
                 poisson_ratio_name: str = "nu"
             ) -> None:
        if not isinstance(viscosity_mu_name, str):
            raise TypeError(
                f"'viscosity_mu_name' is not a str: {type(viscosity_mu_name)}"
            )

        if not isinstance(poisson_ratio_name, str):
            raise TypeError(
                f"'poisson_ratio_name' is not a str: {type(poisson_ratio_name)}"
            )

        mu = sym.SpatialConstant(viscosity_mu_name)
        nu = sym.SpatialConstant(poisson_ratio_name)

        if dim == 3:
            d = make_sym_vector("d", dim)
            r = sym.pymbolic_real_norm_2(d)

            # Kelvin solution
            expr = d[axis] * var("log")(r + d[axis]) - r
            scaling = (1 - 2*nu)/(4*var("pi")*mu)
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling)

        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "viscosity_mu_name", viscosity_mu_name)
        object.__setattr__(self, "poisson_ratio_name", poisson_ratio_name)

    @override
    def __str__(self) -> str:
        return (
            f"LineOfCompressionKnl{self.dim}D_{self.axis}"
            f"({self.viscosity_mu_name}, {self.poisson_ratio_name})")

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (
            self.__class__,
            (self.dim, self.axis,
             self.viscosity_mu_name, self.poisson_ratio_name))

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        return [
            KernelArgument(loopy_arg=lp.ValueArg(self.viscosity_mu_name, np.float64)),
            KernelArgument(loopy_arg=lp.ValueArg(self.poisson_ratio_name, np.float64)),
        ]

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        return laplacian(w)


@dataclass(frozen=True, repr=False)
class BrinkmanComponentKernelBase(ExpressionKernel):
    """Base class for the Brinkman equations.

    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: darcy_impermeability_name
    """

    viscosity_mu_name: str
    """The argument name to use for the dynamic viscosity when generating
    functions to evaluate this kernel.
    """
    darcy_impermeability_name: str
    """The argument name to use for the Darcy impermeability when generating
    functions to evaluate this kernel.
    """

    @property
    @override
    def is_complex_valued(self) -> bool:
        # FIXME: 2D uses Hankel functions (complex-valued); 3D uses real exp
        return self.dim == 2

    @override
    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        from sumpy.codegen import register_bessel_callables
        return register_bessel_callables(loopy_knl)

    @memoize_method
    @override
    def get_args(self) -> Sequence[KernelArgument]:
        return [
            KernelArgument(loopy_arg=lp.ValueArg(self.viscosity_mu_name, np.float64)),
            KernelArgument(loopy_arg=lp.ValueArg(self.darcy_impermeability_name, np.float64)),  # noqa: E501
        ]

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        k = sym.Symbol(self.darcy_impermeability_name)

        return laplacian(laplacian(w) - k**2 * w)


@dataclass(frozen=True, repr=False)
class BrinkmanletComponentKernel(BrinkmanComponentKernelBase):
    r"""The velocity kernel for the Brinkman equations.

    .. math::

        \begin{cases}
        -\mu (\Delta K_{ij}(\mathbf{x}, \mathbf{y})
              - k^2 K_{ij}(\mathbf{x}, \mathbf{y}))
        + \nabla_i P_j(\mathbf{x}, \mathbf{y})
        = \delta_{ij}(\mathbf{x} - \mathbf{y}), \\
        \nabla_i K_{ij} = 0,
        \end{cases}

    where :math:`P_j` is the pressure kernel.

    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    """

    mapper_method: ClassVar[str] = "map_brinkmanlet_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""

    def __init__(self,
                 dim: int,
                 icomp: int,
                 jcomp: int,
                 viscosity_mu_name: str = "mu",
                 darcy_impermeability_name: str = "k") -> None:
        mu = sym.SpatialConstant(viscosity_mu_name)
        k = sym.SpatialConstant(darcy_impermeability_name)

        d = make_sym_vector("d", dim)
        r = sym.pymbolic_real_norm_2(d)
        R = k * r  # noqa: N806
        delta_ij = 1 if icomp == jcomp else 0

        # NOTE:
        #   [1] C. Pozrikidis, A Practical Guide to Boundary Element Methods,
        #   CRC Press, 2002.
        #   [2] https://dlmf.nist.gov/10.27#E8
        #   [3] https://doi.org/10.1080/00036811.2011.614604
        #   [4] https://doi.org/10.1002/mana.200710797

        if dim == 2:
            # transforming Bessel functions to Hankel functions using [2]
            K0 = var("pi") * var("I") / 2 * var("hankel_1")(0, var("I") * R)  # noqa: N806
            K1 = -var("pi") / 2 * var("hankel_1")(1, var("I") * R)            # noqa: N806

            # [1] Equations 7.7.5 and 7.7.6
            # [3] Equations 5.2 and 5.3 (for the scaling we use here)
            a = 2 * (K0 + K1 / R - 1 / R**2)
            b = 2 * (2 / R**2 - 2 * K1 / R - K0)
            expr = a * delta_ij + b * d[icomp] * d[jcomp] / r ** 2
            scaling = -1 / (4 * var("pi") * mu)
        elif dim == 3:
            # [4] Equations 4.2 and 4.3
            a = 2 * var("exp")(-R) * (1 + 1 / R + 1 / R**2) - 2 / R**2
            b = 6 / R**2 - 2 * var("exp")(-R) * (1 + 3 / R + 3 / R**2)
            expr = a * delta_ij / r + b * d[icomp] * d[jcomp] / r ** 3
            scaling = -1 / (8 * var("pi") * mu)
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling,
                         viscosity_mu_name=viscosity_mu_name,
                         darcy_impermeability_name=darcy_impermeability_name)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (
            self.__class__,
            (self.dim, self.icomp, self.jcomp,
             self.viscosity_mu_name, self.darcy_impermeability_name),
        )

    @override
    def __str__(self) -> str:
        return (
            f"BrinkmanletKnl{self.dim}D_{self.icomp}{self.jcomp}"
            f"({self.viscosity_mu_name}, {self.darcy_impermeability_name})")

    @override
    @memoize_method
    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        return BrinkmanletSystemKernel(
            self.dim,
            viscosity_mu_name=self.viscosity_mu_name,
            darcy_impermeability_name=self.darcy_impermeability_name,
        ), (self.icomp, self.jcomp)


@dataclass(frozen=True, repr=False)
class BrinkmanStressComponentKernel(BrinkmanComponentKernelBase):
    r"""A kernel for the Brinkman equations.

    .. math::

        K_{ijk} =
            -p_j \delta_{ik}
            + \mu (\partial_k K_{ij} + \partial_i K_{jk}),

    where the two-index :math:`K_{ij}` is the
    :class:`~sumpy.kernel.BrinkmanletComponentKernel` and :math:`P_j` is the
    pressure kernel.

    .. autoattribute:: icomp
    .. autoattribute:: jcomp
    .. autoattribute:: kcomp
    """

    mapper_method: ClassVar[str] = "map_brinkman_stress_kernel"

    icomp: int
    """Component index for the kernel."""
    jcomp: int
    """Component index for the kernel."""
    kcomp: int
    """Component index for the kernel."""

    def __init__(self,
                 dim: int,
                 icomp: int,
                 jcomp: int,
                 kcomp: int,
                 viscosity_mu_name: str = "mu",
                 darcy_impermeability_name: str = "k") -> None:
        k = sym.SpatialConstant(darcy_impermeability_name)

        d = make_sym_vector("d", dim)
        r = sym.pymbolic_real_norm_2(d)
        R = k * r  # noqa: N806
        delta_ij = 1 if icomp == jcomp else 0
        delta_ik = 1 if icomp == kcomp else 0
        delta_kj = 1 if jcomp == kcomp else 0

        # NOTE:
        #   [1] C. Pozrikidis, A Practical Guide to Boundary Element Methods,
        #   CRC Press, 2002.
        #   [2] https://dlmf.nist.gov/10.27#E8
        #   [3] https://doi.org/10.1080/00036811.2011.614604
        #   [4] https://doi.org/10.1002/mana.200710797

        if dim == 2:
            # transforming Bessel functions to Hankel functions using [2]
            K0 = var("pi") * var("I") / 2 * var("hankel_1")(0, var("I") * R)  # noqa: N806
            K1 = -var("pi") / 2 * var("hankel_1")(1, var("I") * R)            # noqa: N806

            # [1] Equations 7.7.7 and 7.7.8
            # [3] Equations 5.4-5.6 (for the scaling we use here)
            a = 2 * (2 / R**2 - 2 * K1 / R - K0)
            b = 8 / R**2 - 4*K0 - 2*(R + 4 / R)*K1
            c = b + R * K1
            expr = (
                2 * (a - 1) * d[jcomp] / r**2 * delta_ik
                + b * (d[kcomp] * delta_ij + d[icomp] * delta_kj) / r**2
                - 4 * c * d[icomp] * d[jcomp] * d[kcomp] / r**4
            )
            scaling = -1 / (4 * var("pi"))
        elif dim == 3:
            # [4] Equations 4.4-4.6
            d1 = 2 * var("exp")(-R) * (1 + 3 / R + 3 / R**2) - 6 / R**2 + 1
            d2 = var("exp")(-R) * (R + 3 + 6 / R + 6 / R**2) - 6 / R**2
            d3 = var("exp")(-R) * (-2 * R - 12 - 30 / R - 30 / R**2) + 30 / R**2
            expr = (
                d1 * d[jcomp] / r**3 * delta_ik
                + d2 * (d[kcomp] * delta_ij + d[icomp] * delta_kj) / r**3
                + d3 * d[icomp] * d[jcomp] * d[kcomp] / r**5
            )
            scaling = 1/(4*var("pi"))
        else:
            raise NotImplementedError(f"unsupported dimension: '{dim}'")

        super().__init__(dim, expression=expr, global_scaling_const=scaling,
                         viscosity_mu_name=viscosity_mu_name,
                         darcy_impermeability_name=darcy_impermeability_name)

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)
        object.__setattr__(self, "kcomp", kcomp)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (
            self.__class__,
            (self.dim, self.icomp, self.jcomp, self.kcomp,
             self.viscosity_mu_name, self.darcy_impermeability_name),
        )

    @override
    def __str__(self) -> str:
        return (
            f"BrinkmanStressKnl{self.dim}D_{self.icomp}{self.jcomp}{self.kcomp}"
            f"({self.viscosity_mu_name}, {self.darcy_impermeability_name})")

    @override
    @memoize_method
    def get_pde_system_kernel(self) -> tuple[SystemKernel, tuple[int, ...]]:
        return BrinkmanStressSystemKernel(
            self.dim,
            viscosity_mu_name=self.viscosity_mu_name,
            darcy_impermeability_name=self.darcy_impermeability_name,
        ), (self.icomp, self.jcomp, self.kcomp)


@dataclass(frozen=True, repr=False)
class HeatKernel(ExpressionKernel):
    r"""The Green's function for the heat equation.

    .. math::

        \frac{\partial}{\partial t} K(t, \mathbf{x}, \mathbf{y})
            - \alpha \Delta K(t, \mathbf{x}, \mathbf{y})
              = \delta(t) \delta(\mathbf{x} - \mathbf{y})

    .. note::

        This kernel cannot be used in an FMM yet and can only be used in
        expansions and evaluations that occur forward in the time dimension.
    """

    mapper_method: ClassVar[str] = "map_heat_kernel"

    heat_alpha_name: str

    def __init__(self, spatial_dims: int, heat_alpha_name: str = "alpha"):
        dim = spatial_dims + 1
        alpha = sym.SpatialConstant(heat_alpha_name)

        d = make_sym_vector("d", dim)
        t = d[-1]
        r = sym.pymbolic_real_norm_2(d[:-1])

        expr = var("exp")(-r**2/(4 * alpha * t)) / var("sqrt")(t**(dim - 1))
        scaling = 1/var("sqrt")((4*var("pi")*alpha)**(dim - 1))

        super().__init__(dim, expression=expr, global_scaling_const=scaling)
        object.__setattr__(self, "heat_alpha_name", heat_alpha_name)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (self.__class__, (self.dim - 1, self.heat_alpha_name))

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    @override
    def __str__(self):
        return f"HeatKnl{self.dim - 1}D"

    @override
    def get_args(self):
        return [
            KernelArgument(loopy_arg=lp.ValueArg(self.heat_alpha_name, np.float64))
        ]

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import diff, laplacian, make_identity_diff_op

        alpha = sym.Symbol(self.heat_alpha_name)
        w = make_identity_diff_op(self.dim - 1, time_dependent=True)
        t_mi = (*([0] * (self.dim - 1)), 1)

        return diff(w, t_mi) - alpha * laplacian(w)


# }}}


# {{{ vector PDE kernels


@dataclass(frozen=True, repr=False)
class ElasticitySystemKernel(SystemKernel):
    r"""The displacement kernel for the linear elasticity (Navier-Cauchy) equations
    (see e.g. Section 2.2 in [Hsiao2008]_).

    This kernel uses :class:`ElasticityComponentKernel` for its components.

    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: poisson_ratio_name
    """

    mapper_method: ClassVar[str] = "map_elasticity_system_kernel"

    viscosity_mu_name: str = "mu"
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """
    poisson_ratio_name: str = "nu"
    r"""The argument name to use for Poisson's ratio :math:`\nu` when generating
    functions to evaluate this kernel.
    """

    @override
    def __str__(self) -> str:
        return (
            f"ElasticityKnl{self.dim}D"
            f"({self.viscosity_mu_name}, {self.poisson_ratio_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.viscosity_mu_name, self.poisson_ratio_name))

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim)

    @override
    @keyed_memoize_method(key=lambda i, j: tuple(sorted((i, j))))
    def get_scalar_component(self, i: int, j: int, /) -> ScalarKernel:
        # NOTE: the kernel is (i, j) -> (j, i) symmetric
        i, j = sorted([i, j])

        return ElasticityComponentKernel(
            self.dim, i, j,
            viscosity_mu_name=self.viscosity_mu_name,
            poisson_ratio_name=self.poisson_ratio_name,
        )

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import (
            divergence,
            gradient,
            laplacian,
            make_identity_diff_op,
        )

        mu = sym.Symbol(self.viscosity_mu_name)
        nu = sym.Symbol(self.poisson_ratio_name)
        u = make_identity_diff_op(self.dim, self.dim)

        return mu * laplacian(u) + mu / (1 - 2 * nu) * gradient(divergence(u))


@dataclass(frozen=True, repr=False)
class ElasticityStressSystemKernel(SystemKernel):
    r"""The stress kernel for the linear elasticity (Navier-Cauchy) equations
    (see e.g. Section 2.2 in [Hsiao2008]_).

    This kernel uses :class:`ElasticityStressComponentKernel` for its components.

    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: poisson_ratio_name
    """

    mapper_method: ClassVar[str] = "map_elasticity_stress_system_kernel"

    viscosity_mu_name: str = "mu"
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """
    poisson_ratio_name: str = "nu"
    r"""The argument name to use for Poisson's ratio :math:`\nu` when generating
    functions to evaluate this kernel.
    """

    @override
    def __str__(self) -> str:
        return (
            f"ElasticityStressKnl{self.dim}D"
            f"({self.viscosity_mu_name}, {self.poisson_ratio_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.viscosity_mu_name, self.poisson_ratio_name))

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim, self.dim)

    @override
    @keyed_memoize_method(key=lambda i, j, k: ((min(i, j), max(i, j), k)))
    def get_scalar_component(self, i: int, j: int, k: int, /) -> ScalarKernel:
        if i > j:
            i, j = j, i

        return ElasticityStressComponentKernel(
            self.dim, i, j, k,
            viscosity_mu_name=self.viscosity_mu_name,
            poisson_ratio_name=self.poisson_ratio_name,
        )

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import (
            divergence,
            gradient,
            laplacian,
            make_identity_diff_op,
        )

        mu = sym.Symbol(self.viscosity_mu_name)
        nu = sym.Symbol(self.poisson_ratio_name)
        u = make_identity_diff_op(self.dim, self.dim)

        return mu * laplacian(u) + mu / (1 - 2 * nu) * gradient(divergence(u))


@dataclass(frozen=True, repr=False)
class StokesletSystemKernel(SystemKernel):
    r"""A kernel for the Stokes equations (see e.g. Chapter 2 in [Pozrikidis1992]_).

    This kernel uses :class:`StokesletComponentKernel` for its components.

    .. autoattribute:: viscosity_mu_name
    """

    mapper_method: ClassVar[str] = "map_stokeslet_system_kernel"

    viscosity_mu_name: str = "mu"
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """

    @override
    def __str__(self) -> str:
        return (
            f"StokesletKnl{self.dim}D({self.viscosity_mu_name})")

    @override
    def __reduce__(self):
        return (type(self), (self.dim, self.viscosity_mu_name))

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim)

    @override
    @keyed_memoize_method(key=lambda i, j: tuple(sorted((i, j))))
    def get_scalar_component(self, i: int, j: int, /) -> ScalarKernel:
        # NOTE: the kernel is (i, j) -> (j, i) symmetric
        i, j = sorted([i, j])

        return StokesletComponentKernel(
            self.dim, i, j,
            viscosity_mu_name=self.viscosity_mu_name,
        )

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import (
            concat,
            divergence,
            gradient,
            laplacian,
            make_identity_diff_op,
        )

        mu = sym.Symbol(self.viscosity_mu_name)
        u_and_p = make_identity_diff_op(self.dim, self.dim + 1)
        u = u_and_p[:self.dim]
        p = u_and_p[self.dim]

        return concat(mu * laplacian(u) - gradient(p), divergence(u))


@dataclass(frozen=True, repr=False)
class StressletSystemKernel(SystemKernel):
    r"""A kernel for the Stokes equations (see e.g. Chapter 2 in [Pozrikidis1992]_).

    This kernel uses :class:`StressletComponentKernel` for its components.

    .. autoattribute:: viscosity_mu_name
    """

    mapper_method: ClassVar[str] = "map_stresslet_system_kernel"

    viscosity_mu_name: str = "mu"
    r"""The argument name to use for the dynamic viscosity :math:`\mu` when
    generating functions to evaluate this kernel.
    """

    @override
    def __str__(self) -> str:
        return (
            f"StressletKnl{self.dim}D({self.viscosity_mu_name})")

    @override
    def __reduce__(self):
        return (type(self), (self.dim, self.viscosity_mu_name))

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim, self.dim)

    @override
    @keyed_memoize_method(key=lambda i, j, k: tuple(sorted((i, j, k))))
    def get_scalar_component(self, i: int, j: int, k: int, /) -> ScalarKernel:
        # NOTE: the kernel is fully permutation symmetric
        i, j, k = sorted([i, j, k])

        return StressletComponentKernel(
            self.dim, i, j, k,
            viscosity_mu_name=self.viscosity_mu_name,
        )

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import (
            concat,
            divergence,
            gradient,
            laplacian,
            make_identity_diff_op,
        )

        mu = sym.Symbol(self.viscosity_mu_name)
        u_and_p = make_identity_diff_op(self.dim, self.dim + 1)
        u = u_and_p[:self.dim]
        p = u_and_p[self.dim]

        return concat(mu * laplacian(u) - gradient(p), divergence(u))


@dataclass(frozen=True, repr=False)
class BrinkmanletSystemKernel(SystemKernel):
    r"""A kernel for the Brinkman equations.

    This kernel uses :class:`BrinkmanletComponentKernel` for its components.

    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: darcy_impermeability_name
    """

    mapper_method: ClassVar[str] = "map_brinkmanlet_system_kernel"

    viscosity_mu_name: str = "mu"
    """The argument name to use for the dynamic viscosity when generating
    functions to evaluate this kernel.
    """
    darcy_impermeability_name: str = "k"
    """The argument name to use for the Darcy impermeability when generating
    functions to evaluate this kernel.
    """

    @override
    def __str__(self) -> str:
        return (
            f"BrinkmanletKnl{self.dim}D"
            f"({self.viscosity_mu_name}, {self.darcy_impermeability_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.viscosity_mu_name, self.darcy_impermeability_name))

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim)

    @override
    @keyed_memoize_method(key=lambda i, j: tuple(sorted((i, j))))
    def get_scalar_component(self, i: int, j: int, /) -> ScalarKernel:
        # NOTE: the kernel is (i, j) -> (j, i) symmetric
        i, j = sorted([i, j])

        return BrinkmanletComponentKernel(
            self.dim, i, j,
            viscosity_mu_name=self.viscosity_mu_name,
            darcy_impermeability_name=self.darcy_impermeability_name,
        )

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import (
            concat,
            divergence,
            gradient,
            laplacian,
            make_identity_diff_op,
        )

        mu = sym.Symbol(self.viscosity_mu_name)
        k = sym.Symbol(self.darcy_impermeability_name)

        u_and_p = make_identity_diff_op(self.dim, self.dim + 1)
        u = u_and_p[:self.dim]
        p = u_and_p[self.dim]

        return concat(mu * (laplacian(u) - k**2 * u) - gradient(p), divergence(u))


@dataclass(frozen=True, repr=False)
class BrinkmanStressSystemKernel(SystemKernel):
    r"""A kernel for the Brinkman equations.

    This kernel uses :class:`BrinkmanStressComponentKernel` for its components.

    .. autoattribute:: viscosity_mu_name
    .. autoattribute:: darcy_impermeability_name
    """

    mapper_method: ClassVar[str] = "map_brinkman_stress_system_kernel"

    viscosity_mu_name: str = "mu"
    """The argument name to use for the dynamic viscosity when generating
    functions to evaluate this kernel.
    """
    darcy_impermeability_name: str = "k"
    """The argument name to use for the Darcy impermeability when generating
    functions to evaluate this kernel.
    """

    @override
    def __str__(self) -> str:
        return (
            f"BrinkmanStressKnl{self.dim}D"
            f"({self.viscosity_mu_name}, {self.darcy_impermeability_name})")

    @override
    def __reduce__(self):
        return (
            type(self),
            (self.dim, self.viscosity_mu_name, self.darcy_impermeability_name))

    @property
    @override
    def shape(self) -> tuple[int, ...]:
        return (self.dim, self.dim, self.dim)

    @override
    @keyed_memoize_method(key=lambda i, j, k: ((min(i, k), j, max(i, k))))
    def get_scalar_component(self, i: int, j: int, k: int, /) -> ScalarKernel:
        # NOTE: the kernel is (i, j, k) -> (k, j, i) symmetric
        if i > k:
            i, k = k, i

        return BrinkmanStressComponentKernel(
            self.dim, i, j, k,
            viscosity_mu_name=self.viscosity_mu_name,
            darcy_impermeability_name=self.darcy_impermeability_name,
        )

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        from sumpy.expansion.diff_op import (
            concat,
            divergence,
            gradient,
            laplacian,
            make_identity_diff_op,
        )

        mu = sym.Symbol(self.viscosity_mu_name)
        k = sym.Symbol(self.darcy_impermeability_name)

        u_and_p = make_identity_diff_op(self.dim, self.dim + 1)
        u = u_and_p[:self.dim]
        p = u_and_p[self.dim]

        return concat(mu * (laplacian(u) - k**2 * u) - gradient(p), divergence(u))


# }}}


# {{{ a kernel defined as wrapping another one--e.g., derivatives

@dataclass(frozen=True)
class KernelWrapper(ScalarKernel, ABC):
    inner_kernel: ScalarKernel
    """The kernel that is being wrapped (to take a derivative of, etc.)."""

    def __init__(self, inner_kernel: ScalarKernel) -> None:
        ScalarKernel.__init__(self, inner_kernel.dim)
        object.__setattr__(self, "inner_kernel", inner_kernel)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return self.inner_kernel.is_complex_valued

    @override
    def get_base_kernel(self) -> ScalarKernel:
        return self.inner_kernel.get_base_kernel()

    @override
    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        return self.inner_kernel.prepare_loopy_kernel(loopy_knl)

    @override
    def get_expression(self, dist_vec: sym.Matrix) -> sym.Expr:
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
    def get_global_scaling_const(self) -> sym.Expr:
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
    def replace_base_kernel(self, new_base_kernel: ScalarKernel) -> ScalarKernel:
        raise NotImplementedError(
            f"'replace_base_kernel' is not implemented for '{type(self).__name__}'")

    @override
    def get_derivative_taker(self,
            dvec: sym.Matrix,
            rscale: sym.Expr,
            sac: SymbolicAssignmentCollection | None,
        ) -> ExprDerivativeTaker:
        return self.inner_kernel.get_derivative_taker(dvec, rscale, sac)

# }}}


# {{{ derivatives

class DerivativeBase(KernelWrapper, ABC):
    """Bases: :class:`ScalarKernel`

    .. autoattribute:: inner_kernel
    .. automethod:: replace_inner_kernel
    """

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        return self.inner_kernel.get_pde_as_diff_op()

    @abstractmethod
    def replace_inner_kernel(self, new_inner_kernel: ScalarKernel) -> ScalarKernel:
        """Replace the inner kernel of this wrapper.

        This is essentially the same as :meth:`ScalarKernel.replace_base_kernel`,
        but it does not recurse.
        """


@dataclass(frozen=True)
class AxisSourceDerivative(DerivativeBase):
    """
    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_axis_source_derivative"

    axis: int
    """Direction axis for the source derivative."""

    def __init__(self, axis: int, inner_kernel: ScalarKernel) -> None:
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
    def replace_base_kernel(self, new_base_kernel: ScalarKernel) -> ScalarKernel:
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    @override
    def replace_inner_kernel(self, new_inner_kernel: ScalarKernel) -> ScalarKernel:
        return type(self)(self.axis, new_inner_kernel)


@dataclass(frozen=True)
class AxisTargetDerivative(DerivativeBase):
    """
    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_axis_target_derivative"
    target_array_name: ClassVar[str] = "targets"

    axis: int

    def __init__(self, axis: int, inner_kernel: ScalarKernel) -> None:
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
    def replace_base_kernel(self, new_base_kernel: ScalarKernel) -> ScalarKernel:
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    @override
    def replace_inner_kernel(self, new_inner_kernel: ScalarKernel) -> ScalarKernel:
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

    def __init__(self,
                 inner_kernel: ScalarKernel,
                 dir_vec_name: str | None = None) -> None:
        if dir_vec_name is None:
            dir_vec_name = f"{self.directional_kind}_derivative_dir"

        KernelWrapper.__init__(self, inner_kernel)
        object.__setattr__(self, "dir_vec_name", dir_vec_name)

    @override
    def __str__(self) -> str:
        d = "y" if self.directional_kind == "src" else "x"
        return fr"{self.dir_vec_name}·∇_{d} {self.inner_kernel}"

    @override
    def replace_base_kernel(self, new_base_kernel: ScalarKernel) -> ScalarKernel:
        return type(self)(
            self.inner_kernel.replace_base_kernel(new_base_kernel),
            dir_vec_name=self.dir_vec_name)

    @override
    def replace_inner_kernel(self, new_inner_kernel: ScalarKernel) -> ScalarKernel:
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
        dir_vec = sym.make_sym_vector(self.dir_vec_name, self.dim)

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


@dataclass(frozen=True)
class TargetPointMultiplier(KernelWrapper):
    """Bases: :class:`ScalarKernel`

    Wraps a kernel :math:`G(x, y)` and outputs :math:`x_j G(x, y)`
    where :math:`x, y` are targets and sources respectively.

    .. autoattribute:: axis
    """

    mapper_method: ClassVar[str] = "map_target_point_multiplier"
    target_array_name: ClassVar[str] = "targets"

    axis: int
    """Coordinate axis with which to multiply the kernel."""

    def __init__(self, axis: int, inner_kernel: ScalarKernel) -> None:
        super().__init__(inner_kernel)
        object.__setattr__(self, "axis", axis)

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
    def replace_base_kernel(self, new_base_kernel: ScalarKernel) -> ScalarKernel:
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    def replace_inner_kernel(self, new_inner_kernel: ScalarKernel) -> ScalarKernel:
        return type(self)(self.axis, new_inner_kernel)

# }}}


# {{{ kernel mappers

ResultT = TypeVar("ResultT")


class KernelMapper(Generic[ResultT]):
    """
    .. automethod:: __call__
    """
    def rec(self, kernel: ScalarKernel) -> ResultT:
        try:
            method = cast(
                "Callable[[ScalarKernel], ResultT]",
                getattr(self, kernel.mapper_method))
        except AttributeError as err:
            raise RuntimeError(f"{type(self)} cannot handle {type(kernel)}") from err
        else:
            return method(kernel)

    def __call__(self, kernel: ScalarKernel) -> ResultT:
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


class KernelIdentityMapper(KernelMapper[ScalarKernel]):
    def map_expression_kernel(self, kernel: ExpressionKernel) -> ScalarKernel:
        return kernel

    map_laplace_kernel: Callable[[Self, LaplaceKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_biharmonic_kernel: Callable[[Self, BiharmonicKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_helmholtz_kernel: Callable[[Self, HelmholtzKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_yukawa_kernel: Callable[[Self, YukawaKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_elasticity_kernel: Callable[[Self, ElasticityComponentKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_elasticity_stress_kernel: Callable[[Self, ElasticityStressComponentKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_line_of_compression_kernel: Callable[[Self, LineOfCompressionKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_stokeslet_kernel: Callable[[Self, StokesletComponentKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_stresslet_kernel: Callable[[Self, StressletComponentKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_brinkmanlet_kernel: Callable[[Self, BrinkmanletComponentKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_brinkman_stress_kernel: Callable[[Self, BrinkmanStressComponentKernel], ScalarKernel] = map_expression_kernel  # noqa: E501
    map_heat_kernel: Callable[[Self, HeatKernel], ScalarKernel] = map_expression_kernel

    def map_axis_target_derivative(self, kernel: AxisTargetDerivative) -> ScalarKernel:
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    def map_axis_source_derivative(self, kernel: AxisSourceDerivative) -> ScalarKernel:
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    def map_target_point_multiplier(self, kernel: TargetPointMultiplier) -> ScalarKernel:  # noqa: E501
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    def map_directional_source_derivative(
            self, kernel: DirectionalSourceDerivative) -> ScalarKernel:
        return type(kernel)(self.rec(kernel.inner_kernel),
                            dir_vec_name=kernel.dir_vec_name)


class AxisSourceDerivativeRemover(KernelIdentityMapper):
    """Removes all axis source derivatives from the kernel."""

    @override
    def map_axis_source_derivative(self, kernel: AxisSourceDerivative) -> ScalarKernel:
        return self.rec(kernel.inner_kernel)


class AxisTargetDerivativeRemover(KernelIdentityMapper):
    """Removes all axis target derivatives from the kernel."""

    @override
    def map_axis_target_derivative(self, kernel: AxisTargetDerivative) -> ScalarKernel:
        return self.rec(kernel.inner_kernel)


class TargetDerivativeRemover(AxisTargetDerivativeRemover):
    """Removes all target derivatives from the kernel."""


class SourceDerivativeRemover(AxisSourceDerivativeRemover):
    """Removes all source derivatives from the kernel."""

    @override
    def map_directional_source_derivative(
            self, kernel: DirectionalSourceDerivative) -> ScalarKernel:
        return self.rec(kernel.inner_kernel)


class TargetTransformationRemover(TargetDerivativeRemover):
    """Removes all target transformations from the kernel."""

    @override
    def map_target_point_multiplier(
            self, kernel: TargetPointMultiplier) -> ScalarKernel:
        return self.rec(kernel.inner_kernel)


SourceTransformationRemover = SourceDerivativeRemover


class DerivativeCounter(KernelCombineMapper[int]):
    """Counts the number of derivatives in the kernel."""

    @override
    def combine(self, values: Iterable[int]) -> int:
        return sum(values)

    def map_expression_kernel(self, kernel: ExpressionKernel) -> int:
        return 0

    map_laplace_kernel: \
        Callable[[Self, LaplaceKernel], int] = map_expression_kernel
    map_biharmonic_kernel: \
        Callable[[Self, BiharmonicKernel], int] = map_expression_kernel
    map_helmholtz_kernel: \
        Callable[[Self, HelmholtzKernel], int] = map_expression_kernel
    map_yukawa_kernel: \
        Callable[[Self, YukawaKernel], int] = map_expression_kernel
    map_elasticity_kernel: \
        Callable[[Self, ElasticityComponentKernel], int] = map_expression_kernel
    map_elasticity_stress_kernel: \
        Callable[[Self, ElasticityStressComponentKernel], int] = map_expression_kernel
    map_line_of_compression_kernel: \
        Callable[[Self, LineOfCompressionKernel], int] = map_expression_kernel
    map_stokeslet_kernel: \
        Callable[[Self, StokesletComponentKernel], int] = map_expression_kernel
    map_stresslet_kernel: \
        Callable[[Self, StressletComponentKernel], int] = map_expression_kernel
    map_brinkmanlet_kernel: \
        Callable[[Self, BrinkmanletComponentKernel], int] = map_expression_kernel
    map_brinkman_stress_kernel: \
        Callable[[Self, BrinkmanStressComponentKernel], int] = map_expression_kernel
    map_heat_kernel: \
        Callable[[Self, HeatKernel], int] = map_expression_kernel

    @override
    def map_axis_target_derivative(self, kernel: AxisTargetDerivative) -> int:
        return self.combine([1, self.rec(kernel.inner_kernel)])

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative
    map_axis_source_derivative = map_axis_target_derivative

# }}}


# {{{ deprecations

# TODO: once these deprecations expire, rename
#       ElasticitySystemKernel -> ElasticityKernel
#       ...
_DEPRECATED_CLASSES = {
    "BrinkmanStressKernel": (BrinkmanStressComponentKernel, 2027),
    "BrinkmanletKernel": (BrinkmanletComponentKernel, 2027),
    "ElasticityKernel": (ElasticityComponentKernel, 2027),
    "Kernel": (ScalarKernel, 2027),
    "StokesletKernel": (StokesletComponentKernel, 2027),
    "StressletKernel": (StressletComponentKernel, 2027),
}


def __getattr__(name: str) -> Any:
    result = _DEPRECATED_CLASSES.get(name)

    if result is not None:
        cls, year = result
        from warnings import warn
        warn(f"'sumpy.kernel.{name}' is deprecated. "
                f"Use 'sumpy.kernel.{cls.__name__}' instead. "
                f"'sumpy.kernel.{name}' will continue to work until {year}.",
                DeprecationWarning, stacklevel=2)
        return cls
    else:
        raise AttributeError(name)

# }}}

# vim: fdm=marker
