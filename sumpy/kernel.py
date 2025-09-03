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
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar

import numpy as np
from typing_extensions import override

import loopy as lp
import pymbolic.primitives as prim
from pymbolic import ArithmeticExpression, Expression, var
from pymbolic.mapper import CSECachingMapperMixin, IdentityMapper
from pymbolic.primitives import make_sym_vector
from pytools import memoize_method

import sumpy.symbolic as sym
from sumpy.symbolic import SpatialConstant, pymbolic_real_norm_2


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import sympy as sp

    from sumpy.derivative_taker import ExprDerivativeTaker
    from sumpy.expansion.diff_op import LinearPDESystemOperator


__doc__ = """
Kernel interface
----------------

.. autoclass:: Kernel
.. autoclass:: KernelArgument

Symbolic kernels
----------------

.. autoclass:: ExpressionKernel

PDE kernels
-----------

.. autoclass:: LaplaceKernel
.. autoclass:: BiharmonicKernel
.. autoclass:: HelmholtzKernel
.. autoclass:: YukawaKernel
.. autoclass:: StokesletKernel
.. autoclass:: StressletKernel
.. autoclass:: ElasticityKernel
.. autoclass:: LineOfCompressionKernel

Derivatives
-----------

These objects *wrap* other kernels and take derivatives
of them in the process.

.. autoclass:: DerivativeBase
.. autoclass:: AxisTargetDerivative
.. autoclass:: AxisSourceDerivative
.. autoclass:: DirectionalSourceDerivative
.. autoclass:: DirectionalTargetDerivative

Transforming kernels
--------------------

.. autoclass:: KernelMapper
.. autoclass:: KernelCombineMapper
.. autoclass:: KernelIdentityMapper
.. autoclass:: AxisSourceDerivativeRemover
.. autoclass:: AxisTargetDerivativeRemover
.. autoclass:: SourceDerivativeRemover
.. autoclass:: TargetDerivativeRemover
.. autoclass:: TargetPointMultiplier
.. autoclass:: DerivativeCounter
"""


@dataclass(frozen=True)
class KernelArgument:
    """
    .. autoattribute:: loopy_arg
    """

    loopy_arg: lp.KernelArgument
    """A :class:`loopy.KernelArgument` instance describing the type,
    name, and other features of this kernel argument when
    passed to a generated piece of code."""

    @property
    def name(self):
        return self.loopy_arg.name


# {{{ basic kernel interface

@dataclass(frozen=True)
class Kernel(ABC):
    """Basic kernel interface.

    .. attribute:: is_complex_valued
    .. attribute:: is_translation_invariant
    .. attribute:: dim

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
    """

    if TYPE_CHECKING:
        @property
        def is_complex_valued(self) -> bool:
            ...

    dim: int

    def get_base_kernel(self) -> Kernel:
        """Return the kernel being wrapped by this one, or else
        *self*.
        """
        return self

    def replace_base_kernel(self, new_base_kernel: Kernel) -> Kernel:
        """Return the base kernel being wrapped by this one, or else
        *new_base_kernel*.
        """
        return new_base_kernel

    def prepare_loopy_kernel(self, loopy_knl: lp.TranslationUnit) -> lp.TranslationUnit:
        """Apply some changes (such as registering function
        manglers) to the kernel. Return the new kernel.
        """
        return loopy_knl

    def get_code_transformer(self):
        """Return a function to postprocess the :mod:`pymbolic`
        expression generated from the result of
        :meth:`get_expression` on the way to code generation.
        """
        return lambda expr: expr

    @abstractmethod
    def get_expression(self, dist_vec: np.ndarray) -> sp.Expr:
        r"""Return a :mod:`sympy` expression for the kernel."""
        ...

    def _diff(self, expr, vec, mi):
        """Take the derivative of an expression
        """
        for i in range(self.dim):
            if mi[i] == 0:
                continue
            expr = expr.diff(vec[i], mi[i])
        return expr

    def postprocess_at_source(self, expr, avec):
        """Transform a kernel evaluation or expansion expression in a place
        where the vector a (something - source) is known. ("something" may be
        an expansion center or a target.)

        The typical use of this function is to apply source-variable
        derivatives to the kernel.
        """
        from sumpy.derivative_taker import (
            DifferentiatedExprDerivativeTaker,
            ExprDerivativeTaker,
        )
        expr_dict = {(0,)*self.dim: 1}
        expr_dict = self.get_derivative_coeff_dict_at_source(expr_dict)
        if isinstance(expr, ExprDerivativeTaker):
            return DifferentiatedExprDerivativeTaker(expr, expr_dict)

        result = 0
        for mi, coeff in expr_dict.items():
            result += coeff * self._diff(expr, avec, mi)
        return result

    def postprocess_at_target(self, expr, bvec):
        """Transform a kernel evaluation or expansion expression in a place
        where the vector b (target - something) is known. ("something" may be
        an expansion center or a target.)

        The typical use of this function is to apply target-variable
        derivatives to the kernel.

        :arg expr: may be a :class:`sympy.core.expr.Expr` or a
            :class:`sumpy.derivative_taker.DifferentiatedExprDerivativeTaker`.
        """
        return expr

    def get_derivative_coeff_dict_at_source(self, expr_dict):
        r"""Get the derivative transformation of the expression at source
        represented by the dictionary expr_dict which is mapping from multi-index
        `mi` to coefficient `coeff`.
        Expression represented by the dictionary `expr_dict` is
        :math:`\sum_{mi} \frac{\partial^mi}{x^mi}G * coeff`. Returns an
        expression of the same type.

        This function is meant to be overridden by child classes where necessary.
        """
        return expr_dict

    @abstractmethod
    def get_global_scaling_const(self) -> ArithmeticExpression:
        r"""Return a global scaling constant of the kernel.
        Typically, this ensures that the kernel is scaled so that
        :math:`\mathcal L(G)(x)=C\delta(x)` with a constant of 1, where
        :math:`\mathcal L` is the PDE operator associated with the kernel.
        Not to be confused with *rscale*, which keeps expansion
        coefficients benignly scaled.
        """
        ...

    def get_args(self) -> Sequence[KernelArgument]:
        """Return list of :class:`KernelArgument` instances describing
        extra arguments used by the kernel.
        """
        return []

    def get_source_args(self) -> Sequence[KernelArgument]:
        """Return list of :class:`KernelArgument` instances describing
        extra arguments used by kernel in picking up contributions
        from point sources.
        """
        return []

    @abstractmethod
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        ...

    @abstractmethod
    def get_derivative_taker(self, dvec, rscale, sac) -> ExprDerivativeTaker:
        ...

    # TODO: Allow kernels that are not translation invariant
    is_translation_invariant: ClassVar[bool] = True

    mapper_method: ClassVar[str]

# }}}


@dataclass(frozen=True)
class ExpressionKernel(Kernel):
    r"""
    .. autoattribute:: expression
    .. autoattribute:: global_scaling_const
    .. autoattribute:: is_complex_valued
    """

    expression: Expression
    """A :mod:`pymbolic` expression depending on
    variables *d_1* through *d_N* where *N* equals *dim*.
    (These variables match what is returned from
    :func:`pymbolic.primitives.make_sym_vector` with
    argument `"d"`.) Any variable that is not *d* or
    a :class:`~sumpy.symbolic.SpatialConstant` will be
    viewed as potentially spatially varying.
    """

    global_scaling_const: Expression

    r"""A constant :mod:`pymbolic` expression for the
    global scaling of the kernel. Typically, this ensures that
    the kernel is scaled so that :math:`\mathcal L(G)(x)=C\delta(x)`
    with a constant of 1, where :math:`\mathcal L` is the PDE
    operator associated with the kernel. Not to be confused with
    *rscale*, which keeps expansion coefficients benignly scaled.
    """

    def __repr__(self):
        return f"ExprKnl{self.dim}D"

    def get_expression(self, scaled_dist_vec):
        """Return :attr:`expression` as a :class:`sumpy.symbolic.Basic`."""

        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        expr = PymbolicToSympyMapperWithSymbols()(self.expression)

        if self.dim != len(scaled_dist_vec):
            raise ValueError("dist_vec length does not match expected dimension")

        from sumpy.symbolic import Symbol
        expr = expr.xreplace({
            Symbol(f"d{i}"): dist_vec_i
            for i, dist_vec_i in enumerate(scaled_dist_vec)
            })

        return expr

    def get_global_scaling_const(self):
        """Return a global scaling of the kernel as a :class:`sumpy.symbolic.Basic`.
        """

        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        return PymbolicToSympyMapperWithSymbols()(
                self.global_scaling_const)

    mapper_method: ClassVar[str] = "map_expression_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.derivative_taker.ExprDerivativeTaker` instance
        that supports taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.derivative_taker import ExprDerivativeTaker
        return ExprDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        r"""
        Returns the PDE for the kernel as a
        :class:`sumpy.expansion.diff_op.LinearPDESystemOperator` object `L`
        where `L(u) = 0` is the PDE.
        """
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
    def __init__(self, dim: int):
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
            raise NotImplementedError("unsupported dimensionality")

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    def __repr__(self):
        return f"LapKnl{self.dim}D"

    mapper_method: ClassVar[str] = "map_laplace_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.derivative_taker.ExprDerivativeTaker` instance
        that supports taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.derivative_taker import LaplaceDerivativeTaker
        return LaplaceDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


class BiharmonicKernel(ExpressionKernel):
    def __init__(self, dim: int):
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
            raise RuntimeError("unsupported dimensionality")

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    def __repr__(self):
        return f"BiharmKnl{self.dim}D"

    mapper_method: ClassVar[str] = "map_biharmonic_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.derivative_taker.ExprDerivativeTaker` instance
        that supports taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.derivative_taker import RadialDerivativeTaker
        return RadialDerivativeTaker(self.get_expression(dvec), dvec, rscale,
                sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


@dataclass(frozen=True)
class HelmholtzKernel(ExpressionKernel):
    helmholtz_k_name: str
    """
    The argument name to use for the Helmholtz
    parameter when generating functions to evaluate this kernel.
    """
    allow_evanescent: bool

    def __init__(self, dim: int, helmholtz_k_name: str = "k",
            allow_evanescent: bool = False):
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
            raise RuntimeError("unsupported dimensionality")

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

        object.__setattr__(self, "helmholtz_k_name", helmholtz_k_name)
        object.__setattr__(self, "allow_evanescent", allow_evanescent)

    @property
    @override
    def is_complex_valued(self) -> bool:
        return True

    def __repr__(self):
        return f"HelmKnl{self.dim}D({self.helmholtz_k_name})"

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import register_bessel_callables
        return register_bessel_callables(loopy_knl)

    def get_args(self):
        k_dtype = np.complex128 if self.allow_evanescent else np.float64
        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.helmholtz_k_name, k_dtype),
                    )]

    mapper_method: ClassVar[str] = "map_helmholtz_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.derivative_taker.ExprDerivativeTaker` instance
        that supports taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.derivative_taker import HelmholtzDerivativeTaker
        return HelmholtzDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op

        w = make_identity_diff_op(self.dim)
        k = sym.Symbol(self.helmholtz_k_name)
        return (laplacian(w) + k**2 * w)


@dataclass(frozen=True)
class YukawaKernel(ExpressionKernel):
    yukawa_lambda_name: str

    def __init__(self, dim: int, yukawa_lambda_name: str = "lam"):
        """
        :arg yukawa_lambda_name: The argument name to use for the Yukawa
            parameter when generating functions to evaluate this kernel.
        """
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

            scaling = -1/(2*var("pi")) * scaling_for_K0
        elif dim == 3:
            # NOTE: to get the expression, we do the following and simplify
            # 1. express K(1/2, lam r) as a modified spherical Bessel function
            #   k(0, lam r) using [3] and use expression for k(0, lam r) from [4]
            # 2. or use (AS 10.2.17) directly
            expr = var("exp")(-lam*r) / r

            scaling = -1/(4 * var("pi")**2)
        else:
            raise RuntimeError("unsupported dimensionality")

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

        object.__setattr__(self, "yukawa_lambda_name", yukawa_lambda_name)

    @property
    @override
    def is_complex_valued(self) -> bool:
        # FIXME
        return True

    def __repr__(self):
        return f"YukKnl{self.dim}D({self.yukawa_lambda_name})"

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import register_bessel_callables
        return register_bessel_callables(loopy_knl)

    def get_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.yukawa_lambda_name, np.float64),
                    )]

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.derivative_taker.ExprDerivativeTaker` instance
        that supports taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.derivative_taker import HelmholtzDerivativeTaker
        return HelmholtzDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        lam = sym.Symbol(self.yukawa_lambda_name)
        return (laplacian(w) - lam**2 * w)

    mapper_method: ClassVar[str] = "map_yukawa_kernel"


class ElasticityKernel(ExpressionKernel):
    def __new__(cls, dim, icomp, jcomp, viscosity_mu="mu", poisson_ratio="nu"):
        if poisson_ratio == 0.5:
            instance = super().__new__(StokesletKernel)
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, dim, icomp, jcomp, viscosity_mu="mu", poisson_ratio="nu"):
        r"""
        :arg viscosity_mu: The argument name to use for
                dynamic viscosity :math:`\mu` when generating functions to
                evaluate this kernel. Can also be a numeric value.
        :arg poisson_ratio: The argument name to use for
                Poisson's ratio :math:`\nu` when generating functions to
                evaluate this kernel. Can also be a numeric value.
        """
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
            raise RuntimeError("unsupported dimensionality")

        self.viscosity_mu = mu
        self.poisson_ratio = nu
        self.icomp = icomp
        self.jcomp = jcomp

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    def __repr__(self):
        return f"ElasticityKnl{self.dim}D_{self.icomp}{self.jcomp}"

    @memoize_method
    def get_args(self):
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.viscosity_mu)
        res = []
        for v in variables:
            res.append(KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64)))
        return res + self.get_source_args()

    @memoize_method
    def get_source_args(self):
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.poisson_ratio)
        res = []
        for v in variables:
            res.append(KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64)))
        return res

    mapper_method: ClassVar[str] = "map_elasticity_kernel"

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


class StokesletKernel(ElasticityKernel):
    def __new__(cls, dim, icomp, jcomp, viscosity_mu="mu", poisson_ratio="0.5"):
        return object.__new__(cls)

    def __init__(self, dim, icomp, jcomp, viscosity_mu="mu", poisson_ratio=0.5):
        super().__init__(dim, icomp, jcomp, viscosity_mu, poisson_ratio)

    def __repr__(self):
        return f"StokesletKnl{self.dim}D_{self.icomp}{self.jcomp}"


@dataclass(frozen=True)
class StressletKernel(ExpressionKernel):
    icomp: int
    jcomp: int
    kcomp: int
    viscosity_mu: SpatialConstant

    def __init__(self, dim, icomp, jcomp, kcomp, viscosity_mu="mu"):
        r"""
        :arg viscosity_mu: The argument name to use for
                dynamic viscosity :math:`\mu` the then generating functions to
                evaluate this kernel.
        """
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
            raise RuntimeError("unsupported dimensionality")

        object.__setattr__(self, "icomp", icomp)
        object.__setattr__(self, "jcomp", jcomp)
        object.__setattr__(self, "kcomp", kcomp)
        object.__setattr__(self, "viscosity_mu", mu)

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    def __repr__(self):
        return f"StressletKnl{self.dim}D_{self.icomp}{self.jcomp}{self.kcomp}"

    @memoize_method
    def get_args(self):
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.viscosity_mu)
        return [
                KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64))
                for v in variables]

    mapper_method: ClassVar[str] = "map_stresslet_kernel"

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


class LineOfCompressionKernel(ExpressionKernel):
    """A kernel for the line of compression or dilatation of constant strength
    along the axis "axis" from zero to negative infinity. This is used for the
    explicit solution to half-space Elasticity problem. See [1] for details.

    [1]: Mindlin, R.: Force at a Point in the Interior of a Semi-Infinite Solid
         https://doi.org/10.1063/1.1745385
    """
    axis: int
    viscosity_mu: SpatialConstant
    poisson_ratio: SpatialConstant

    def __init__(self,
                 dim: int = 3,
                 axis: int = 2,
                 viscosity_mu: str | SpatialConstant = "mu",
                 poisson_ratio: str | SpatialConstant = "nu"
             ):
        r"""
        :arg axis: axis number defaulting to 2 for the z axis.
        :arg viscosity_mu: The argument name to use for
                dynamic viscosity :math:`\mu` when generating functions to
                evaluate this kernel. Can also be a numeric value.
        :arg poisson_ratio: The argument name to use for
                Poisson's ratio :math:`\nu` when generating functions to
                evaluate this kernel. Can also be a numeric value.
        """
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
            raise RuntimeError("unsupported dimensionality")

        self.viscosity_mu = mu
        self.poisson_ratio = nu
        self.axis = axis

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                )

    @property
    @override
    def is_complex_valued(self) -> bool:
        return False

    def __repr__(self):
        return f"LineOfCompressionKnl{self.dim}D_{self.axis}"

    @memoize_method
    def get_args(self):
        from sumpy.tools import get_all_variables
        variables = list(get_all_variables(self.viscosity_mu)) \
            + list(get_all_variables(self.poisson_ratio))
        res = []
        for v in variables:
            res.append(KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64)))
        return res

    mapper_method: ClassVar[str] = "map_line_of_compression_kernel"

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import laplacian, make_identity_diff_op
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


# }}}


# {{{ a kernel defined as wrapping another one--e.g., derivatives

@dataclass(frozen=True)
class KernelWrapper(Kernel, ABC):
    inner_kernel: Kernel

    def __init__(self, inner_kernel: Kernel):
        Kernel.__init__(self, inner_kernel.dim)
        object.__setattr__(self, "inner_kernel", inner_kernel)

    def get_base_kernel(self):
        return self.inner_kernel.get_base_kernel()

    def prepare_loopy_kernel(self, loopy_knl):
        return self.inner_kernel.prepare_loopy_kernel(loopy_knl)

    @property
    def is_complex_valued(self):
        return self.inner_kernel.is_complex_valued

    def get_expression(self, scaled_dist_vec):
        return self.inner_kernel.get_expression(scaled_dist_vec)

    def get_derivative_coeff_dict_at_source(self, expr_dict):
        return self.inner_kernel.get_derivative_coeff_dict_at_source(expr_dict)

    def postprocess_at_target(self, expr, bvec):
        return self.inner_kernel.postprocess_at_target(expr, bvec)

    def get_global_scaling_const(self):
        return self.inner_kernel.get_global_scaling_const()

    def get_code_transformer(self):
        return self.inner_kernel.get_code_transformer()

    def get_args(self):
        return self.inner_kernel.get_args()

    def get_source_args(self):
        return self.inner_kernel.get_source_args()

    def replace_base_kernel(self, new_base_kernel):
        raise NotImplementedError("replace_base_kernel is not implemented "
            "for this wrapper.")

    def get_derivative_taker(self, dvec, rscale, sac):
        return self.inner_kernel.get_derivative_taker(dvec, rscale, sac)

# }}}


# {{{ derivatives

class DerivativeBase(KernelWrapper, ABC):
    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        return self.inner_kernel.get_pde_as_diff_op()


@dataclass(frozen=True)
class AxisSourceDerivative(DerivativeBase):
    axis: int

    def __init__(self, axis: int, inner_kernel: Kernel):
        super().__init__(inner_kernel)
        object.__setattr__(self, "axis", axis)

    def __str__(self):
        return f"d/dy{self.axis} {self.inner_kernel}"

    def get_derivative_coeff_dict_at_source(self, expr_dict):
        expr_dict = self.inner_kernel.get_derivative_coeff_dict_at_source(
            expr_dict)
        result = {}
        for mi, coeff in expr_dict.items():
            new_mi = list(mi)
            new_mi[self.axis] += 1
            result[tuple(new_mi)] = -coeff
        return result

    def replace_base_kernel(self, new_base_kernel):
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    def replace_inner_kernel(self, new_inner_kernel):
        return type(self)(self.axis, new_inner_kernel)

    mapper_method: ClassVar[str] = "map_axis_source_derivative"


@dataclass(frozen=True)
class AxisTargetDerivative(DerivativeBase):
    target_array_name: ClassVar[str] = "targets"

    axis: int

    def __init__(self, axis: int, inner_kernel: Kernel):
        super().__init__(inner_kernel)
        object.__setattr__(self, "axis", axis)

    @override
    def __str__(self):
        return f"d/dx{self.axis} {self.inner_kernel}"

    def postprocess_at_target(self, expr, bvec):
        from sumpy.derivative_taker import (
            DifferentiatedExprDerivativeTaker,
            diff_derivative_coeff_dict,
        )
        from sumpy.symbolic import make_sym_vector as make_sympy_vector

        target_vec = make_sympy_vector(self.target_array_name, self.dim)

        # bvec = tgt - ctr
        expr = self.inner_kernel.postprocess_at_target(expr, bvec)
        if isinstance(expr, DifferentiatedExprDerivativeTaker):
            transformation = diff_derivative_coeff_dict(expr.derivative_coeff_dict,
                    self.axis, target_vec)
            return DifferentiatedExprDerivativeTaker(expr.taker, transformation)
        else:
            # Since `bvec` and `tgt` are two different symbolic variables
            # need to differentiate by both to get the correct answer
            return expr.diff(bvec[self.axis]) + expr.diff(target_vec[self.axis])

    def replace_base_kernel(self, new_base_kernel):
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    def replace_inner_kernel(self, new_inner_kernel):
        return type(self)(self.axis, new_inner_kernel)

    mapper_method: ClassVar[str] = "map_axis_target_derivative"


class _VectorIndexAdder(CSECachingMapperMixin[Expression, []], IdentityMapper[[]]):
    def __init__(self, vec_name, additional_indices):
        self.vec_name = vec_name
        self.additional_indices = additional_indices

    def map_subscript(self, expr):
        from pymbolic.primitives import CommonSubexpression, cse_scope
        if (expr.aggregate.name == self.vec_name
                and isinstance(expr.index, int)):
            return CommonSubexpression(
                    expr.aggregate[(expr.index, *self.additional_indices)],
                    prefix=None, scope=cse_scope.EVALUATION)
        else:
            return IdentityMapper.map_subscript(self, expr)

    def map_common_subexpression_uncached(self,
                expr: prim.CommonSubexpression) -> Expression:
        result = self.rec(expr.child)
        if result is expr.child:
            return expr

        return type(expr)(
                      result, expr.prefix, expr.scope, **expr.get_extra_properties())


@dataclass(frozen=True)
class DirectionalDerivative(DerivativeBase):
    directional_kind: ClassVar[Literal["src", "tgt"]]

    dir_vec_name: str

    def __init__(self, inner_kernel: Kernel, dir_vec_name: str | None = None):
        if dir_vec_name is None:
            dir_vec_name = f"{self.directional_kind}_derivative_dir"

        KernelWrapper.__init__(self, inner_kernel)
        object.__setattr__(self, "dir_vec_name", dir_vec_name)

    def replace_base_kernel(self, new_base_kernel):
        return type(self)(self.inner_kernel.replace_base_kernel(new_base_kernel),
            dir_vec_name=self.dir_vec_name)

    @override
    def __str__(self):
        return r"{}·∇_{} {}".format(
                self.dir_vec_name,
                "y" if self.directional_kind == "src" else "x",
                self.inner_kernel)


class DirectionalTargetDerivative(DirectionalDerivative):
    directional_kind: ClassVar[Literal["src", "tgt"]] = "tgt"
    target_array_name = "targets"

    @override
    def get_code_transformer(self):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.dir_vec_name])
        from pymbolic.primitives import Variable
        via = _VectorIndexAdder(self.dir_vec_name, (Variable("itgt"),))

        inner_transform = self.inner_kernel.get_code_transformer()

        def transform(expr):
            return via(vcr(inner_transform(expr)))

        return transform

    def postprocess_at_target(self, expr, bvec):
        from sumpy.derivative_taker import (
            DifferentiatedExprDerivativeTaker,
            diff_derivative_coeff_dict,
        )
        from sumpy.symbolic import make_sym_vector as make_sympy_vector
        dir_vec = make_sympy_vector(self.dir_vec_name, self.dim)
        target_vec = make_sympy_vector(self.target_array_name, self.dim)

        expr = self.inner_kernel.postprocess_at_target(expr, bvec)

        # bvec = tgt - center
        if not isinstance(expr, DifferentiatedExprDerivativeTaker):
            result = 0
            for axis in range(self.dim):
                # Since `bvec` and `tgt` are two different symbolic variables
                # need to differentiate by both to get the correct answer
                result += (expr.diff(bvec[axis]) + expr.diff(target_vec[axis])) \
                        * dir_vec[axis]
            return result

        new_transformation = defaultdict(lambda: 0)
        for axis in range(self.dim):
            axis_transformation = diff_derivative_coeff_dict(
                    expr.derivative_coeff_dict, axis, target_vec)
            for mi, coeff in axis_transformation.items():
                new_transformation[mi] += coeff * dir_vec[axis]

        return DifferentiatedExprDerivativeTaker(expr.taker,
                dict(new_transformation))

    def get_args(self):
        return [
            KernelArgument(
                loopy_arg=lp.GlobalArg(
                    self.dir_vec_name,
                    None,
                    shape=(self.dim, "ntargets"),
                    offset=lp.auto
                ),
            ),
            *self.inner_kernel.get_args()
        ]

    def prepare_loopy_kernel(self, loopy_knl):
        loopy_knl = self.inner_kernel.prepare_loopy_kernel(loopy_knl)
        return lp.tag_array_axes(loopy_knl, self.dir_vec_name, "sep,C")

    mapper_method: ClassVar[str] = "map_directional_target_derivative"


class DirectionalSourceDerivative(DirectionalDerivative):
    directional_kind: ClassVar[Literal["src", "tgt"]] = "src"

    def get_code_transformer(self):
        inner = self.inner_kernel.get_code_transformer()
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.dir_vec_name])
        from pymbolic.primitives import Variable
        via = _VectorIndexAdder(self.dir_vec_name, (Variable("isrc"),))

        def transform(expr):
            return via(vcr(inner(expr)))

        return transform

    def get_derivative_coeff_dict_at_source(self, expr_dict):
        from sumpy.symbolic import make_sym_vector as make_sympy_vector
        dir_vec = make_sympy_vector(self.dir_vec_name, self.dim)

        expr_dict = self.inner_kernel.get_derivative_coeff_dict_at_source(
            expr_dict)

        # avec = center-src -> minus sign from chain rule
        result = defaultdict(lambda: 0)
        for mi, coeff in expr_dict.items():
            for axis in range(self.dim):
                new_mi = list(mi)
                new_mi[axis] += 1
                result[tuple(new_mi)] += -coeff * dir_vec[axis]
        return result

    def get_source_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.GlobalArg(
                        self.dir_vec_name,
                        None,
                        shape=(self.dim, "nsources"),
                        offset=lp.auto),
                    ),
                    *self.inner_kernel.get_source_args()]

    def prepare_loopy_kernel(self, loopy_knl):
        loopy_knl = self.inner_kernel.prepare_loopy_kernel(loopy_knl)
        return lp.tag_array_axes(loopy_knl, self.dir_vec_name, "sep,C")

    mapper_method: ClassVar[str] = "map_directional_source_derivative"


class TargetPointMultiplier(KernelWrapper):
    """Wraps a kernel :math:`G(x, y)` and outputs :math:`x_j G(x, y)`
    where :math:`x, y` are targets and sources respectively.
    """

    axis: int

    target_array_name: ClassVar[str] = "targets"

    def __init__(self, axis, inner_kernel):
        KernelWrapper.__init__(self, inner_kernel)
        self.axis = axis

    def __str__(self):
        return f"x{self.axis} {self.inner_kernel}"

    def replace_base_kernel(self, new_base_kernel):
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    def replace_inner_kernel(self, new_inner_kernel):
        return type(self)(self.axis, new_inner_kernel)

    def postprocess_at_target(self, expr, avec):
        from sumpy.derivative_taker import (
            DifferentiatedExprDerivativeTaker,
            ExprDerivativeTaker,
        )
        from sumpy.symbolic import make_sym_vector as make_sympy_vector

        expr = self.inner_kernel.postprocess_at_target(expr, avec)
        target_vec = make_sympy_vector(self.target_array_name, self.dim)

        zeros = tuple([0]*self.dim)
        mult = target_vec[self.axis]

        if isinstance(expr, DifferentiatedExprDerivativeTaker):
            transform = {mi: coeff * mult for mi, coeff in
                    expr.derivative_coeff_dict.items()}
            return DifferentiatedExprDerivativeTaker(expr.taker, transform)
        elif isinstance(expr, ExprDerivativeTaker):
            return DifferentiatedExprDerivativeTaker({zeros: mult})
        else:
            return mult * expr

    @override
    def get_code_transformer(self):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.target_array_name])
        from pymbolic.primitives import Variable
        via = _VectorIndexAdder(self.target_array_name, (Variable("itgt"),))

        inner_transform = self.inner_kernel.get_code_transformer()

        def transform(expr):
            return via(vcr(inner_transform(expr)))

        return transform

    @override
    def get_pde_as_diff_op(self) -> LinearPDESystemOperator:
        raise NotImplementedError("no PDE is known")

    mapper_method: ClassVar[str] = "map_target_point_multiplier"

# }}}


# {{{ kernel mappers

ResultT = TypeVar("ResultT")


class KernelMapper(Generic[ResultT]):
    def rec(self, kernel: Kernel) -> ResultT:
        try:
            method = getattr(self, kernel.mapper_method)
        except AttributeError as err:
            raise RuntimeError(f"{type(self)} cannot handle {type(kernel)}") from err
        else:
            return method(kernel)

    __call__ = rec


class KernelCombineMapper(KernelMapper[ResultT], ABC):
    @abstractmethod
    def combine(self, values: Iterable[ResultT]) -> ResultT:
        raise NotImplementedError

    def map_axis_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative
    map_axis_source_derivative = map_axis_target_derivative
    map_target_point_multiplier = map_axis_target_derivative


class KernelIdentityMapper(KernelMapper[Kernel]):
    def map_expression_kernel(self, kernel):
        return kernel

    map_laplace_kernel = map_expression_kernel
    map_biharmonic_kernel = map_expression_kernel
    map_helmholtz_kernel = map_expression_kernel
    map_yukawa_kernel = map_expression_kernel
    map_elasticity_kernel = map_expression_kernel
    map_line_of_compression_kernel = map_expression_kernel
    map_stresslet_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel):
        return type(kernel)(kernel.axis, self.rec(kernel.inner_kernel))

    map_axis_source_derivative = map_axis_target_derivative
    map_target_point_multiplier = map_axis_target_derivative

    def map_directional_target_derivative(self, kernel):
        return type(kernel)(
                self.rec(kernel.inner_kernel),
                kernel.dir_vec_name)

    map_directional_source_derivative = map_directional_target_derivative


class AxisSourceDerivativeRemover(KernelIdentityMapper):
    def map_axis_source_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)


class AxisTargetDerivativeRemover(KernelIdentityMapper):
    def map_axis_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)


class TargetDerivativeRemover(AxisTargetDerivativeRemover):
    def map_directional_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)


class SourceDerivativeRemover(AxisSourceDerivativeRemover):
    def map_directional_source_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)


class TargetTransformationRemover(TargetDerivativeRemover):
    def map_target_point_multiplier(self, kernel):
        return self.rec(kernel.inner_kernel)


SourceTransformationRemover = SourceDerivativeRemover


class DerivativeCounter(KernelCombineMapper[int]):
    def combine(self, values):
        return max(values)

    def map_expression_kernel(self, kernel):
        return 0

    map_laplace_kernel = map_expression_kernel
    map_biharmonic_kernel = map_expression_kernel
    map_helmholtz_kernel = map_expression_kernel
    map_yukawa_kernel = map_expression_kernel
    map_line_of_compression_kernel = map_expression_kernel
    map_stresslet_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel):
        return 1 + self.rec(kernel.inner_kernel)

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative
    map_axis_source_derivative = map_axis_target_derivative

# }}}


# vim: fdm=marker
