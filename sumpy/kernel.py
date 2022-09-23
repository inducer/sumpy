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

from typing import ClassVar, Tuple

import loopy as lp
import numpy as np
from pymbolic.mapper import IdentityMapper, CSECachingMapperMixin
from sumpy.symbolic import pymbolic_real_norm_2, SpatialConstant
import sumpy.symbolic as sym
from pymbolic.primitives import make_sym_vector
from pymbolic import var
from pytools import memoize_method
from collections import defaultdict

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


class KernelArgument:
    """
    .. attribute:: loopy_arg

        A :class:`loopy.KernelArgument` instance describing the type,
        name, and other features of this kernel argument when
        passed to a generated piece of code.
    """

    def __init__(self, loopy_arg):
        self.loopy_arg = loopy_arg

    @property
    def name(self):
        return self.loopy_arg.name

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        if not type(self) == KernelArgument:
            return NotImplemented
        if not type(other) == KernelArgument:
            return NotImplemented
        return self.loopy_arg == other.loopy_arg

    def __ne__(self, other):
        # Needed for python2
        return not self == other

    def __hash__(self):                 # pylint: disable=invalid-hash-returned
        return (type(self), self.loopy_arg)


# {{{ basic kernel interface

class Kernel:
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

    init_arg_names: ClassVar[Tuple[str, ...]]

    def __init__(self, dim):
        self.dim = dim

    # {{{ hashing/pickling/equality

    def __eq__(self, other):
        if self is other:
            return True
        elif hash(self) != hash(other):
            return False
        else:
            return (type(self) is type(other)
                    and self.__getinitargs__() == other.__getinitargs__())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        try:
            return self.hash_value
        except AttributeError:
            self.hash_value = hash((type(self),) + self.__getinitargs__())
            return self.hash_value

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.__getinitargs__())

    def __getinitargs__(self):
        return (self.dim,)

    def __getstate__(self):
        return self.__getinitargs__()

    def __setstate__(self, state):
        # Can't use trivial pickling: hash_value cache must stay unset
        assert len(self.init_arg_names) == len(state)
        self.__init__(*state)

    # }}}

    def get_base_kernel(self):
        """Return the kernel being wrapped by this one, or else
        *self*.
        """
        return self

    def replace_base_kernel(self, new_base_kernel):
        """Return the base kernel being wrapped by this one, or else
        *new_base_kernel*.
        """
        return new_base_kernel

    def prepare_loopy_kernel(self, loopy_knl):
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

    def get_expression(self, dist_vec):
        r"""Return a :mod:`sympy` expression for the kernel."""
        raise NotImplementedError

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
        from sumpy.tools import (ExprDerivativeTaker,
            DifferentiatedExprDerivativeTaker)
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
            :class:`sumpy.tools.DifferentiatedExprDerivativeTaker`.
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

    def get_global_scaling_const(self):
        r"""Return a global scaling constant of the kernel.
        Typically, this ensures that the kernel is scaled so that
        :math:`\mathcal L(G)(x)=C\delta(x)` with a constant of 1, where
        :math:`\mathcal L` is the PDE operator associated with the kernel.
        Not to be confused with *rscale*, which keeps expansion
        coefficients benignly scaled.
        """
        raise NotImplementedError

    def get_args(self):
        """Return list of :class:`KernelArgument` instances describing
        extra arguments used by the kernel.
        """
        return []

    def get_source_args(self):
        """Return list of :class:`KernelArgument` instances describing
        extra arguments used by kernel in picking up contributions
        from point sources.
        """
        return []

    # TODO: Allow kernels that are not translation invariant
    is_translation_invariant = True

# }}}


class ExpressionKernel(Kernel):
    r"""
    .. attribute:: expression

        A :mod:`pymbolic` expression depending on
        variables *d_1* through *d_N* where *N* equals *dim*.
        (These variables match what is returned from
        :func:`pymbolic.primitives.make_sym_vector` with
        argument `"d"`.) Any variable that is not *d* or
        a :class:`~sumpy.symbolic.SpatialConstant` will be
        viewed as potentially spatially varying.

    .. attribute:: global_scaling_const

        A constant :mod:`pymbolic` expression for the
        global scaling of the kernel. Typically, this ensures that
        the kernel is scaled so that :math:`\mathcal L(G)(x)=C\delta(x)`
        with a constant of 1, where :math:`\mathcal L` is the PDE
        operator associated with the kernel. Not to be confused with
        *rscale*, which keeps expansion coefficients benignly scaled.

    .. attribute:: is_complex_valued

    .. automethod:: __init__
    .. automethod:: get_expression
    """

    init_arg_names = ("dim", "expression", "global_scaling_const",
            "is_complex_valued")

    def __init__(self, dim, expression, global_scaling_const,
            is_complex_valued):
        # expression and global_scaling_const are pymbolic objects because
        # those pickle cleanly. D'oh, sympy!

        Kernel.__init__(self, dim)

        self.expression = expression
        self.global_scaling_const = global_scaling_const
        self.is_complex_valued = is_complex_valued

    def __getinitargs__(self):
        return (self.dim, self.expression, self.global_scaling_const,
                self.is_complex_valued)

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

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        for name, value in zip(self.init_arg_names, self.__getinitargs__()):
            if name in ["expression", "global_scaling_const"]:
                from pymbolic.mapper.persistent_hash import (
                        PersistentHashWalkMapper as PersistentHashWalkMapper)
                PersistentHashWalkMapper(key_hash)(value)
            else:
                key_builder.rec(key_hash, value)

    mapper_method = "map_expression_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.tools.ExprDerivativeTaker` instance that supports
        taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.tools import ExprDerivativeTaker
        return ExprDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        r"""
        Returns the PDE for the kernel as a
        :class:`sumpy.expansion.diff_op.LinearPDESystemOperator` object `L`
        where `L(u) = 0` is the PDE.
        """
        raise NotImplementedError


one_kernel_2d = ExpressionKernel(
        dim=2,
        expression=1,
        global_scaling_const=1,
        is_complex_valued=False)
one_kernel_3d = ExpressionKernel(
        dim=3,
        expression=1,
        global_scaling_const=1,
        is_complex_valued=False)


# {{{ PDE kernels

class LaplaceKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim):
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
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return f"LapKnl{self.dim}D"

    mapper_method = "map_laplace_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.tools.ExprDerivativeTaker` instance that supports
        taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.tools import LaplaceDerivativeTaker
        return LaplaceDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


class BiharmonicKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim):
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
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self.dim,)

    def __repr__(self):
        return f"BiharmKnl{self.dim}D"

    mapper_method = "map_biharmonic_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.tools.ExprDerivativeTaker` instance that supports
        taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.tools import RadialDerivativeTaker
        return RadialDerivativeTaker(self.get_expression(dvec), dvec, rscale,
                sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


class HelmholtzKernel(ExpressionKernel):
    init_arg_names = ("dim", "helmholtz_k_name", "allow_evanescent")

    def __init__(self, dim, helmholtz_k_name="k",
            allow_evanescent=False):
        """
        :arg helmholtz_k_name: The argument name to use for the Helmholtz
            parameter when generating functions to evaluate this kernel.
        """
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
                is_complex_valued=True)

        self.helmholtz_k_name = helmholtz_k_name
        self.allow_evanescent = allow_evanescent

    def __getinitargs__(self):
        return (self.dim, self.helmholtz_k_name,
                self.allow_evanescent)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.helmholtz_k_name,
            self.allow_evanescent))

    def __repr__(self):
        return f"HelmKnl{self.dim}D({self.helmholtz_k_name})"

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import register_bessel_callables
        return register_bessel_callables(loopy_knl)

    def get_args(self):
        if self.allow_evanescent:
            k_dtype = np.complex128
        else:
            k_dtype = np.float64

        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.helmholtz_k_name, k_dtype),
                    )]

    mapper_method = "map_helmholtz_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.tools.ExprDerivativeTaker` instance that supports
        taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.tools import HelmholtzDerivativeTaker
        return HelmholtzDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian

        w = make_identity_diff_op(self.dim)
        k = sym.Symbol(self.helmholtz_k_name)
        return (laplacian(w) + k**2 * w)


class YukawaKernel(ExpressionKernel):
    init_arg_names = ("dim", "yukawa_lambda_name")

    def __init__(self, dim, yukawa_lambda_name="lam"):
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
                is_complex_valued=True)

        self.yukawa_lambda_name = yukawa_lambda_name

    def __getinitargs__(self):
        return (self.dim, self.yukawa_lambda_name)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.yukawa_lambda_name))

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

    mapper_method = "map_yukawa_kernel"

    def get_derivative_taker(self, dvec, rscale, sac):
        """Return a :class:`sumpy.tools.ExprDerivativeTaker` instance that supports
        taking derivatives of the base kernel with respect to dvec.
        """
        from sumpy.tools import HelmholtzDerivativeTaker
        return HelmholtzDerivativeTaker(self.get_expression(dvec), dvec, rscale, sac)

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian
        w = make_identity_diff_op(self.dim)
        lam = sym.Symbol(self.yukawa_lambda_name)
        return (laplacian(w) - lam**2 * w)


class ElasticityKernel(ExpressionKernel):
    init_arg_names = ("dim", "icomp", "jcomp", "viscosity_mu", "poisson_ratio")

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
                +  # noqa: W504
                d[icomp]*d[jcomp]/r**2
                )
            scaling = -1/(8*var("pi")*(1 - nu)*mu)

        elif dim == 3:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            # Kelvin solution
            expr = (
                (1/r)*((3 - 4*nu) if icomp == jcomp else 0)
                +  # noqa: W504
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
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self.dim, self.icomp, self.jcomp, self.viscosity_mu,
                self.poisson_ratio)

    def __reduce__(self):
        return (ElasticityKernel, self.__getinitargs__())

    def update_persistent_hash(self, key_hash, key_builder):
        from pymbolic.mapper.persistent_hash import PersistentHashWalkMapper
        key_hash.update(type(self).__name__.encode())
        key_builder.rec(key_hash,
                (self.dim, self.icomp, self.jcomp))
        mapper = PersistentHashWalkMapper(key_hash)
        mapper(self.viscosity_mu)
        mapper(self.poisson_ratio)

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

    mapper_method = "map_elasticity_kernel"

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


class StokesletKernel(ElasticityKernel):
    def __new__(cls, dim, icomp, jcomp, viscosity_mu="mu", poisson_ratio="0.5"):
        return object.__new__(cls)

    def __init__(self, dim, icomp, jcomp, viscosity_mu="mu", poisson_ratio=0.5):
        super().__init__(dim, icomp, jcomp, viscosity_mu, poisson_ratio)

    def __repr__(self):
        return f"StokesletKnl{self.dim}D_{self.icomp}{self.jcomp}"


class StressletKernel(ExpressionKernel):
    init_arg_names = ("dim", "icomp", "jcomp", "kcomp", "viscosity_mu")

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

        self.icomp = icomp
        self.jcomp = jcomp
        self.kcomp = kcomp
        self.viscosity_mu = mu

        super().__init__(
                dim,
                expression=expr,
                global_scaling_const=scaling,
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self.dim, self.icomp, self.jcomp, self.kcomp, self.viscosity_mu)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode())
        key_builder.rec(key_hash, (self.dim, self.icomp, self.jcomp, self.kcomp))

        from pymbolic.mapper.persistent_hash import PersistentHashWalkMapper
        mapper = PersistentHashWalkMapper(key_hash)
        mapper(self.viscosity_mu)

    def __repr__(self):
        return f"StressletKnl{self.dim}D_{self.icomp}{self.jcomp}{self.kcomp}"

    @memoize_method
    def get_args(self):
        from sumpy.tools import get_all_variables
        variables = get_all_variables(self.viscosity_mu)
        return [
                KernelArgument(loopy_arg=lp.ValueArg(v.name, np.float64))
                for v in variables]

    mapper_method = "map_stresslet_kernel"

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian
        w = make_identity_diff_op(self.dim)
        return laplacian(laplacian(w))


class LineOfCompressionKernel(ExpressionKernel):
    """A kernel for the line of compression or dilatation of constant strength
    along the axis "axis" from zero to negative infinity. This is used for the
    explicit solution to half-space Elasticity problem. See [1] for details.

    [1]: Mindlin, R.: Force at a Point in the Interior of a Semi-Infinite Solid
         https://doi.org/10.1063/1.1745385
    """
    init_arg_names = ("dim", "axis", "viscosity_mu", "poisson_ratio")

    def __init__(self, dim=3, axis=2, viscosity_mu="mu", poisson_ratio="nu"):
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
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self.dim, self.axis, self.viscosity_mu, self.poisson_ratio)

    def update_persistent_hash(self, key_hash, key_builder):
        from pymbolic.mapper.persistent_hash import PersistentHashWalkMapper
        key_hash.update(type(self).__name__.encode())
        key_builder.rec(key_hash, (self.dim, self.axis))
        mapper = PersistentHashWalkMapper(key_hash)
        mapper(self.viscosity_mu)
        mapper(self.poisson_ratio)

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

    mapper_method = "map_line_of_compression_kernel"

    def get_pde_as_diff_op(self):
        from sumpy.expansion.diff_op import make_identity_diff_op, laplacian
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


# }}}


# {{{ a kernel defined as wrapping another one--e.g., derivatives

class KernelWrapper(Kernel):
    def __init__(self, inner_kernel):
        Kernel.__init__(self, inner_kernel.dim)
        self.inner_kernel = inner_kernel

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

class DerivativeBase(KernelWrapper):
    pass


class AxisSourceDerivative(DerivativeBase):
    init_arg_names = ("axis", "inner_kernel")

    def __init__(self, axis, inner_kernel):
        KernelWrapper.__init__(self, inner_kernel)
        self.axis = axis

    def __getinitargs__(self):
        return (self.axis, self.inner_kernel)

    def __str__(self):
        return f"d/dy{self.axis} {self.inner_kernel}"

    def __repr__(self):
        return f"AxisSourceDerivative({self.axis}, {self.inner_kernel!r})"

    def get_derivative_coeff_dict_at_source(self, expr_dict):
        expr_dict = self.inner_kernel.get_derivative_coeff_dict_at_source(
            expr_dict)
        result = dict()
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

    mapper_method = "map_axis_source_derivative"


class AxisTargetDerivative(DerivativeBase):
    init_arg_names = ("axis", "inner_kernel")
    target_array_name = "targets"

    def __init__(self, axis, inner_kernel):
        KernelWrapper.__init__(self, inner_kernel)
        self.axis = axis

    def __getinitargs__(self):
        return (self.axis, self.inner_kernel)

    def __str__(self):
        return f"d/dx{self.axis} {self.inner_kernel}"

    def __repr__(self):
        return f"AxisTargetDerivative({self.axis}, {self.inner_kernel!r})"

    def postprocess_at_target(self, expr, bvec):
        from sumpy.tools import (DifferentiatedExprDerivativeTaker,
                diff_derivative_coeff_dict)
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

    mapper_method = "map_axis_target_derivative"


class _VectorIndexAdder(CSECachingMapperMixin, IdentityMapper):
    def __init__(self, vec_name, additional_indices):
        self.vec_name = vec_name
        self.additional_indices = additional_indices

    def map_subscript(self, expr):
        from pymbolic.primitives import CommonSubexpression
        if expr.aggregate.name == self.vec_name \
                and isinstance(expr.index, int):
            return CommonSubexpression(expr.aggregate.index(
                    (expr.index,) + self.additional_indices))
        else:
            return IdentityMapper.map_subscript(self, expr)

    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression


class DirectionalDerivative(DerivativeBase):
    directional_kind: ClassVar[str]
    init_arg_names = ("inner_kernel", "dir_vec_name")

    def __init__(self, inner_kernel, dir_vec_name=None):
        if dir_vec_name is None:
            dir_vec_name = f"{self.directional_kind}_derivative_dir"

        KernelWrapper.__init__(self, inner_kernel)
        self.dir_vec_name = dir_vec_name

    def __getinitargs__(self):
        return (self.inner_kernel, self.dir_vec_name)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.inner_kernel)
        key_builder.rec(key_hash, self.dir_vec_name)

    def replace_base_kernel(self, new_base_kernel):
        return type(self)(self.inner_kernel.replace_base_kernel(new_base_kernel),
            dir_vec_name=self.dir_vec_name)

    def __str__(self):
        return r"{}·∇_{} {}".format(
                self.dir_vec_name,
                "y" if self.directional_kind == "src" else "x",
                self.inner_kernel)

    def __repr__(self):
        return "{}({!r}, {})".format(
                type(self).__name__,
                self.inner_kernel,
                self.dir_vec_name)


class DirectionalTargetDerivative(DirectionalDerivative):
    directional_kind = "tgt"
    target_array_name = "targets"

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
        from sumpy.tools import (DifferentiatedExprDerivativeTaker,
                diff_derivative_coeff_dict)

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

    def get_source_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.GlobalArg(
                        self.dir_vec_name,
                        None,
                        shape=(self.dim, "ntargets"),
                        dim_tags="sep,C",
                        offset=lp.auto),
                    )
                    ] + self.inner_kernel.get_source_args()

    mapper_method = "map_directional_target_derivative"


class DirectionalSourceDerivative(DirectionalDerivative):
    directional_kind = "src"

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
                        dim_tags="sep,C",
                        offset=lp.auto),
                    )
                    ] + self.inner_kernel.get_source_args()

    mapper_method = "map_directional_source_derivative"


class TargetPointMultiplier(KernelWrapper):
    """Wraps a kernel :math:`G(x, y)` and outputs :math:`x_j G(x, y)`
    where :math:`x, y` are targets and sources respectively.
    """

    init_arg_names = ("axis", "inner_kernel")
    target_array_name = "targets"

    def __init__(self, axis, inner_kernel):
        KernelWrapper.__init__(self, inner_kernel)
        self.axis = axis

    def __getinitargs__(self):
        return (self.axis, self.inner_kernel)

    def __str__(self):
        return f"x{self.axis} {self.inner_kernel}"

    def __repr__(self):
        return f"TargetPointMultiplier({self.axis}, {self.inner_kernel!r})"

    def replace_base_kernel(self, new_base_kernel):
        return type(self)(self.axis,
            self.inner_kernel.replace_base_kernel(new_base_kernel))

    def replace_inner_kernel(self, new_inner_kernel):
        return type(self)(self.axis, new_inner_kernel)

    def postprocess_at_target(self, expr, avec):
        from sumpy.symbolic import make_sym_vector as make_sympy_vector
        from sumpy.tools import (ExprDerivativeTaker,
            DifferentiatedExprDerivativeTaker)

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

    def get_code_transformer(self):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.target_array_name])
        from pymbolic.primitives import Variable
        via = _VectorIndexAdder(self.target_array_name, (Variable("itgt"),))

        inner_transform = self.inner_kernel.get_code_transformer()

        def transform(expr):
            return via(vcr(inner_transform(expr)))

        return transform

    mapper_method = "map_target_point_multiplier"

# }}}


# {{{ kernel mappers

class KernelMapper:
    def rec(self, kernel):
        try:
            method = getattr(self, kernel.mapper_method)
        except AttributeError:
            raise RuntimeError("{} cannot handle {}".format(
                type(self), type(kernel)))
        else:
            return method(kernel)

    __call__ = rec


class KernelCombineMapper(KernelMapper):
    def combine(self, values):
        raise NotImplementedError

    def map_difference_kernel(self, kernel):
        return self.combine([
                self.rec(kernel.kernel_plus),
                self.rec(kernel.kernel_minus)])

    def map_axis_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative
    map_axis_source_derivative = map_axis_target_derivative
    map_target_point_multiplier = map_axis_target_derivative


class KernelIdentityMapper(KernelMapper):
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


class DerivativeCounter(KernelCombineMapper):
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


def to_kernel_and_args(kernel_like):
    if (isinstance(kernel_like, tuple)
            and len(kernel_like) == 2
            and isinstance(kernel_like[0], Kernel)):
        # already gone through to_kernel_and_args
        return kernel_like

    if not isinstance(kernel_like, Kernel):
        if kernel_like == 0:
            return LaplaceKernel(None), {}
        elif isinstance(kernel_like, str):
            return HelmholtzKernel(None), {"k": var(kernel_like)}
        else:
            raise ValueError("Only Kernel instances, 0 (for Laplace) and "
                    "variable names (strings) "
                    "for the Helmholtz parameter are allowed as kernels.")

    return kernel_like, {}


# vim: fdm=marker
