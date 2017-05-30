from __future__ import division, absolute_import

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

from six.moves import range, zip

import loopy as lp
import numpy as np
from pymbolic.mapper import IdentityMapper, CSECachingMapperMixin
from sumpy.symbolic import pymbolic_real_norm_2
from pymbolic.primitives import make_sym_vector
from pymbolic import var

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
.. autoclass:: HelmholtzKernel
.. autoclass:: StokesletKernel
.. autoclass:: StressletKernel

Derivatives
-----------

These objects *wrap* other kernels and take derivatives
of them in the process.

.. autoclass:: DerivativeBase
.. autoclass:: AxisTargetDerivative
.. autoclass:: DirectionalTargetDerivative
.. autoclass:: DirectionalSourceDerivative

Transforming kernels
--------------------

.. autoclass:: KernelMapper
.. autoclass:: KernelCombineMapper
.. autoclass:: KernelIdentityMapper
.. autoclass:: AxisTargetDerivativeRemover
.. autoclass:: TargetDerivativeRemover
.. autoclass:: DerivativeCounter
.. autoclass:: KernelDimensionSetter
"""


class KernelArgument(object):
    """
    .. attribute:: loopy_arg

        A :class:`loopy.Argument` instance describing the type,
        name, and other features of this kernel argument when
        passed to a generated piece of code.
    """

    def __init__(self, loopy_arg):
        self.loopy_arg = loopy_arg

    @property
    def name(self):
        return self.loopy_arg.name


# {{{ basic kernel interface

class Kernel(object):
    """Basic kernel interface.

    .. attribute:: is_complex_valued
    .. attribute:: dim

        *dim* is allowed to be *None* if the dimensionality is not yet
        known.
    """

    def __init__(self, dim=None):
        self._dim = dim

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

    def __getstate__(self):
        return self.__getinitargs__()

    def __setstate__(self, state):
        # Can't use trivial pickling: hash_value cache must stay unset
        assert len(self.init_arg_names) == len(state)
        for name, value in zip(self.init_arg_names, state):
            if name == "dim":
                name = "_dim"

            setattr(self, name, value)

    # }}}

    @property
    def dim(self):
        if self._dim is None:
            raise RuntimeError("the number of dimensions for this kernel "
                    "has not yet been set")

        return self._dim

    def get_base_kernel(self):
        """Return the kernel being wrapped by this one, or else
        *self*.
        """
        return self

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
        """Return a :mod:`sympy` expression for the kernel.

        :arg dist_vec: target - source

        (Assumes translation invariance of the kernel.)
        """
        raise NotImplementedError

    def postprocess_at_source(self, expr, avec):
        """Transform a kernel evaluation or expansion expression in a place
        where the vector a (something - source) is known. ("something" may be
        an expansion center or a target.)
        """
        return expr

    def postprocess_at_target(self, expr, bvec):
        """Transform a kernel evaluation or expansion expression in a place
        where the vector b (target - something) is known. ("something" may be
        an expansion center or a target.)
        """
        return expr

    def get_scaling(self):
        """Return a global scaling of the kernel."""
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

# }}}


class ExpressionKernel(Kernel):
    is_complex_valued = False

    init_arg_names = ("dim", "expression", "scaling", "is_complex_valued")

    def __init__(self, dim, expression, scaling, is_complex_valued):
        """
        :arg expression: A :mod:`pymbolic` expression depending on
            variables *d_1* through *d_N* where *N* equals *dim*.
            (These variables match what is returned from
            :func:`pymbolic.primitives.make_sym_vector` with
            argument `"d"`.)
        :arg scaling: A :mod:`pymbolic` expression for the scaling
            of the kernel.
        """

        # expression and scaling are pymbolic objects because those pickle
        # cleanly. D'oh, sympy!

        Kernel.__init__(self, dim)

        self.expression = expression
        self.scaling = scaling
        self.is_complex_valued = is_complex_valued

    def __getinitargs__(self):
        return (self._dim, self.expression, self.scaling,
                self.is_complex_valued)

    def __repr__(self):
        if self._dim is not None:
            return "ExprKnl%dD" % self.dim
        else:
            return "ExprKnl"

    def get_expression(self, dist_vec):
        if self.expression is None:
            raise RuntimeError("expression in ExpressionKernel has not "
                    "been determined yet (this could be due to a PDE kernel "
                    "not having learned its dimensionality yet)")

        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        expr = PymbolicToSympyMapperWithSymbols()(self.expression)

        if self.dim != len(dist_vec):
            raise ValueError("dist_vec length does not match expected dimension")

        from sumpy.symbolic import Symbol
        expr = expr.subs(dict(
            (Symbol("d%d" % i), dist_vec_i)
            for i, dist_vec_i in enumerate(dist_vec)
            ))

        return expr

    def get_scaling(self):
        """Return a global scaling of the kernel."""

        if self.scaling is None:
            raise RuntimeError("scaling in ExpressionKernel has not "
                    "been determined yet (this could be due to a PDE kernel "
                    "not having learned its dimensionality yet)")

        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        return PymbolicToSympyMapperWithSymbols()(self.scaling)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        for name, value in zip(self.init_arg_names, self.__getinitargs__()):
            if name in ["expression", "scaling"]:
                from pymbolic.mapper.persistent_hash import (
                        PersistentHashWalkMapper as PersistentHashWalkMapper)
                PersistentHashWalkMapper(key_hash)(value)
            else:
                key_builder.rec(key_hash, value)

    mapper_method = "map_expression_kernel"


one_kernel_2d = ExpressionKernel(
        dim=2,
        expression=1,
        scaling=1,
        is_complex_valued=False)
one_kernel_3d = ExpressionKernel(
        dim=3,
        expression=1,
        scaling=1,
        is_complex_valued=False)


# {{{ PDE kernels

class LaplaceKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim=None):
        # See (Kress LIE, Thm 6.2) for scaling
        if dim == 2:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = var("log")(r)
            scaling = 1/(-2*var("pi"))
        elif dim == 3:
            r = pymbolic_real_norm_2(make_sym_vector("d", dim))
            expr = 1/r
            scaling = 1/(4*var("pi"))
        elif dim is None:
            expr = None
            scaling = None
        else:
            raise RuntimeError("unsupported dimensionality")

        ExpressionKernel.__init__(
                self,
                dim,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self._dim,)

    def __repr__(self):
        if self._dim is not None:
            return "LapKnl%dD" % self.dim
        else:
            return "LapKnl"

    mapper_method = "map_laplace_kernel"


class BiharmonicKernel(ExpressionKernel):
    init_arg_names = ("dim",)

    def __init__(self, dim=None):
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        if dim == 2:
            expr = r**2 * var("log")(r)
            scaling = 1/(8*var("pi"))
        elif dim == 3:
            expr = r
            scaling = 1  # FIXME: Unknown
        else:
            raise RuntimeError("unsupported dimensionality")

        ExpressionKernel.__init__(
                self,
                dim,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self._dim,)

    def __repr__(self):
        if self._dim is not None:
            return "BiharmKnl%dD" % self.dim
        else:
            return "BiharmKnl"

    mapper_method = "map_biharmonic_kernel"


class HelmholtzKernel(ExpressionKernel):
    init_arg_names = ("dim", "helmholtz_k_name", "allow_evanescent")

    def __init__(self, dim=None, helmholtz_k_name="k",
            allow_evanescent=False):
        """
        :arg helmholtz_k_name: The argument name to use for the Helmholtz
            parameter when generating functions to evaluate this kernel.
        """
        k = var(helmholtz_k_name)

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
        elif dim is None:
            expr = None
            scaling = None
        else:
            raise RuntimeError("unsupported dimensionality")

        ExpressionKernel.__init__(
                self,
                dim,
                expression=expr,
                scaling=scaling,
                is_complex_valued=True)

        self.helmholtz_k_name = helmholtz_k_name
        self.allow_evanescent = allow_evanescent

    def __getinitargs__(self):
        return (self._dim, self.helmholtz_k_name,
                self.allow_evanescent)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.helmholtz_k_name,
            self.allow_evanescent))

    def __repr__(self):
        if self._dim is not None:
            return "HelmKnl%dD(%s)" % (
                    self.dim, self.helmholtz_k_name)
        else:
            return "HelmKnl(%s)" % (self.helmholtz_k_name)

    def prepare_loopy_kernel(self, loopy_knl):
        from sumpy.codegen import (bessel_preamble_generator, bessel_mangler)
        loopy_knl = lp.register_function_manglers(loopy_knl,
                [bessel_mangler])
        loopy_knl = lp.register_preamble_generators(loopy_knl,
                [bessel_preamble_generator])

        return loopy_knl

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


class StokesletKernel(ExpressionKernel):
    init_arg_names = ("dim", "icomp", "jcomp", "viscosity_mu_name")

    def __init__(self, dim, icomp, jcomp, viscosity_mu_name="mu"):
        r"""
        :arg viscosity_mu_name: The argument name to use for
                dynamic viscosity :math:`\mu` the then generating functions to
                evaluate this kernel.
        """
        mu = var(viscosity_mu_name)

        if dim == 2:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            expr = (
                -var("log")(r)*(1 if icomp == jcomp else 0)
                +
                d[icomp]*d[jcomp]/r**2
                )
            scaling = -1/(4*var("pi")*mu)

        elif dim == 3:
            d = make_sym_vector("d", dim)
            r = pymbolic_real_norm_2(d)
            expr = (
                (1/r)*(1 if icomp == jcomp else 0)
                +
                d[icomp]*d[jcomp]/r**3
                )
            scaling = -1/(8*var("pi")*mu)

        elif dim is None:
            expr = None
            scaling = None
        else:
            raise RuntimeError("unsupported dimensionality")

        self.viscosity_mu_name = viscosity_mu_name
        self.icomp = icomp
        self.jcomp = jcomp

        ExpressionKernel.__init__(
                self,
                dim,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self._dim, self.icomp, self.jcomp, self.viscosity_mu_name)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode())
        key_builder.rec(key_hash,
                (self.dim, self.icomp, self.jcomp, self.viscosity_mu_name))

    def __repr__(self):
        return "StokesletKnl%dD_%d%d" % (self.dim, self.icomp, self.jcomp)

    def get_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.viscosity_mu_name, np.float64),
                    )]

    mapper_method = "map_stokeslet_kernel"


class StressletKernel(ExpressionKernel):
    init_arg_names = ("dim", "icomp", "jcomp", "kcomp", "viscosity_mu_name")

    def __init__(self, dim=None, icomp=None, jcomp=None, kcomp=None,
                        viscosity_mu_name="mu"):
        r"""
        :arg viscosity_mu_name: The argument name to use for
                dynamic viscosity :math:`\mu` the then generating functions to
                evaluate this kernel.
        """
        # Mu is unused but kept for consistency with the stokeslet.

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

        elif dim is None:
            expr = None
            scaling = None
        else:
            raise RuntimeError("unsupported dimensionality")

        self.viscosity_mu_name = viscosity_mu_name
        self.icomp = icomp
        self.jcomp = jcomp
        self.kcomp = kcomp

        ExpressionKernel.__init__(
                self,
                dim,
                expression=expr,
                scaling=scaling,
                is_complex_valued=False)

    def __getinitargs__(self):
        return (self._dim, self.icomp, self.jcomp, self.kcomp,
                      self.viscosity_mu_name)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode())
        key_builder.rec(key_hash, (
            self.dim, self.icomp, self.jcomp, self.kcomp,
            self.viscosity_mu_name))

    def __repr__(self):
        return "StressletKnl%dD_%d%d%d" % (self.dim, self.icomp, self.jcomp,
                self.kcomp)

    def get_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.ValueArg(self.viscosity_mu_name, np.float64),
                    )
                ]

    mapper_method = "map_stresslet_kernel"

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

    def get_expression(self, dist_vec):
        return self.inner_kernel.get_expression(dist_vec)

    def postprocess_at_source(self, expr, avec):
        return self.inner_kernel.postprocess_at_source(expr, avec)

    def postprocess_at_target(self, expr, avec):
        return self.inner_kernel.postprocess_at_target(expr, avec)

    def get_scaling(self):
        return self.inner_kernel.get_scaling()

    def get_code_transformer(self):
        return self.inner_kernel.get_code_transformer()

    def get_args(self):
        return self.inner_kernel.get_args()

    def get_source_args(self):
        return self.inner_kernel.get_source_args()

# }}}


# {{{ derivatives

class DerivativeBase(KernelWrapper):
    pass


class AxisTargetDerivative(DerivativeBase):
    init_arg_names = ("axis", "inner_kernel")

    def __init__(self, axis, inner_kernel):
        KernelWrapper.__init__(self, inner_kernel)
        self.axis = axis

    def __getinitargs__(self):
        return (self.axis, self.inner_kernel)

    def __str__(self):
        return "d/dx%d %s" % (self.axis, self.inner_kernel)

    def __repr__(self):
        return "AxisTargetDerivative(%d, %r)" % (self.axis, self.inner_kernel)

    def postprocess_at_target(self, expr, bvec):
        expr = self.inner_kernel.postprocess_at_target(expr, bvec)
        return expr.diff(bvec[self.axis])

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
    init_arg_names = ("inner_kernel", "dir_vec_name")

    def __init__(self, inner_kernel, dir_vec_name=None):
        if dir_vec_name is None:
            dir_vec_name = self.directional_kind + "_derivative_dir"
        else:
            from warnings import warn
            warn("specified the name of the direction vector",
                    stacklevel=2)

        KernelWrapper.__init__(self, inner_kernel)
        self.dir_vec_name = dir_vec_name

    def __getinitargs__(self):
        return (self.inner_kernel, self.dir_vec_name)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.inner_kernel)
        key_builder.rec(key_hash, self.dir_vec_name)

    def __str__(self):
        return r"%s . \/_%s %s" % (
                self.dir_vec_name, self.directional_kind[0], self.inner_kernel)

    def __repr__(self):
        return "%s(%r, %s)" % (
                type(self).__name__,
                self.inner_kernel,
                self.dir_vec_name)

    def get_source_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.GlobalArg(
                        self.dir_vec_name,
                        None,
                        shape=(self.dim, "nsources"),
                        dim_tags="sep,C"),
                    )
                    ] + self.inner_kernel.get_source_args()


class DirectionalTargetDerivative(DirectionalDerivative):
    directional_kind = "tgt"

    def get_code_transformer(self):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.dir_vec_name])
        from pymbolic.primitives import Variable
        via = _VectorIndexAdder(self.dir_vec_name, (Variable("itgt"),))

        def transform(expr):
            return via(vcr(expr))

        return transform

    def postprocess_at_target(self, expr, bvec):
        expr = self.inner_kernel.postprocess_at_target(expr, bvec)

        dim = len(bvec)
        assert dim == self.dim

        from sumpy.symbolic import make_sym_vector as make_sympy_vector
        dir_vec = make_sympy_vector(self.dir_vec_name, dim)

        # bvec = tgt-center
        return sum(dir_vec[axis]*expr.diff(bvec[axis])
                for axis in range(dim))

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

    def postprocess_at_source(self, expr, avec):
        expr = self.inner_kernel.postprocess_at_source(expr, avec)

        dimensions = len(avec)
        assert dimensions == self.dim

        from sumpy.symbolic import make_sym_vector as make_sympy_vector
        dir_vec = make_sympy_vector(self.dir_vec_name, dimensions)

        # avec = center-src -> minus sign from chain rule
        return sum(-dir_vec[axis]*expr.diff(avec[axis])
                for axis in range(dimensions))

    mapper_method = "map_directional_source_derivative"

# }}}


# {{{ kernel mappers

class KernelMapper(object):
    def rec(self, kernel):
        try:
            method = getattr(self, kernel.mapper_method)
        except AttributeError:
            raise RuntimeError("%s cannot handle %s" % (
                type(self), type(kernel)))
        else:
            return method(kernel)

    __call__ = rec


class KernelCombineMapper(KernelMapper):
    def map_difference_kernel(self, kernel):
        return self.combine([
                self.rec(kernel.kernel_plus),
                self.rec(kernel.kernel_minus)])

    def map_axis_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative


class KernelIdentityMapper(KernelMapper):
    def map_expression_kernel(self, kernel):
        return kernel

    map_laplace_kernel = map_expression_kernel
    map_biharmonic_kernel = map_expression_kernel
    map_helmholtz_kernel = map_expression_kernel
    map_stokeslet_kernel = map_expression_kernel
    map_stresslet_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel):
        return AxisTargetDerivative(kernel.axis, self.rec(kernel.inner_kernel))

    def map_directional_target_derivative(self, kernel):
        return type(kernel)(
                self.rec(kernel.inner_kernel),
                kernel.dir_vec_name)

    map_directional_source_derivative = map_directional_target_derivative


class AxisTargetDerivativeRemover(KernelIdentityMapper):
    def map_axis_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)


class TargetDerivativeRemover(AxisTargetDerivativeRemover):
    def map_directional_target_derivative(self, kernel):
        return self.rec(kernel.inner_kernel)


class DerivativeCounter(KernelCombineMapper):
    def combine(self, values):
        return max(values)

    def map_expression_kernel(self, kernel):
        return 0

    map_laplace_kernel = map_expression_kernel
    map_biharmonic_kernel = map_expression_kernel
    map_helmholtz_kernel = map_expression_kernel
    map_stokeslet_kernel = map_expression_kernel
    map_stresslet_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel):
        return 1 + self.rec(kernel.inner_kernel)

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative


class KernelDimensionSetter(KernelIdentityMapper):
    """Deprecated: This is no longer used and will be removed in 2018.
    """

    def __init__(self, dim):
        self.dim = dim

    def map_expression_kernel(self, kernel):
        if kernel._dim is not None and self.dim != kernel.dim:
            raise RuntimeError("cannot set kernel dimension to new value (%d)"
                    "different from existing one (%d)"
                    % (self.dim, kernel.dim))

        return kernel

    def map_laplace_kernel(self, kernel):
        if kernel._dim is not None and self.dim != kernel.dim:
            raise RuntimeError("cannot set kernel dimension to new value (%d)"
                    "different from existing one (%d)"
                    % (self.dim, kernel.dim))

        return LaplaceKernel(self.dim)

    def map_helmholtz_kernel(self, kernel):
        if kernel._dim is not None and self.dim != kernel.dim:
            raise RuntimeError("cannot set kernel dimension to new value (%d)"
                    "different from existing one (%d)"
                    % (self.dim, kernel.dim))

        return HelmholtzKernel(self.dim,
                helmholtz_k_name=kernel.helmholtz_k_name,
                allow_evanescent=kernel.allow_evanescent)

    def map_stokeslet_kernel(self, kernel):
        if kernel._dim is not None and self.dim != kernel.dim:
            raise RuntimeError("cannot set kernel dimension to new value (%d)"
                    "different from existing one (%d)"
                    % (self.dim, kernel.dim))

        return StokesletKernel(self.dim,
                kernel.icomp,
                kernel.jcomp,
                viscosity_mu_name=kernel.viscosity_mu_name)

    def map_stresslet_kernel(self, kernel):
        if kernel._dim is not None and self.dim != kernel.dim:
            raise RuntimeError("cannot set kernel dimension to new value (%d)"
                    "different from existing one (%d)"
                    % (self.dim, kernel.dim))

        return StressletKernel(self.dim,
                kernel.icomp,
                kernel.jcomp,
                kernel.kcomp,
                viscosity_mu_name=kernel.viscosity_mu_name)

# }}}


def to_kernel_and_args(kernel_like):
    if (isinstance(kernel_like, tuple)
            and len(kernel_like) == 2
            and isinstance(kernel_like[0], Kernel)):
        # already gone through to_kernel_and_args
        return kernel_like

    if not isinstance(kernel_like, Kernel):
        if kernel_like == 0:
            return LaplaceKernel(), {}
        elif isinstance(kernel_like, str):
            return HelmholtzKernel(None), {"k": var(kernel_like)}
        else:
            raise ValueError("Only Kernel instances, 0 (for Laplace) and "
                    "variable names (strings) "
                    "for the Helmholtz parameter are allowed as kernels.")

    return kernel_like, {}


# vim: fdm=marker
