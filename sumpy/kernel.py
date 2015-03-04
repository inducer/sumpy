from __future__ import division

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


import loopy as lp
import numpy as np
from pymbolic.mapper import IdentityMapper
from sumpy.symbolic import pymbolic_real_norm_2
from pymbolic.primitives import make_sym_vector
from pymbolic import var


class KernelArgument(object):
    """
    .. attribute:: loopy_arg

        A :class:`loopy.Argument` instance describing the type,
        name, and other features of this kernel argument when
        passed to a generated piece of code.

    .. attribute:: expression

        A :class:`pymbolic.primitives.Expression` instance that
        should be evaluted to obtain the value of this argument
        when this kernel is used in symbolic evaluation, such
        as in :mod:`pytential.
    """

    def __init__(self, loopy_arg, expression=None):
        self.loopy_arg = loopy_arg
        self.expression = expression

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

    def transform_to_code(self, expr):
        """Postprocess the :mod:`pymbolic` expression
        generated from the result of :meth:`get_expression`
        on the way to code generation.
        """
        return expr

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
        """Return list of :cls:`KernelArgument` instances describing
        extra arguments used by the kernel.
        """
        return []

    def get_source_args(self):
        """Return list of :cls:`KernelArgument` instances describing
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
        return (self._dim,)

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

        expr = expr.subs([
            ("d_%d" % i, dist_vec_i)
            for i, dist_vec_i in enumerate(dist_vec)
            ])

        return expr

    def get_scaling(self):
        """Return a global scaling of the kernel."""

        if self.scaling is None:
            raise RuntimeError("scaling in ExpressionKernel has not "
                    "been determined yet (this could be due to a PDE kernel "
                    "not having learned its dimensionality yet)")

        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols
        return PymbolicToSympyMapperWithSymbols()(self.scaling)

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


class HelmholtzKernel(ExpressionKernel):
    init_arg_names = ("dim", "helmholtz_k_name", "helmholtz_k_expression",
            "allow_evanescent")

    def __init__(self, dim=None, helmholtz_k_name="k",
            helmholtz_k_expression=None,
            allow_evanescent=False):
        """
        :arg helmholtz_k_name: The argument name to use for the Helmholtz
            parameter when generating functions to evaluate this kernel.
        :arg helmholtz_k_expression: The expression to evaluate to find
            the Helmholtz parameter when evaluating this kernel as part
            of a :mod:`pymbolic` operator.
        """
        k = var(helmholtz_k_name)

        # Guard against code using the old positional interface.
        assert not isinstance(helmholtz_k_expression, bool)

        if helmholtz_k_expression is None:
            helmholtz_k_expression = k

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
        self.helmholtz_k_expression = helmholtz_k_expression
        self.allow_evanescent = allow_evanescent

    def __getinitargs__(self):
        return (self._dim, self.helmholtz_k_name,
                self.helmholtz_k_expression, self.allow_evanescent)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.helmholtz_k_name,
            self.allow_evanescent))

        # pymbolic expressions don't fit into persistent hashes:
        # omit helmholtz_k_expression

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
                    expression=self.helmholtz_k_expression)]

    mapper_method = "map_helmholtz_kernel"

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

    def transform_to_code(self, expr):
        return self.inner_kernel.transform_to_code(expr)

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

    def postprocess_at_target(self, expr, bvec):
        expr = self.inner_kernel.postprocess_at_target(expr, bvec)
        return expr.diff(bvec[self.axis])

    mapper_method = "map_axis_target_derivative"


class _VectorIndexAdder(IdentityMapper):
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


class DirectionalDerivative(DerivativeBase):
    init_arg_names = ("inner_kernel", "dir_vec_name", "dir_vec_expression")

    def __init__(self, inner_kernel, dir_vec_name=None, dir_vec_expression=None):
        """
        :arg dir_vec_expression: an object of unspecified type, available for
            use by client applications to store information about the
            direction vector. :mod:`pytential` for instance uses this
            to store the direction vector expression to be evaluated.
        """

        if dir_vec_name is None:
            dir_vec_name = self.directional_kind + "_derivative_dir"

        KernelWrapper.__init__(self, inner_kernel)
        self.dir_vec_name = dir_vec_name
        self.dir_vec_expression = dir_vec_expression

    def __getinitargs__(self):
        dir_vec_expression = self.dir_vec_expression

        # for hashability
        if isinstance(dir_vec_expression, np.ndarray):
            dir_vec_expression = tuple(dir_vec_expression)

        return (self.inner_kernel, self.dir_vec_name, dir_vec_expression)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.inner_kernel)
        key_builder.rec(key_hash, self.dir_vec_name)

        # pymbolic expressions don't fit into persistent hashes:
        # omit dir_vec_expression.

    def __str__(self):
        return r"%s . \/_%s %s" % (
                self.dir_vec_name, self.directional_kind[0], self.inner_kernel)

    def get_source_args(self):
        return [
                KernelArgument(
                    loopy_arg=lp.GlobalArg(
                        self.dir_vec_name,
                        None,
                        shape=(self.dim, "nsources"),
                        dim_tags="sep,C"),
                    expression=self.dir_vec_expression)
                    ] + self.inner_kernel.get_source_args()


class DirectionalTargetDerivative(DirectionalDerivative):
    directional_kind = "tgt"

    def transform_to_code(self, expr):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.dir_vec_name])
        from pymbolic.primitives import Variable
        return _VectorIndexAdder(self.dir_vec_name, (Variable("itgt"),))(
                vcr(self.inner_kernel.transform_to_code(expr)))

    def postprocess_at_target(self, expr, bvec):
        expr = self.inner_kernel.postprocess_at_target(expr, bvec)

        dim = len(bvec)
        assert dim == self.dim

        from sumpy.symbolic import make_sympy_vector
        dir_vec = make_sympy_vector(self.dir_vec_name, dim)

        # bvec = tgt-center
        return sum(dir_vec[axis]*expr.diff(bvec[axis])
                for axis in range(dim))

    mapper_method = "map_directional_target_derivative"


class DirectionalSourceDerivative(DirectionalDerivative):
    directional_kind = "src"

    def transform_to_code(self, expr):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.dir_vec_name])
        from pymbolic.primitives import Variable
        return _VectorIndexAdder(self.dir_vec_name, (Variable("isrc"),))(
                vcr(self.inner_kernel.transform_to_code(expr)))

    def postprocess_at_source(self, expr, avec):
        expr = self.inner_kernel.postprocess_at_source(expr, avec)

        dimensions = len(avec)
        assert dimensions == self.dim

        from sumpy.symbolic import make_sympy_vector
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
    map_helmholtz_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel):
        return AxisTargetDerivative(kernel.axis, self.rec(kernel.inner_kernel))

    def map_directional_target_derivative(self, kernel):
        return type(kernel)(
                self.rec(kernel.inner_kernel),
                kernel.dir_vec_name, kernel.dir_vec_expression)

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
    map_helmholtz_kernel = map_expression_kernel

    def map_axis_target_derivative(self, kernel):
        return 1 + self.rec(kernel.inner_kernel)

    map_directional_target_derivative = map_axis_target_derivative
    map_directional_source_derivative = map_axis_target_derivative


class KernelDimensionSetter(KernelIdentityMapper):
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
                helmholtz_k_expression=(
                    kernel.helmholtz_k_expression),
                allow_evanescent=kernel.allow_evanescent)

# }}}


def normalize_kernel(kernel_like):
    if not isinstance(kernel_like, Kernel):
        if kernel_like == 0:
            kernel_like = LaplaceKernel()
        elif isinstance(kernel_like, str):
            kernel_like = HelmholtzKernel(None, kernel_like)
        else:
            raise ValueError("Only Kernel instances, 0 (for Laplace) and "
                    "variable names (strings) "
                    "for the Helmholtz parameter are allowed as kernels.")

    return kernel_like


# vim: fdm=marker
