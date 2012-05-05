from __future__ import division

import loopy as lp
import numpy as np
from pymbolic import var
from pymbolic.mapper import IdentityMapper




class Kernel:
    """Basic kernel interface."""

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def get_expression(self, dist_vec):
        """Return a :mod:`pymbolic` expression for the kernel.

        :arg dist_vec: target - source

        (Assumes translation invariance of the kernel.)
        """
        raise NotImplementedError

    def get_scaling(self):
        """Return a global scaling of the kernel."""
        raise NotImplementedError

    def transform_to_code(self, expr):
        """Postprocess the :mod:`pymbolic` expression
        generated from the result of :meth:`get_expression`
        on the way to code generation.
        """
        return expr

    def get_args(self):
        """Return list of :cls:`loopy.Argument` instances describing 
        extra arguments used by kernel.
        """
        return []

    def get_preambles(self):
        return []




# {{{ PDE kernels

class LaplaceKernel(Kernel):
    is_complex = False

    def get_expression(self, dist_vec):
        assert self.dimensions == len(dist_vec)
        r = var("sqrt")(np.dot(dist_vec, dist_vec))

        if self.dimensions == 2:
            return var("log")(r)
        elif self.dimensions == 3:
            return 1/r
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_scaling(self):
        """Return a global scaling of the kernel."""

        if self.dimensions == 2:
            return 1/(-2*var("pi"))
        elif self.dimensions == 3:
            return 1/(4*var("pi"))
        else:
            raise RuntimeError("unsupported dimensionality")



class HelmholtzKernel(Kernel):
    is_complex = True

    def get_expression(self, dist_vec):
        assert self.dimensions == len(dist_vec)

        r = var("sqrt")(np.dot(dist_vec, dist_vec))

        k = var("k")

        if self.dimensions == 2:
            return var("hankel_1")(0, k*r)
        elif self.dimensions == 3:
            return var("exp")(1j*k*r)/r
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_scaling(self):
        """Return a global scaling of the kernel."""

        if self.dimensions == 2:
            return 1j/4
        elif self.dimensions == 3:
            return 1/(4*var("pi"))
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_args(self):
        return [
                lp.ScalarArg("k", np.complex128),
                ]

    def get_preambles(self):
        from sumpy.codegen import HANKEL_PREAMBLE
        return [("sumpy-hankel", HANKEL_PREAMBLE)]

# }}}

class KernelWrapper(Kernel):
    def __init__(self, kernel):
        Kernel.__init__(self, kernel.dimensions)
        self.kernel = kernel

    @property
    def is_complex(self):
        return self.kernel.is_complex

    def get_scaling(self):
        return self.kernel.get_scaling()

    def transform_to_code(self, expr):
        return self.kernel.transform_to_code(expr)

    def get_args(self):
        return self.kernel.get_args()

    def get_preambles(self):
        return self.kernel.get_preambles()

# {{{ derivatives

class TargetDerivative(KernelWrapper):
    def __init__(self, axis, kernel):
        KernelWrapper.__init__(self, kernel)
        self.axis = axis

    def get_expression(self, dist_vec):
        from pymbolic.maxima import diff
        return diff(self.kernel.get_expression(dist_vec), dist_vec[self.axis])




class _SourceDerivativeToCodeMapper(IdentityMapper):
    def __init__(self, vec_name):
        self.vec_name = vec_name

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        if expr.aggregate.name == self.vec_name:
            return expr.aggregate[
                    (Variable("isrc"), expr.index)]
        else:
            return IdentityMapper.map_subscript(self, expr)




class SourceDerivative(KernelWrapper):
    def __init__(self, kernel, dir_vec_name="src_derivative_dir", dir_vec_dtype=np.float64):
        KernelWrapper.__init__(self, kernel)
        self.dir_vec_name = dir_vec_name
        self.dir_vec_dtype = dir_vec_dtype

    def get_expression(self, dist_vec):
        dimensions = len(dist_vec)
        assert dimensions == self.dimensions

        knl = self.kernel.get_expression(dist_vec)
        from pymbolic.primitives import make_sym_vector
        dir_vec = make_sym_vector(self.dir_vec_name, dimensions)

        # dist_vec = tgt-src -> minus sign from chain rule
        from pymbolic.maxima import diff
        return sum(-dir_vec[axis]*diff(knl, dist_vec[axis])
                for axis in range(dimensions))

    def transform_to_code(self, expr):
        return _SourceDerivativeToCodeMapper(self.dir_vec_name)(
                self.kernel.transform_to_code(expr))

    def get_args(self):
        return self.kernel.get_args() + [
            lp.GlobalArg(self.dir_vec_name, self.dir_vec_dtype,
                shape=("nsrc", self.dimensions), order="C")]

# }}}

# vim: fdm=marker
