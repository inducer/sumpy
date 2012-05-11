from __future__ import division

import loopy as lp
import sympy as sp
import numpy as np
from pymbolic.mapper import IdentityMapper




class Kernel:
    """Basic kernel interface.

    :ivar is_complex:
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions

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
        """Return a :mod:`pymbolic` expression for the kernel.

        :arg dist_vec: target - source

        (Assumes translation invariance of the kernel.)
        """
        raise NotImplementedError

    def postprocess_expression(self, expr, src_location, tgt_location):
        return expr

    def get_scaling(self):
        """Return a global scaling of the kernel."""
        raise NotImplementedError

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
        from sumpy.symbolic import sympy_real_norm_2
        r = sympy_real_norm_2(dist_vec)

        if self.dimensions == 2:
            return sp.log(r)
        elif self.dimensions == 3:
            return 1/r
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_scaling(self):
        """Return a global scaling of the kernel."""

        if self.dimensions == 2:
            return 1/(-2*sp.pi)
        elif self.dimensions == 3:
            return 1/(4*sp.pi)
        else:
            raise RuntimeError("unsupported dimensionality")



class HelmholtzKernel(Kernel):
    def __init__(self, dimensions, allow_evanescent=False):
        Kernel.__init__(self, dimensions)
        self.allow_evanescent = allow_evanescent

    is_complex = True

    def prepare_loopy_kernel(self, loopy_knl):
        # does loopy_knl already know about hank1_01?
        mangle_result = loopy_knl.mangle_function(
                "hank1_01", (np.dtype(np.complex128),))
        from sumpy.codegen import hank1_01_result_dtype, bessel_mangler
        if mangle_result is not hank1_01_result_dtype:
            return loopy_knl.register_function_mangler(bessel_mangler)
        else:
            return loopy_knl

    def get_expression(self, dist_vec):
        assert self.dimensions == len(dist_vec)

        from sumpy.symbolic import sympy_real_norm_2
        r = sympy_real_norm_2(dist_vec)

        k = sp.Symbol("k")

        if self.dimensions == 2:
            return sp.Function("hankel_1")(0, k*r)
        elif self.dimensions == 3:
            return sp.exp(sp.I*k*r)/r
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_scaling(self):
        """Return a global scaling of the kernel."""

        if self.dimensions == 2:
            return sp.I/4
        elif self.dimensions == 3:
            return 1/(4*sp.pi)
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_args(self):
        if self.allow_evanescent:
            k_dtype = np.complex128
        else:
            k_dtype = np.float64

        return [lp.ScalarArg("k", k_dtype)]

    def get_preambles(self):
        from sumpy.codegen import BESSEL_PREAMBLE
        return [("sumpy-bessel", BESSEL_PREAMBLE)]

# }}}

class KernelWrapper(Kernel):
    def __init__(self, kernel):
        Kernel.__init__(self, kernel.dimensions)
        self.kernel = kernel

    def prepare_loopy_kernel(self, loopy_knl):
        return self.kernel.prepare_loopy_kernel(loopy_knl)

    @property
    def is_complex(self):
        return self.kernel.is_complex

    def get_expression(self, dist_vec):
        return self.kernel.get_expression(dist_vec)

    def postprocess_expression(self, expr, avec, bvec):
        return self.kernel.postprocess_expression(expr, avec, bvec)

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

    def postprocess_expression(self, expr, avec, bvec):
        expr = self.kernel.postprocess_expression(expr, avec, bvec)
        return expr.diff(bvec[self.axis])




class _SourceDerivativeToCodeMapper(IdentityMapper):
    def __init__(self, vec_name):
        self.vec_name = vec_name

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable, CommonSubexpression
        if expr.aggregate.name == self.vec_name and isinstance(expr.index, int):
            return CommonSubexpression(expr.aggregate[
                    (Variable("isrc"), expr.index)])
        else:
            return IdentityMapper.map_subscript(self, expr)




class SourceDerivative(KernelWrapper):
    def __init__(self, kernel, dir_vec_name="src_derivative_dir", dir_vec_dtype=np.float64):
        KernelWrapper.__init__(self, kernel)
        self.dir_vec_name = dir_vec_name
        self.dir_vec_dtype = dir_vec_dtype

    def transform_to_code(self, expr):
        from sumpy.codegen import VectorComponentRewriter
        vcr = VectorComponentRewriter([self.dir_vec_name])
        return _SourceDerivativeToCodeMapper(self.dir_vec_name)(
                vcr(self.kernel.transform_to_code(expr)))

    def postprocess_expression(self, expr, avec, bvec):
        dimensions = len(avec)
        assert dimensions == self.dimensions

        from sumpy.symbolic import make_sym_vector
        dir_vec = make_sym_vector(self.dir_vec_name, dimensions)

        # avec = center-src -> minus sign from chain rule
        return sum(-dir_vec[axis]*expr.diff(avec[axis])
                for axis in range(dimensions))

    def get_args(self):
        return self.kernel.get_args() + [
            lp.GlobalArg(self.dir_vec_name, self.dir_vec_dtype,
                shape=("nsrc", self.dimensions), order="C")]

# }}}

# vim: fdm=marker
