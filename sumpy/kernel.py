from __future__ import division

import sympy as sp
import loopy as lp
import numpy as np




class Kernel:
    """Basic kernel interface."""

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def get_expression(self, dist_vec):
        """Return a :mod:`sympy` expression for the kernel.

        :arg dist_vec: target - source

        (Assumes translation invariance of the kernel.)
        """
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
        r = sp.sqrt((dist_vec.T*dist_vec)[0,0])

        if self.dimensions == 2:
            return sp.log(r) / (2*sp.pi)
        elif self.dimensions == 3:
            return 1/r
        else:
            raise RuntimeError("unsupported dimensionality")




class HelmholtzKernel(Kernel):
    is_complex = True

    def get_expression(self, dist_vec):
        assert self.dimensions == len(dist_vec)

        r = sp.sqrt((dist_vec.T*dist_vec)[0,0])

        i = sp.sqrt(-1)
        k = sp.Symbol("k")

        if self.dimensions == 2:
            return i/4 * sp.Function("H1_0")(k*r)
        elif self.dimensions == 3:
            return sp.exp(i*k*r)/r
        else:
            raise RuntimeError("unsupported dimensionality")

    def get_args(self):
        return [
                lp.ScalarArg("k", np.complex128),
                ]

    def get_preambles(self):
        from sumpy.codegen import HANKEL_PREAMBLE
        return [HANKEL_PREAMBLE]

# }}}

# {{{ derivatives

class TargetDerivative(Kernel):
    def __init__(self, axis, kernel):
        Kernel.__init__(self, kernel.dimensions)
        self.kernel = kernel
        self.axis = axis

    @property
    def is_complex(self):
        return self.kernel.is_complex

    def get_expression(self, dist_vec):
        return self.kernel.get_expression(dist_vec).diff(dist_vec[self.axis])

    def get_args(self):
        return self.kernel.get_args()

    def get_preambles(self):
        return self.kernel.get_preambles()




class SourceDerivative(Kernel):
    def __init__(self, kernel, dir_vec_name="deriv_dir", dir_vec_dtype=np.float64):
        Kernel.__init__(self, kernel.dimensions)

        self.kernel = kernel
        self.dir_vec_name = dir_vec_name
        self.dir_vec_dtype = dir_vec_dtype

    @property
    def is_complex(self):
        return self.kernel.is_complex

    def get_expression(self, dist_vec):
        dimensions = len(dist_vec)

        knl = self.kernel.get_expression(dist_vec)
        from sumpy.symbolic import make_sym_vector
        dir_vec = make_sym_vector(self.deriv_dir, dimensions)

        # dist_vec = tgt-src -> minus sign
        return sum(-dir_vec[axis]*knl.diff(axis) for axis in range(dimensions))

    def get_args(self):
        return self.kernel.get_args() + [
            lp.ArrayArg(self.dir_vec_name, self.dir_vec_dtype,
                shape=("nsrc", self.dimensions), order="C")]

    def get_preambles(self):
        return self.kernel.get_preambles()

# }}}

# vim: fdm=marker
