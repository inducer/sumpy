from __future__ import division

import numpy as np
import loopy as lp




# {{{ multi_index helpers

def add_mi(mi1, mi2):
    return tuple(mi1i+mi2i for mi1i, mi2i in zip(mi1, mi2))

def mi_factorial(mi):
    from pytools import factorial
    result = 1
    for mi_i in mi:
        result *= factorial(mi_i)
    return result

def mi_power(vector, mi):
    result = 1
    for mi_i, vec_i in zip(mi, vector):
        result *= vec_i**mi_i
    return result

def mi_derivative(expr, vector, mi):
    for mi_i, vec_i in zip(mi, vector):
        expr = expr.diff(vec_i, mi_i)
    return expr

# }}}



def build_matrix(op, dtype=None):
    dtype = dtype or op.dtype
    from pytools import ProgressBar
    rows, cols = op.shape
    pb = ProgressBar("matrix", cols)
    mat = np.zeros(op.shape, dtype)

    for i in range(cols):
        unit_vec = np.zeros(rows, dtype=dtype)
        unit_vec[i] = 1
        mat[:,i] = op.matvec(unit_vec)
        pb.progress()

    pb.finished()

    return mat




def vector_to_device(queue, vec):
    from pytools.obj_array import with_object_array_or_scalar

    from pyopencl.array import to_device
    def to_dev(ary):
        return to_device(queue, ary)

    return with_object_array_or_scalar(to_dev, vec)





class KernelComputation:
    """Common input processing for kernel computations."""

    def __init__(self, ctx, kernels, strength_usage,
            value_dtypes, strength_dtypes,
            geometry_dtype,
            name="kernel", options=[], device=None):
        """
        :arg kernels: list of :class:`sumpy.kernel.Kernel` instances
            :class:`sumpy.kernel.TargetDerivative` wrappers should be
            the outermost kernel wrappers, if present.
        :arg strength_usage: A list of integers indicating which expression
          uses which density. This implicitly specifies the
          number of density arrays that need to be passed.
          Default: all kernels use the same density.
        """

        if geometry_dtype is None:
            geometry_dtype = np.float64

        geometry_dtype = np.dtype(geometry_dtype)

        # {{{ process value_dtypes

        if value_dtypes is None:
            value_dtypes = []
            for knl in kernels:
                if knl.is_complex:
                    value_dtypes.append(np.complex128)
                else:
                    value_dtypes.append(np.float64)

        if not isinstance(value_dtypes, (list, tuple)):
            value_dtypes = [np.dtype(value_dtypes)] * len(kernels)
        value_dtypes = [np.dtype(vd) for vd in value_dtypes]

        # }}}

        # {{{ process strength_usage

        if strength_usage is None:
            strength_usage = [0] * len(kernels)

        if len(kernels) != len(strength_usage):
            raise ValueError("exprs and strength_usage must have the same length")
        strength_count = max(strength_usage)+1

        # }}}

        # {{{ process strength_dtypes

        if strength_dtypes is None:
            strength_dtypes = value_dtypes[0]

        if not isinstance(strength_dtypes, (list, tuple)):
            strength_dtypes = [np.dtype(strength_dtypes)] * strength_count

        if len(strength_dtypes) != strength_count:
            raise ValueError("exprs and strength_usage must have the same length")

        strength_dtypes = [np.dtype(dtype) for dtype in strength_dtypes]

        # }}}

        if device is None:
            device = ctx.devices[0]

        self.context = ctx
        self.device = device

        self.kernels = kernels
        self.value_dtypes = value_dtypes
        self.strength_usage = strength_usage
        self.strength_dtypes = strength_dtypes
        self.geometry_dtype = geometry_dtype

        self.name = name

    def gather_kernel_arguments(self):
        result = {}
        for knl in self.kernels:
            for arg in knl.get_args():
                result[arg.name] = arg
                # FIXME: possibly check that arguments match before overwriting

        return sorted(result.itervalues(), key=lambda arg: arg.name)

    def gather_kernel_preambles(self):
        result = []

        for knl in self.kernels:
            result.extend(knl.get_preambles())

        return result

    def get_kernel_scaling_assignments(self):
        from sumpy.codegen import to_loopy_insns
        return to_loopy_insns(
                ("knl_%d_scaling" % i, kernel.get_scaling())
                for i, kernel in enumerate(self.kernels))
