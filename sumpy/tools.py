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

from pytools import memoize_method
import numpy as np

import logging
logger = logging.getLogger(__name__)


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


def build_matrix(op, dtype=None, shape=None):
    dtype = dtype or op.dtype
    from pytools import ProgressBar
    shape = shape or op.shape
    rows, cols = shape
    pb = ProgressBar("matrix", cols)
    mat = np.zeros(shape, dtype)

    try:
        matvec_method = op.matvec
    except AttributeError:
        matvec_method = op.__call__

    for i in range(cols):
        unit_vec = np.zeros(cols, dtype=dtype)
        unit_vec[i] = 1
        mat[:, i] = matvec_method(unit_vec)
        pb.progress()

    pb.finished()

    return mat


def vector_to_device(queue, vec):
    from pytools.obj_array import with_object_array_or_scalar

    from pyopencl.array import to_device

    def to_dev(ary):
        return to_device(queue, ary)

    return with_object_array_or_scalar(to_dev, vec)


def gather_arguments(kernel_likes):
    result = {}
    for knl in kernel_likes:
        for arg in knl.get_args():
            result[arg.name] = arg
            # FIXME: possibly check that arguments match before overwriting

    return sorted(result.itervalues(), key=lambda arg: arg.name)


def gather_source_arguments(kernel_likes):
    result = {}
    for knl in kernel_likes:
        for arg in knl.get_args() + knl.get_source_args():
            result[arg.name] = arg
            # FIXME: possibly check that arguments match before overwriting

    return sorted(result.itervalues(), key=lambda arg: arg.name)


# {{{  KernelComputation

class KernelComputation(object):
    """Common input processing for kernel computations."""

    def __init__(self, ctx, kernels, strength_usage,
            value_dtypes, name, options=[], device=None):
        """
        :arg kernels: list of :class:`sumpy.kernel.Kernel` instances
            :class:`sumpy.kernel.TargetDerivative` wrappers should be
            the outermost kernel wrappers, if present.
        :arg strength_usage: A list of integers indicating which expression
            uses which density. This implicitly specifies the
            number of density arrays that need to be passed.
            Default: all kernels use the same density.
        """

        # {{{ process value_dtypes

        if value_dtypes is None:
            value_dtypes = []
            for knl in kernels:
                if knl.is_complex_valued:
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

        if device is None:
            device = ctx.devices[0]

        self.context = ctx
        self.device = device

        self.kernels = kernels
        self.value_dtypes = value_dtypes
        self.strength_usage = strength_usage
        self.strength_count = strength_count

        self.name = name or self.default_name

    def get_kernel_scaling_assignments(self):
        from pymbolic.sympy_interface import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()

        import loopy as lp
        return [
                lp.ExpressionInstruction(id=None,
                    assignee="knl_%d_scaling" % i,
                    expression=sympy_conv(kernel.get_scaling()),
                    temp_var_type=dtype)
                for i, (kernel, dtype) in enumerate(
                    zip(self.kernels, self.value_dtypes))]

# }}}


class KernelCacheWrapper(object):
    @memoize_method
    def get_cached_optimized_kernel(self):
        from sumpy import code_cache, CACHING_ENABLED

        if CACHING_ENABLED:
            cache_key = self.get_cache_key()

            try:
                result = code_cache[cache_key]
                logger.info("%s: kernel cache hit" % self.name)
                return result
            except KeyError:
                pass

        knl = self.get_optimized_kernel()

        if CACHING_ENABLED:
            code_cache[cache_key] = knl

        return knl

# vim: fdm=marker
