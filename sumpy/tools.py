from __future__ import annotations


__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
Copyright (C) 2020 Isuru Fernando
"""

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

import enum
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import loopy as lp
import pyopencl as cl
from pymbolic.mapper import WalkMapper
from pytools import memoize_method
from pytools.tag import Tag, tag_dataclass

import sumpy.symbolic as sym
from sumpy.array_context import PyOpenCLArrayContext, make_loopy_program


if TYPE_CHECKING:
    import numpy

    import pyopencl

    from sumpy.kernel import Kernel

logger = logging.getLogger(__name__)


__doc__ = """
Tools
=====

.. autofunction:: to_complex_dtype
.. autofunction:: is_obj_array_like
.. autofunction:: vector_to_device
.. autofunction:: vector_from_device
.. autoclass:: OrderedSet

Multi-index Helpers
-------------------

.. autofunction:: add_mi
.. autofunction:: mi_factorial
.. autofunction:: mi_increment_axis
.. autofunction:: mi_set_axis
.. autofunction:: mi_power

Symbolic Helpers
----------------

.. autofunction:: add_to_sac
.. autofunction:: gather_arguments
.. autofunction:: gather_source_arguments
.. autofunction:: gather_loopy_arguments
.. autofunction:: gather_loopy_source_arguments

.. autoclass:: ScalingAssignmentTag
.. autoclass:: KernelComputation
.. autoclass:: KernelCacheMixin

.. autofunction:: reduced_row_echelon_form
.. autofunction:: nullspace

FFT
---

.. autofunction:: fft
.. autofunction:: fft_toeplitz_upper_triangular
.. autofunction:: matvec_toeplitz_upper_triangular

.. autoclass:: FFTBackend
    :members:
.. autofunction:: loopy_fft
.. autofunction:: get_opencl_fft_app
.. autofunction:: run_opencl_fft

Profiling
---------

.. autofunction:: get_native_event
.. autoclass:: ProfileGetter
.. autoclass:: AggregateProfilingEvent
.. autoclass:: MarkerBasedProfilingEvent
"""


# {{{ multi_index helpers

def add_mi(mi1: Sequence[int], mi2: Sequence[int]) -> tuple[int, ...]:
    # NOTE: these are used a lot and `tuple([])` is faster
    return tuple([mi1i + mi2i for mi1i, mi2i in zip(mi1, mi2, strict=True)])  # noqa: C409


def mi_factorial(mi: Sequence[int]) -> int:
    import math
    result = 1
    for mi_i in mi:
        result *= math.factorial(mi_i)
    return result


def mi_increment_axis(
        mi: Sequence[int], axis: int, increment: int
        ) -> tuple[int, ...]:
    new_mi = list(mi)
    new_mi[axis] += increment
    return tuple(new_mi)


def mi_set_axis(mi: Sequence[int], axis: int, value: int) -> tuple[int, ...]:
    new_mi = list(mi)
    new_mi[axis] = value
    return tuple(new_mi)


def mi_power(
        vector: Sequence[Any], mi: Sequence[int],
        evaluate: bool = True) -> Any:
    result = 1
    for mi_i, vec_i in zip(mi, vector, strict=True):
        if mi_i == 1:
            result *= vec_i
        elif evaluate:
            result *= vec_i**mi_i
        else:
            result *= sym.unevaluated_pow(vec_i, mi_i)
    return result


def add_to_sac(sac, expr):
    if sac is None:
        return expr

    from numbers import Number
    if isinstance(expr, Number | sym.Number | sym.Symbol):
        return expr

    name = sac.assign_temp("temp", expr)
    return sym.Symbol(name)


# }}}


# {{{ get variables

class GatherAllVariables(WalkMapper):
    def __init__(self):
        self.vars = set()

    def map_variable(self, expr):
        self.vars.add(expr)


def get_all_variables(expr):
    mapper = GatherAllVariables()
    mapper(expr)
    return mapper.vars

# }}}


def build_matrix(op, dtype=None, shape=None):
    dtype = dtype or op.dtype
    from pytools import ProgressBar
    shape = shape or op.shape
    _rows, cols = shape
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


def _merge_kernel_arguments(dictionary, arg):
    # Check for strict equality until there's a usecase
    if dictionary.setdefault(arg.name, arg) != arg:
        msg = "Merging two different kernel arguments {} and {} with the same name"
        raise ValueError(msg.format(arg.loopy_arg, dictionary[arg].loopy_arg))


def gather_arguments(kernel_likes):
    result = {}
    for knl in kernel_likes:
        for arg in knl.get_args():
            _merge_kernel_arguments(result, arg)

    return sorted(result.values(), key=lambda arg: arg.name)


def gather_source_arguments(kernel_likes):
    result = {}
    for knl in kernel_likes:
        for arg in knl.get_args() + knl.get_source_args():
            _merge_kernel_arguments(result, arg)

    return sorted(result.values(), key=lambda arg: arg.name)


def gather_loopy_arguments(kernel_likes):
    return [arg.loopy_arg for arg in gather_arguments(kernel_likes)]


def gather_loopy_source_arguments(kernel_likes):
    return [arg.loopy_arg for arg in gather_source_arguments(kernel_likes)]


# {{{  KernelComputation

@tag_dataclass
class ScalingAssignmentTag(Tag):
    pass


class KernelComputation(ABC):
    """Common input processing for kernel computations.

    .. attribute:: name
    .. attribute:: target_kernels
    .. attribute:: source_kernels
    .. attribute:: strength_usage

    .. automethod:: get_kernel
    """

    def __init__(self, ctx: Any,
            target_kernels: list[Kernel],
            source_kernels: list[Kernel],
            strength_usage: list[int] | None = None,
            value_dtypes: list[numpy.dtype[Any]] | None = None,
            name: str | None = None) -> None:
        """
        :arg target_kernels: list of :class:`~sumpy.kernel.Kernel` instances,
            with :class:`sumpy.kernel.DirectionalTargetDerivative` as
            the outermost kernel wrappers, if present.
        :arg source_kernels: list of :class:`~sumpy.kernel.Kernel` instances
            with :class:`~sumpy.kernel.DirectionalSourceDerivative` as the
            outermost kernel wrappers, if present.
        :arg strength_usage: list of integers indicating which expression
            uses which density. This implicitly specifies the
            number of density arrays that need to be passed.
            Default: all kernels use the same density.
        """

        # {{{ process value_dtypes

        if value_dtypes is None:
            value_dtypes = []
            for knl in target_kernels:
                if knl.is_complex_valued:
                    value_dtypes.append(np.dtype(np.complex128))
                else:
                    value_dtypes.append(np.dtype(np.float64))

        if not isinstance(value_dtypes, list | tuple):
            value_dtypes = [np.dtype(value_dtypes)] * len(target_kernels)
        value_dtypes = [np.dtype(vd) for vd in value_dtypes]

        # }}}

        # {{{ process strength_usage

        if strength_usage is None:
            strength_usage = list(range(len(source_kernels)))

        if len(source_kernels) != len(strength_usage):
            raise ValueError("exprs and strength_usage must have the same length")
        strength_count = max(strength_usage)+1

        # }}}

        self.source_kernels = tuple(source_kernels)
        self.target_kernels = tuple(target_kernels)
        self.value_dtypes = value_dtypes
        self.strength_usage = strength_usage
        self.strength_count = strength_count

        self.name = name or self.default_name

    @property
    def nresults(self):
        return len(self.target_kernels)

    @property
    @abstractmethod
    def default_name(self):
        pass

    def get_kernel_scaling_assignments(self):
        from sumpy.symbolic import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()

        import loopy as lp
        return [
                lp.Assignment(id=f"knl_{i}_scaling",
                    assignee=f"knl_{i}_scaling",
                    expression=sympy_conv(kernel.get_global_scaling_const()),
                    temp_var_type=lp.Optional(dtype),
                    tags=frozenset([ScalingAssignmentTag()]))
                for i, (kernel, dtype) in enumerate(
                    zip(self.target_kernels, self.value_dtypes, strict=True))]

    @abstractmethod
    def get_kernel(self):
        pass

# }}}


# {{{ OrderedSet

# Source: https://code.activestate.com/recipes/576694-orderedset/
# Author: Raymond Hettinger
# License: MIT

from collections.abc import MutableSet


class OrderedSet(MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError("set is empty")
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({list(self)!r})"

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

# }}}


class KernelCacheMixin(ABC):
    context: cl.Context
    name: str

    @abstractmethod
    def get_cache_key(self) -> tuple[Hashable, ...]:
        ...

    @abstractmethod
    def get_kernel(self, **kwargs) -> lp.TranslationUnit:
        ...

    @abstractmethod
    def get_optimized_kernel(self, **kwargs) -> lp.TranslationUnit:
        ...

    @memoize_method
    def get_cached_kernel_executor(self, **kwargs) -> lp.ExecutorBase:
        from sumpy import CACHING_ENABLED, NO_CACHE_KERNELS, OPT_ENABLED, code_cache

        if CACHING_ENABLED and not (
                NO_CACHE_KERNELS and self.name in NO_CACHE_KERNELS):
            import loopy.version

            from sumpy.version import KERNEL_VERSION
            cache_key = (
                    self.get_cache_key()
                    + tuple(sorted(kwargs.items()))
                    + (loopy.version.DATA_MODEL_VERSION,)
                    + (KERNEL_VERSION,)
                    + (OPT_ENABLED,))

            try:
                result = code_cache[cache_key]
                logger.debug("%s: kernel cache hit [key=%s]", self.name, cache_key)
                return result.executor(self.context)
            except KeyError:
                pass

        logger.info("%s: kernel cache miss", self.name)
        if CACHING_ENABLED and not (
                NO_CACHE_KERNELS and self.name in NO_CACHE_KERNELS):
            logger.info("%s: kernel cache miss [key=%s]",
                self.name, cache_key)

        from pytools import MinRecursionLimit
        with MinRecursionLimit(3000):
            if OPT_ENABLED:
                knl = self.get_optimized_kernel(**kwargs)
            else:
                knl = self.get_kernel()

        if CACHING_ENABLED and not (
                NO_CACHE_KERNELS and self.name in NO_CACHE_KERNELS):
            code_cache.store_if_not_present(cache_key, knl)

        return knl.executor(self.context)

    @staticmethod
    def _allow_redundant_execution_of_knl_scaling(knl):
        from loopy.match import ObjTagged
        return lp.add_inames_for_unused_hw_axes(
                knl, within=ObjTagged(ScalingAssignmentTag()))


KernelCacheWrapper = KernelCacheMixin


def is_obj_array_like(ary):
    return (
            isinstance(ary, tuple | list)
            or (isinstance(ary, np.ndarray) and ary.dtype.char == "O"))


# {{{ matrices

def reduced_row_echelon_form(m, atol=0):
    """Calculates a reduced row echelon form of a
    matrix `m`.

    :arg m: a 2D :class:`numpy.ndarray` or a list of lists or a sympy Matrix
    :arg atol: absolute tolerance for values to be considered zero
    :return: reduced row echelon form as a 2D :class:`numpy.ndarray`
             and a list of pivots
    """

    mat = np.array(m, dtype=object)
    index = 0
    nrows = mat.shape[0]
    ncols = mat.shape[1]
    pivot_cols = []
    for i in range(ncols):
        if index == nrows:
            break
        pivot = nrows
        for k in range(index, nrows):
            symbolic = isinstance(mat[k, i], sym.Basic) and not mat[k, i].is_number
            if (symbolic or abs(mat[k, i]) > atol) and pivot == nrows:
                pivot = k
            # If there's a pivot that's close to 1 use that as it avoids
            # having to divide.
            # When checking for a number close to 1, we shouldn't consider
            # symbolic values
            if not symbolic and abs(mat[k, i] - 1) <= atol:
                pivot = k
                break
        if pivot == nrows:
            # no nonzero pivot found, next column
            continue
        if pivot != index:
            mat[[pivot, index], :] = mat[[index, pivot], :]

        pivot_cols.append(i)
        scale = mat[index, i]
        if isinstance(scale, int | sym.Integer):
            scale = int(scale)

        for j in range(mat.shape[1]):
            elem = mat[index, j]
            if isinstance(scale, int) and isinstance(elem, int | sym.Integer):
                quo = int(elem) // scale
                if quo * scale == elem:
                    mat[index, j] = quo
                    continue
            mat[index, j] = sym.sympify(elem)/scale

        for j in range(nrows):
            if (j == index):
                continue

            scale = mat[j, i]
            if scale != 0:
                mat[j, :] = mat[j, :] - mat[index, :]*scale

        index = index + 1

    return mat, pivot_cols


def nullspace(m, atol=0):
    """Calculates the nullspace of a matrix `m`.

    :arg m: a 2D :class:`numpy.ndarray` or a list of lists or a sympy Matrix
    :arg atol: absolute tolerance for values to be considered zero
    :return: nullspace of `m` as a 2D :class:`numpy.ndarray`
    """
    mat, pivot_cols = reduced_row_echelon_form(m, atol=atol)
    pivot_cols = list(pivot_cols)
    cols = mat.shape[1]

    free_vars = [i for i in range(cols) if i not in pivot_cols]

    n = []
    for free_var in free_vars:
        vec = [0]*cols
        vec[free_var] = 1
        for piv_row, piv_col in enumerate(pivot_cols):
            for pos in pivot_cols[piv_row+1:] + [free_var]:
                if isinstance(mat[piv_row, pos], sym.Integer):
                    vec[piv_col] -= int(mat[piv_row, pos])
                else:
                    vec[piv_col] -= mat[piv_row, pos]
        n.append(vec)
    return np.array(n, dtype=object).T

# }}}


# {{{ FFT

def fft(seq, inverse=False, sac=None):
    """
    Return the discrete fourier transform of the sequence seq.
    seq should be a python iterable with tuples of length 2
    corresponding to the real part and imaginary part.
    """

    from pymbolic.algorithm import fft as _fft, ifft as _ifft

    def wrap(level, expr):
        if isinstance(expr, np.ndarray):
            res = [wrap(level, a) for a in expr]
            return np.array(res, dtype=object).reshape(expr.shape)
        return add_to_sac(sac, expr)

    if inverse:
        return _ifft(np.array(seq), wrap_intermediate_with_level=wrap,
                complex_dtype=np.complex128).tolist()
    else:
        return _fft(np.array(seq), wrap_intermediate_with_level=wrap,
                complex_dtype=np.complex128).tolist()


def fft_toeplitz_upper_triangular(first_row, x, sac=None):
    """
    Returns the matvec of the Toeplitz matrix given by
    the first row and the vector x using a Fourier transform
    """
    assert len(first_row) == len(x)
    n = len(first_row)
    v = list(first_row)
    v += [0]*(n-1)

    x = list(reversed(x))
    x += [0]*(n-1)

    v_fft = fft(v, sac)
    x_fft = fft(x, sac)
    res_fft = [add_to_sac(sac, a * b) for a, b in zip(v_fft, x_fft, strict=True)]
    res = fft(res_fft, inverse=True, sac=sac)
    return list(reversed(res[:n]))


def matvec_toeplitz_upper_triangular(first_row, vector):
    n = len(first_row)
    assert len(vector) == n
    output = [0]*n
    for row in range(n):
        terms = tuple(first_row[col-row]*vector[col] for col in range(row, n))
        output[row] = sym.Add(*terms)
    return output


to_complex_type_dict = {
    np.complex64: np.complex64,
    np.complex128: np.complex128,
    np.float32: np.complex64,
    np.float64: np.complex128,
}


def to_complex_dtype(dtype):
    np_type = np.dtype(dtype).type
    try:
        return to_complex_type_dict[np_type]
    except KeyError as err:
        raise RuntimeError(f"Unknown dtype: {dtype}") from err


@dataclass(frozen=True)
class ProfileGetter:
    start: int
    end: int


def get_native_event(evt):
    from pyopencl import Event
    return evt if isinstance(evt, Event) else evt.native_event


class AggregateProfilingEvent:
    """An object to hold a list of events and provides compatibility
    with some of the functionality of :class:`pyopencl.Event`.
    Assumes that the last event waits on all of the previous events.
    """
    def __init__(self, events):
        self.events = events[:]
        self.native_event = get_native_event(events[-1])

    @property
    def profile(self):
        total = sum(evt.profile.end - evt.profile.start for evt in self.events)
        end = self.native_event.profile.end
        return ProfileGetter(start=end - total, end=end)

    def wait(self):
        return self.native_event.wait()


class MarkerBasedProfilingEvent:
    """An object to hold two marker events and provides compatibility
    with some of the functionality of :class:`pyopencl.Event`.
    """
    def __init__(self, *, end_event, start_event):
        self.native_event = end_event
        self.start_event = start_event

    @property
    def profile(self):
        return ProfileGetter(start=self.start_event.profile.start,
                             end=self.native_event.profile.end)

    def wait(self):
        return self.native_event.wait()


def loopy_fft(shape, inverse, complex_dtype, index_dtype=None,
        name=None):
    from math import pi

    from pymbolic import var
    from pymbolic.algorithm import find_factors

    sign = 1 if not inverse else -1
    n = shape[-1]

    m = n
    factors = []
    while m != 1:
        N1, m = find_factors(m)  # noqa: N806
        factors.append(N1)

    nfft = n
    broadcast_dims = tuple(var(f"j{d}") for d in range(len(shape) - 1))

    domains = [
        "{[i]: 0<=i<n}",
        "{[i2]: 0<=i2<n}",
    ]
    domains += [f"{{[j{d}]: 0<=j{d}<{shape[d]} }}" for d in range(len(shape) - 1)]

    x = var("x")
    y = var("y")
    i = var("i")
    i2 = var("i2")
    i3 = var("i3")

    fixed_parameters = {"const": complex_dtype(sign*(-2j)*pi/n), "n": n}

    index = (*broadcast_dims, i2)
    insns = [
        "exp_table[i] = exp(const * i) {id=exp_table}",
        lp.Assignment(
            assignee=x[index],
            expression=y[index],
            id="copy",
            happens_after=frozenset(["exp_table"]),
        ),
    ]

    for ilev, N1 in enumerate(list(reversed(factors))):  # noqa: N806
        nfft //= N1
        N2 = n // (nfft * N1)  # noqa: N806
        init_happens_after = "copy" if ilev == 0 else f"update_{ilev-1}"

        temp = var("temp")
        exp_table = var("exp_table")
        i = var(f"i_{ilev}")
        i2 = var(f"i2_{ilev}")
        ifft = var(f"ifft_{ilev}")
        iN1 = var(f"iN1_{ilev}")           # noqa: N806
        iN1_sum = var(f"iN1_sum_{ilev}")   # noqa: N806
        iN2 = var(f"iN2_{ilev}")           # noqa: N806
        table_idx = var(f"table_idx_{ilev}")
        exp = var(f"exp_{ilev}")

        i_bcast = (*broadcast_dims, i)
        i2_bcast = (*broadcast_dims, i2)
        iN_bcast = (*broadcast_dims, ifft + nfft * (iN1 * N2 + iN2))  # noqa: N806

        insns += [
            lp.Assignment(
                assignee=temp[i],
                expression=x[i_bcast],
                id=f"copy_{ilev}",
                happens_after=frozenset([init_happens_after]),
            ),
            lp.Assignment(
                assignee=x[i2_bcast],
                expression=0,
                id=f"reset_{ilev}",
                happens_after=frozenset([f"copy_{ilev}"])),
            lp.Assignment(
                assignee=table_idx,
                expression=nfft*iN1_sum*(iN2 + N2*iN1),
                id=f"idx_{ilev}",
                happens_after=frozenset([f"reset_{ilev}"]),
                temp_var_type=lp.Optional(np.uint32)),
            lp.Assignment(
                assignee=exp,
                expression=exp_table[table_idx % n],
                id=f"exp_{ilev}",
                happens_after=frozenset([f"idx_{ilev}"]),
                within_inames=frozenset({x.name for x in
                    [*broadcast_dims, iN1_sum, iN1, iN2]}),
                temp_var_type=lp.Optional(complex_dtype)),
            lp.Assignment(
                assignee=x[iN_bcast],
                expression=(x[iN_bcast]
                    + exp * temp[ifft + nfft * (iN2*N1 + iN1_sum)]),
                id=f"update_{ilev}",
                happens_after=frozenset([f"exp_{ilev}"])),
        ]

        domains += [
            f"[ifft_{ilev}]: 0<=ifft_{ilev}<{nfft}",
            f"[iN1_{ilev}]: 0<=iN1_{ilev}<{N1}",
            f"[iN1_sum_{ilev}]: 0<=iN1_sum_{ilev}<{N1}",
            f"[iN2_{ilev}]: 0<=iN2_{ilev}<{N2}",
            f"[i_{ilev}]: 0<=i_{ilev}<{n}",
            f"[i2_{ilev}]: 0<=i2_{ilev}<{n}",
        ]

    for idom, dom in enumerate(domains):
        if not dom.startswith("{"):
            domains[idom] = "{" + dom + "}"

    kernel_data = [
        lp.GlobalArg("x", shape=shape, is_input=False, is_output=True,
            dtype=complex_dtype),
        lp.GlobalArg("y", shape=shape, is_input=True, is_output=False,
            dtype=complex_dtype),
        lp.TemporaryVariable("exp_table", shape=(n,),
            dtype=complex_dtype),
        lp.TemporaryVariable("temp", shape=(n,),
            dtype=complex_dtype),
        ...
    ]

    if n == 1:
        domains = domains[2:]
        index = (*broadcast_dims, 0)
        insns = [
            lp.Assignment(
                assignee=x[index],
                expression=y[index],
            ),
        ]
        kernel_data = kernel_data[:2]
    elif inverse:
        domains += ["{[i3]: 0<=i3<n}"]
        index = (*broadcast_dims, i3)
        insns += [
            lp.Assignment(
                assignee=x[index],
                expression=x[index] / n,
                happens_after=frozenset([f"update_{len(factors) - 1}"]),
            ),
        ]

    if name is None:
        name = f"ifft_{n}" if inverse else f"fft_{n}"

    knl = make_loopy_program(
        domains, insns,
        kernel_data=kernel_data,
        name=name,
        fixed_parameters=fixed_parameters,
        index_dtype=index_dtype,
    )

    if broadcast_dims:
        knl = lp.split_iname(knl, "j0", 32, inner_tag="l.0", outer_tag="g.0")
        knl = lp.add_inames_for_unused_hw_axes(knl)

    return knl


class FFTBackend(enum.Enum):
    #: FFT backend based on the vkFFT library.
    pyvkfft = 1
    #: FFT backend based on :mod:`loopy` used as a fallback.
    loopy = 2


def _get_fft_backend(queue: pyopencl.CommandQueue) -> FFTBackend:
    import os

    env_val = os.environ.get("SUMPY_FFT_BACKEND")
    if env_val:
        if env_val not in ["loopy", "pyvkfft"]:
            raise ValueError("Expected 'loopy' or 'pyvkfft' for SUMPY_FFT_BACKEND. "
                   f"Found {env_val}.")
        return FFTBackend[env_val]

    try:
        import pyvkfft.opencl  # noqa: F401
    except ImportError:
        warnings.warn("VkFFT not found. FFT runs will be slower.", stacklevel=3)
        return FFTBackend.loopy

    from pyopencl import command_queue_properties

    if queue.properties & command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE:
        warnings.warn(
            "VkFFT does not support out of order queues yet. "
            "Falling back to slower implementation.", stacklevel=3)
        return FFTBackend.loopy

    import platform
    import sys

    if (sys.platform == "darwin"
            and platform.machine() == "x86_64"
            and queue.context.devices[0].platform.name
            == "Portable Computing Language"):
        warnings.warn(
            "PoCL miscompiles some VkFFT kernels. "
            "See https://github.com/inducer/sumpy/issues/129. "
            "Falling back to slower implementation.", stacklevel=3)
        return FFTBackend.loopy

    return FFTBackend.pyvkfft


def get_opencl_fft_app(
        actx: PyOpenCLArrayContext,
        shape: tuple[int, ...],
        dtype: numpy.dtype[Any],
        inverse: bool) -> Any:
    """Setup an object for out-of-place FFT on with given shape and dtype
    on given queue.
    """
    assert dtype.type in (np.float32, np.float64, np.complex64,
                           np.complex128)

    backend = _get_fft_backend(actx.queue)

    if backend == FFTBackend.loopy:
        return loopy_fft(shape, inverse=inverse, complex_dtype=dtype.type), backend
    elif backend == FFTBackend.pyvkfft:
        from pyvkfft.opencl import VkFFTApp
        app = VkFFTApp(
            shape=shape, dtype=dtype,
            queue=actx.queue, ndim=1, inplace=False)
        return app, backend
    else:
        raise RuntimeError(f"Unsupported FFT backend {backend}")


def run_opencl_fft(
        actx: PyOpenCLArrayContext,
        fft_app: tuple[Any, FFTBackend],
        input_vec: Any,
        inverse: bool = False,
        wait_for: list[pyopencl.Event] | None = None
    ) -> tuple[pyopencl.Event, Any]:
    """Runs an FFT on input_vec and returns a :class:`MarkerBasedProfilingEvent`
    that indicate the end and start of the operations carried out and the output
    vector.
    Only supports in-order queues.
    """
    app, backend = fft_app

    if backend == FFTBackend.loopy:
        evt, output_vec = app(actx.queue, y=input_vec, wait_for=wait_for)
        return (evt, output_vec["x"])
    elif backend == FFTBackend.pyvkfft:
        if wait_for is None:
            wait_for = []

        import pyopencl as cl

        queue = actx.queue
        if queue.device.platform.name == "NVIDIA CUDA":
            # NVIDIA OpenCL gives wrong event profile values with wait_for
            # Not passing wait_for will wait for all events queued before
            # and therefore correctness is preserved if it's the same queue
            for evt in wait_for:
                if not evt.command_queue != queue:
                    raise RuntimeError(
                        "Different queues not supported with NVIDIA CUDA")
            start_evt = cl.enqueue_marker(queue)
        else:
            start_evt = cl.enqueue_marker(queue, wait_for=wait_for[:])

        if app.inplace:
            raise RuntimeError("inplace fft is not supported")
        else:
            output_vec = actx.np.empty_like(input_vec)

        # FIXME: use the public API once
        # https://github.com/vincefn/pyvkfft/pull/17 is in
        from pyvkfft.opencl import _vkfft_opencl
        if inverse:  # noqa: SIM108
            meth = _vkfft_opencl.ifft
        else:
            meth = _vkfft_opencl.fft

        meth(app.app, int(input_vec.data.int_ptr),
            int(output_vec.data.int_ptr), int(actx.queue.int_ptr))

        if queue.device.platform.name == "NVIDIA CUDA":
            end_evt = cl.enqueue_marker(queue)
        else:
            end_evt = cl.enqueue_marker(queue, wait_for=[start_evt])

        output_vec.add_event(end_evt)

        return (MarkerBasedProfilingEvent(end_event=end_evt, start_event=start_evt),
            output_vec)
    else:
        raise RuntimeError(f"Unsupported FFT backend {backend}")

# }}}


# {{{ deprecations

_depr_name_to_replacement_and_obj = {
    "KernelCacheWrapper": ("KernelCacheMixin", 2023),
    }


def __getattr__(name):
    replacement_and_obj = _depr_name_to_replacement_and_obj.get(name)
    if replacement_and_obj is not None:
        replacement, obj, year = replacement_and_obj
        from warnings import warn
        warn(f"'sumpy.tools.{name}' is deprecated. "
                f"Use '{replacement}' instead. "
                f"'sumpy.tools.{name}' will continue to work until {year}.",
                DeprecationWarning, stacklevel=2)
        return obj
    else:
        raise AttributeError(name)

# }}}

# vim: fdm=marker
