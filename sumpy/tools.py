__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
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

__doc__ = """

 Misc tools
 ==========

 .. autoclass:: BlockIndexRanges
 .. autoclass:: MatrixBlockIndexRanges
 .. autoclass:: ExprDerivativeTaker
 .. autoclass:: LaplaceDerivativeTaker
 .. autoclass:: RadialDerivativeTaker
 .. autoclass:: HelmholtzDerivativeTaker
 .. autoclass:: DifferentiatedExprDerivativeTaker
"""

from pytools import memoize_method, memoize_in
from pytools.tag import Tag, tag_dataclass
from pymbolic.mapper import WalkMapper

import numpy as np
import sumpy.symbolic as sym

import pyopencl as cl
import pyopencl.array  # noqa

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION

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


def mi_increment_axis(mi, axis, increment):
    new_mi = list(mi)
    new_mi[axis] += increment
    return tuple(new_mi)


def mi_set_axis(mi, axis, value):
    new_mi = list(mi)
    new_mi[axis] = value
    return tuple(new_mi)


def mi_power(vector, mi, evaluate=True):
    result = 1
    for mi_i, vec_i in zip(mi, vector):
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

    if expr.is_Number or expr.is_Symbol:
        return expr

    name = sac.assign_temp("temp", expr)
    return sym.Symbol(name)


class ExprDerivativeTaker(object):
    """Facilitates the efficient computation of (potentially) high-order
    derivatives of a given :mod:`sympy` expression *expr* while attempting
    to maximize the number of common subexpressions generated.

    This class defines the interface and realizes a baseline implementation.
    More specialized implementations may offer better efficiency for special
    cases.

    .. automethod:: diff
    """

    def __init__(self, expr, var_list, rscale=1, sac=None):
        r"""
        A class to take scaled derivatives of the symbolic expression
        expr w.r.t. variables var_list and the scaling parameter rscale.

        Consider a Taylor multipole expansion:

        .. math::

            f (x - y) = \sum_{i = 0}^{\infty} (\partial_y^i f) (x - y) \big|_{y = c}
           \frac{(y - c)^i}{i!} .

        Now suppose we would like to use a scaled version :math:`g` of the
        kernel :math:`f`:

        .. math::

            \begin{eqnarray*}
              f (x) & = & g (x / \alpha),\\
              f^{(i)} (x) & = & \frac{1}{\alpha^i} g^{(i)} (x / \alpha) .
            \end{eqnarray*}

        where :math:`\alpha` is chosen to be on a length scale similar to
        :math:`x` (for example by choosing :math:`\alpha` proporitional to the
        size of the box for which the expansion is intended) so that :math:`x /
        \alpha` is roughly of unit magnitude, to avoid arithmetic issues with
        small arguments. This yields

        .. math::

            f (x - y) = \sum_{i = 0}^{\infty} (\partial_y^i g)
            \left( \frac{x - y}{\alpha} \right) \Bigg|_{y = c}
            \cdot
            \frac{(y - c)^i}{\alpha^i \cdot i!}.

        Observe that the :math:`(y - c)` term is now scaled to unit magnitude,
        as is the argument of :math:`g`.

        With :math:`\xi = x / \alpha`, we find

        .. math::

            \begin{eqnarray*}
              g (\xi) & = & f (\alpha \xi),\\
              g^{(i)} (\xi) & = & \alpha^i f^{(i)} (\alpha \xi) .
            \end{eqnarray*}

        Generically for all kernels, :math:`f^{(i)} (\alpha \xi)` is computable
        by taking a sufficient number of symbolic derivatives of :math:`f` and
        providing :math:`\alpha \xi = x` as the argument.

        Now, for some kernels, like :math:`f (x) = C \log x`, the powers of
        :math:`\alpha^i` from the chain rule cancel with the ones from the
        argument substituted into the kernel derivatives:

        .. math::

            g^{(i)} (\xi) = \alpha^i f^{(i)} (\alpha \xi) = C' \cdot \alpha^i \cdot
            \frac{1}{(\alpha x)^i} \quad (i > 0),

        making them what you might call *scale-invariant*.

        This derivative taker returns :math:`g^{(i)}(\xi) = \alpha^i f^{(i)}`
        given :math:`f^{(0)}` as *expr* and :math:`\alpha` as :attr:`rscale`.
        """

        assert isinstance(expr, sym.Basic)
        self.var_list = var_list
        zero_mi = (0,) * len(var_list)
        self.cache_by_mi = {zero_mi: expr}
        self.rscale = rscale
        self.sac = sac
        self.dim = len(self.var_list)
        self.orig_expr = expr

    def mi_dist(self, a, b):
        return np.array(a, dtype=int) - np.array(b, dtype=int)

    def diff(self, mi):
        """Take the derivative of the expression represented by
        :class:`ExprDerivativeTaker`.

        :param mi: multi-index representing the derivative
        """
        try:
            return self.cache_by_mi[mi]
        except KeyError:
            pass

        current_mi = self.get_closest_cached_mi(mi)
        expr = self.cache_by_mi[current_mi]

        for next_deriv, next_mi in self.get_derivative_taking_sequence(
                current_mi, mi):
            expr = expr.diff(next_deriv) * self.rscale
            self.cache_by_mi[next_mi] = expr

        return expr

    def get_derivative_taking_sequence(self, start_mi, end_mi):
        current_mi = np.array(start_mi, dtype=int)
        for idx, (mi_i, vec_i) in enumerate(
                zip(self.mi_dist(end_mi, start_mi), self.var_list)):
            for i in range(1, 1 + mi_i):
                current_mi[idx] += 1
                yield vec_i, tuple(current_mi)

    def get_closest_cached_mi(self, mi):
        return min((other_mi
                for other_mi in self.cache_by_mi.keys()
                if (np.array(mi) >= np.array(other_mi)).all()),
            key=lambda other_mi: sum(self.mi_dist(mi, other_mi)))


class LaplaceDerivativeTaker(ExprDerivativeTaker):
    """Specialized derivative taker for Laplace potential.
    """

    def __init__(self, expr, var_list, rscale=1, sac=None):
        super(LaplaceDerivativeTaker, self).__init__(expr, var_list, rscale, sac)
        self.scaled_var_list = [add_to_sac(self.sac, v/rscale) for v in var_list]
        self.scaled_r = add_to_sac(self.sac,
                sym.sqrt(sum(v**2 for v in self.scaled_var_list)))

    def diff(self, mi):
        """
        Implements the algorithm described in [Fernando2021] to take cartesian
        derivatives of Laplace potential using recurrences. Cost of each derivative
        is amortized constant.

        .. [Fernando2021]: Fernando, I., Klöckner, A., 2021. Automatic Synthesis of
                           Low Complexity Translation Operators for the Fast
                           Multipole Method. In preparation.
        """
        # Return zero for negative values. Makes the algorithm readable.
        if min(mi) < 0:
            return 0
        try:
            return self.cache_by_mi[mi]
        except KeyError:
            pass

        dim = self.dim
        if max(mi) == 1:
            return ExprDerivativeTaker.diff(self, mi)
        d = -1
        for i in range(dim):
            if mi[i] >= 2:
                d = i
                break
        assert d >= 0
        expr = 0
        for i in range(dim):
            mi_minus_one = list(mi)
            mi_minus_one[i] -= 1
            mi_minus_one = tuple(mi_minus_one)
            mi_minus_two = list(mi)
            mi_minus_two[i] -= 2
            mi_minus_two = tuple(mi_minus_two)
            x = self.scaled_var_list[i]
            n = mi[i]
            if i == d:
                if dim == 3:
                    expr -= (2*n - 1) * x * self.diff(mi_minus_one)
                    expr -= (n - 1)**2 * self.diff(mi_minus_two)
                else:
                    expr -= 2 * x * (n - 1) * self.diff(mi_minus_one)
                    expr -= (n - 1) * (n - 2) * self.diff(mi_minus_two)
                    if n == 2 and sum(mi) == 2:
                        expr += 1
            else:
                expr -= 2 * n * x * self.diff(mi_minus_one)
                expr -= n * (n - 1) * self.diff(mi_minus_two)
        expr /= self.scaled_r**2
        expr = add_to_sac(self.sac, expr)
        self.cache_by_mi[mi] = expr
        return expr


class RadialDerivativeTaker(ExprDerivativeTaker):
    """Specialized derivative taker for radial expressions.
    """

    def __init__(self, expr, var_list, rscale=1, sac=None):
        """
        Takes the derivatives of a radial function.
        """
        import sumpy.symbolic as sym
        super(RadialDerivativeTaker, self).__init__(expr, var_list, rscale, sac)
        empty_mi = (0,) * len(var_list)
        self.cache_by_mi_q = {(empty_mi, 0): expr}
        self.r = sym.sqrt(sum(v**2 for v in var_list))
        rsym = sym.Symbol("_r")
        r_expr = expr.xreplace({self.r**2: rsym**2})
        self.is_radial = not any(r_expr.has(v) for v in var_list)
        self.var_list_multiplied = [add_to_sac(sac, v * rscale) for v in var_list]

    def diff(self, mi, q=0):
        """
        Implements the algorithm described in [Tausch2003] to take cartesian
        derivatives of radial functions using recurrences. Cost of each derivative
        is amortized linear in the degree.

        .. [Tausch2003]: Tausch, J., 2003. The fast multipole method for arbitrary
                         Green's functions.
                         Contemporary Mathematics, 329, pp.307-314.
        """
        if not self.is_radial:
            assert q == 0
            return ExprDerivativeTaker.diff(self, mi)

        try:
            return self.cache_by_mi_q[(mi, q)]
        except KeyError:
            pass

        for i in range(self.dim):
            if mi[i] == 1:
                mi_minus_one = list(mi)
                mi_minus_one[i] = 0
                mi_minus_one = tuple(mi_minus_one)
                expr = self.var_list_multiplied[i] * self.diff(mi_minus_one, q=q+1)
                self.cache_by_mi_q[(mi, q)] = expr
                return expr

        for i in range(self.dim):
            if mi[i] >= 2:
                mi_minus_one = list(mi)
                mi_minus_one[i] -= 1
                mi_minus_one = tuple(mi_minus_one)
                mi_minus_two = list(mi)
                mi_minus_two[i] -= 2
                mi_minus_two = tuple(mi_minus_two)
                expr = (mi[i]-1)*self.diff(mi_minus_two, q=q+1) * self.rscale ** 2
                expr += self.var_list_multiplied[i] * self.diff(mi_minus_one, q=q+1)
                expr = add_to_sac(self.sac, expr)
                self.cache_by_mi_q[(mi, q)] = expr
                return expr

        assert mi == (0,)*self.dim
        assert q > 0

        prev_expr = self.diff(mi, q=q-1)
        # Need to get expr.diff(r)/r, but we can only do expr.diff(x)
        # Use expr.diff(x) = expr.diff(r) * x / r
        expr = prev_expr.diff(self.var_list[0])/self.var_list[0]
        # We need to distribute the division above
        expr = expr.expand(deep=False)
        self.cache_by_mi_q[(mi, q)] = expr
        return expr


class HelmholtzDerivativeTaker(RadialDerivativeTaker):
    """Specialized derivative taker for Helmholtz potential.
    """

    def diff(self, mi, q=0):
        import sumpy.symbolic as sym
        if q < 2 or mi != (0,)*self.dim:
            return RadialDerivativeTaker.diff(self, mi, q)
        try:
            return self.cache_by_mi_q[(mi, q)]
        except KeyError:
            pass

        if self.dim == 2:
            # See https://dlmf.nist.gov/10.6.E6
            # and https://dlmf.nist.gov/10.6#E1
            k = self.orig_expr.args[1] / self.r
            expr = -  2 * (q - 1) * self.diff(mi, q - 1)
            expr += - k**2 * self.diff(mi, q - 2)
            expr /= self.r**2
        else:
            # See reference [Tausch2003] in RadialDerivativeTaker.diff
            k = (self.orig_expr * self.r).args[-1] / sym.I / self.r
            expr = -(2*q - 1)/self.r**2 * self.diff(mi, q - 1)
            expr += -k**2 / self.r * self.diff(mi, q - 2)
        self.cache_by_mi_q[(mi, q)] = expr
        return expr


@tag_dataclass
class DifferentiatedExprDerivativeTaker:
    """Implements the :class:`ExprDerivativeTaker` interface
    for an expression that is itself a linear combination of
    derivatives of a base expression. To take the actual derivatives,
    it makes use of an underlying derivative taker *taker*.

    .. attribute:: taker
        A :class:`ExprDerivativeTaker` for the base expression.

    .. attribute:: derivative_transformation
        A dictionary mapping a derivative multi-index to a coefficient.
        The expression represented by this derivative taker is the linear
        combination of the derivatives of the expression for the
        base expression.
    """
    taker: ExprDerivativeTaker
    derivative_transformation: dict

    def diff(self, mi, save_intermediate=lambda x: x):
        # By passing `rscale` to the derivative taker we are taking a scaled
        # version of the derivative which is `expr.diff(mi)*rscale**sum(mi)`
        # which might be implemented efficiently for kernels like Laplace.
        # One caveat is that we are taking more derivatives because of
        # :attr:`derivative_transformation` which would multiply the
        # expression by more `rscale`s than necessary. This is corrected by
        # dividing by `rscale`.
        max_order = max(sum(extra_mi) for extra_mi in
                self.derivative_transformation.keys())

        result = sum(
            coeff * self.taker.diff(add_mi(mi, extra_mi))
            / self.taker.rscale ** (sum(extra_mi) - max_order)
            for extra_mi, coeff in self.derivative_transformation.items())

        return result * save_intermediate(1 / self.taker.rscale ** max_order)

# }}}

# {{{ get variables

class GatherAllVariables(WalkMapper):
    def __init__(self):
        self.vars = set()

    def map_variable(self, expr):
        self.vars.add(expr)


def get_all_variables(exprs):
    mapper = GatherAllVariables()
    for expr in exprs:
        mapper(expr)
    return mapper.vars

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
    from pytools.obj_array import obj_array_vectorize

    from pyopencl.array import to_device

    def to_dev(ary):
        return to_device(queue, ary)

    return obj_array_vectorize(to_dev, vec)


def vector_from_device(queue, vec):
    from pytools.obj_array import obj_array_vectorize

    def from_dev(ary):
        from numbers import Number
        if isinstance(ary, (np.number, Number)):
            # zero, most likely
            return ary

        return ary.get(queue=queue)

    return obj_array_vectorize(from_dev, vec)


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


class KernelComputation:
    """Common input processing for kernel computations."""

    def __init__(self, ctx, target_kernels, source_kernels, strength_usage,
            value_dtypes, name, device=None):
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
            for knl in target_kernels:
                if knl.is_complex_valued:
                    value_dtypes.append(np.complex128)
                else:
                    value_dtypes.append(np.float64)

        if not isinstance(value_dtypes, (list, tuple)):
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

        if device is None:
            device = ctx.devices[0]

        self.context = ctx
        self.device = device

        self.source_kernels = tuple(source_kernels)
        self.target_kernels = tuple(target_kernels)
        self.value_dtypes = value_dtypes
        self.strength_usage = strength_usage
        self.strength_count = strength_count

        self.name = name or self.default_name

    def get_kernel_scaling_assignments(self):
        from sumpy.symbolic import SympyToPymbolicMapper
        sympy_conv = SympyToPymbolicMapper()

        import loopy as lp
        return [
                lp.Assignment(id=None,
                    assignee="knl_%d_scaling" % i,
                    expression=sympy_conv(kernel.get_global_scaling_const()),
                    temp_var_type=lp.Optional(dtype),
                    tags=frozenset([ScalingAssignmentTag()]))
                for i, (kernel, dtype) in enumerate(
                    zip(self.target_kernels, self.value_dtypes))]

# }}}


# {{{


def _to_host(x, queue=None):
    if isinstance(x, cl.array.Array):
        queue = queue or x.queue
        return x.get(queue)
    return x


class BlockIndexRanges:
    """Convenience class for working with blocks of a global array.

    .. attribute:: indices

        A list of not necessarily continuous or increasing integers
        representing the indices of a global array. The individual blocks are
        delimited using :attr:`ranges`.

    .. attribute:: ranges

        A list of nondecreasing integers used to index into :attr:`indices`.
        A block :math:`i` can be retrieved using
        `indices[ranges[i]:ranges[i + 1]]`.

    .. automethod:: block_shape
    .. automethod:: get
    .. automethod:: take
    """

    def __init__(self, cl_context, indices, ranges):
        self.cl_context = cl_context
        self.indices = indices
        self.ranges = ranges

    @property
    @memoize_method
    def _ranges(self):
        with cl.CommandQueue(self.cl_context) as queue:
            return _to_host(self.ranges, queue=queue)

    @property
    def nblocks(self):
        return self.ranges.shape[0] - 1

    def block_shape(self, i):
        return (self._ranges[i + 1] - self._ranges[i],)

    def block_indices(self, i):
        return self.indices[self._ranges[i]:self._ranges[i + 1]]

    def get(self, queue=None):
        return BlockIndexRanges(self.cl_context,
                                _to_host(self.indices, queue=queue),
                                _to_host(self.ranges, queue=queue))

    def take(self, x, i):
        """Return the subset of a global array `x` that is defined by
        the :attr:`indices` in block :math:`i`.
        """

        return x[self.block_indices(i)]


class MatrixBlockIndexRanges:
    """Keep track of different ways to index into matrix blocks.

    .. attribute:: row

        A :class:`BlockIndexRanges` encapsulating row block indices.

    .. attribute:: col

        A :class:`BlockIndexRanges` encapsulating column block indices.

    .. automethod:: block_shape
    .. automethod:: block_take
    .. automethod:: get
    .. autoattribute:: linear_row_indices
    .. automethod:: take

    """

    def __init__(self, cl_context, row, col):
        self.cl_context = cl_context
        self.row = row
        self.col = col
        assert self.row.nblocks == self.col.nblocks

        self.blkranges = np.cumsum([0] + [
            self.row.block_shape(i)[0] * self.col.block_shape(i)[0]
            for i in range(self.row.nblocks)])

        if isinstance(self.row.indices, cl.array.Array):
            with cl.CommandQueue(self.cl_context) as queue:
                self.blkranges = \
                    cl.array.to_device(queue, self.blkranges).with_queue(None)

    @property
    def nblocks(self):
        return self.row.nblocks

    def block_shape(self, i):
        return self.row.block_shape(i) + self.col.block_shape(i)

    def block_indices(self, i):
        return (self.row.block_indices(i),
                self.col.block_indices(i))

    @property
    def linear_row_indices(self):
        r, _ = self._linear_indices()
        return r

    @property
    def linear_col_indices(self):
        _, c = self._linear_indices()
        return c

    @property
    def linear_ranges(self):
        return self.blkranges

    def get(self, queue=None):
        """Transfer data to the host. Only the initial given data is
        transfered, not the arrays returned by :meth:`linear_row_indices` and
        friends.

        :return: a copy of `self` in which all data lives on the host, i.e.
                 all :class:`pyopencl.array.Array` instances are replaces by
                 :class:`numpy.ndarray` instances.
        """
        return MatrixBlockIndexRanges(self.cl_context,
                row=self.row.get(queue=queue),
                col=self.col.get(queue=queue))

    def take(self, x, i):
        """Retrieve a block from a global matrix.

        :arg x: a 2D :class:`numpy.ndarray`.
        :arg i: block index.
        :return: requested block from the matrix.
        """

        if isinstance(self.row.indices, cl.array.Array) or \
                isinstance(self.col.indices, cl.array.Array):
            raise ValueError("CL `Array`s are not supported."
                    "Use MatrixBlockIndexRanges.get() and then view into matrices.")

        irow, icol = self.block_indices(i)
        return x[np.ix_(irow, icol)]

    def block_take(self, x, i):
        """Retrieve a block from a linear representation of the matrix blocks.
        A linear representation of the matrix blocks can be obtained, or
        should be consistent with

        .. code-block:: python

            i = index.linear_row_indices()
            j = index.linear_col_indices()
            linear_blks = global_mat[i, j]

            for k in range(index.nblocks):
                assert np.allclose(index.block_take(linear_blks, k),
                                   index.take(global_mat, k))

        :arg x: a 1D :class:`numpy.ndarray`.
        :arg i: block index.
        :return: requested block, reshaped into a 2D array.
        """

        iblk = np.s_[self.blkranges[i]:self.blkranges[i + 1]]
        return x[iblk].reshape(*self.block_shape(i))

    @memoize_method
    def _linear_indices(self):
        """
        :return: a tuple of `(rowindices, colindices)` that can be
            used to provide linear indexing into a set of matrix blocks. These
            index arrays are just the concatenated Cartesian products of all
            the block arrays described by :attr:`row` and :attr:`col`.

            They can be used to index directly into a matrix as follows:

            .. code-block:: python

                mat[rowindices[blkranges[i]:blkranges[i + 1]],
                    colindices[blkranges[i]:blkranges[i + 1]]]

            The same block can be obtained more easily using

            .. code-block:: python

                index.view(mat, i).reshape(-1)
        """

        @memoize_in(self, "block_index_knl")
        def _build_index():
            loopy_knl = lp.make_kernel([
                "{[irange]: 0 <= irange < nranges}",
                "{[itgt, isrc]: 0 <= itgt < ntgtblock and 0 <= isrc < nsrcblock}"
                ],
                """
                for irange
                    <> ntgtblock = tgtranges[irange + 1] - tgtranges[irange]
                    <> nsrcblock = srcranges[irange + 1] - srcranges[irange]

                    for itgt, isrc
                        <> imat = blkranges[irange] + (nsrcblock * itgt + isrc)

                        rowindices[imat] = tgtindices[tgtranges[irange] + itgt] \
                            {id_prefix=write_index}
                        colindices[imat] = srcindices[srcranges[irange] + isrc] \
                            {id_prefix=write_index}
                    end
                end
                """,
                [
                    lp.GlobalArg("blkranges", None, shape="nranges + 1"),
                    lp.GlobalArg("rowindices", None, shape="nresults"),
                    lp.GlobalArg("colindices", None, shape="nresults"),
                    lp.ValueArg("nresults", None),
                    "..."
                ],
                name="block_index_knl",
                default_offset=lp.auto,
                assumptions="nranges>=1",
                silenced_warnings="write_race(write_index*)",
                lang_version=MOST_RECENT_LANGUAGE_VERSION)
            loopy_knl = lp.split_iname(loopy_knl, "irange", 128, outer_tag="g.0")

            return loopy_knl

        with cl.CommandQueue(self.cl_context) as queue:
            _, (rowindices, colindices) = _build_index()(queue,
                tgtindices=self.row.indices,
                srcindices=self.col.indices,
                tgtranges=self.row.ranges,
                srcranges=self.col.ranges,
                blkranges=self.blkranges,
                nresults=_to_host(self.blkranges[-1], queue=queue))
            return (rowindices.with_queue(None),
                    colindices.with_queue(None))

# }}}


# {{{ OrderedSet

# Source: https://code.activestate.com/recipes/576694-orderedset/
# Author: Raymond Hettinger
# License: MIT

try:
    from collections.abc import MutableSet
except ImportError:
    from collections import MutableSet


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
        return "{}({!r})".format(self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

# }}}


class KernelCacheWrapper:
    @memoize_method
    def get_cached_optimized_kernel(self, **kwargs):
        from sumpy import code_cache, CACHING_ENABLED, OPT_ENABLED

        if CACHING_ENABLED:
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
                logger.debug("{}: kernel cache hit [key={}]".format(
                    self.name, cache_key))
                return result
            except KeyError:
                pass

        logger.info("%s: kernel cache miss" % self.name)
        if CACHING_ENABLED:
            logger.info("{}: kernel cache miss [key={}]".format(
                self.name, cache_key))

        from pytools import MinRecursionLimit
        with MinRecursionLimit(3000):
            if OPT_ENABLED:
                knl = self.get_optimized_kernel(**kwargs)
            else:
                knl = self.get_kernel()

        if CACHING_ENABLED:
            code_cache.store_if_not_present(cache_key, knl)

        return knl

    @staticmethod
    def _allow_redundant_execution_of_knl_scaling(knl):
        from loopy.match import ObjTagged
        from sumpy.tools import ScalingAssignmentTag
        return lp.add_inames_for_unused_hw_axes(
                knl, within=ObjTagged(ScalingAssignmentTag()))


def my_syntactic_subs(expr, subst_dict):
    # Workaround for differing substitution semantics between sympy and symengine.
    # FIXME: This is a hack.
    from sumpy.symbolic import Basic, Subs, Derivative

    if not isinstance(expr, Basic):
        return expr

    elif expr.is_Symbol:
        return subst_dict.get(expr, expr)

    elif isinstance(expr, Subs):
        new_point = tuple(my_syntactic_subs(p, subst_dict) for p in expr.point)

        new_subst_dict = {
            var: subs for var, subs in subst_dict.items()
            if var not in expr.variables}

        new_expr = my_syntactic_subs(expr.expr, new_subst_dict)

        if new_point != expr.point or new_expr != expr.expr:
            return Subs(new_expr, expr.variables, new_point)

        return expr

    elif isinstance(expr, Derivative):
        new_expr = my_syntactic_subs(expr.expr, subst_dict)
        new_variables = my_syntactic_subs(expr.variables, subst_dict)

        if new_expr != expr.expr or any(new_var != var for new_var, var in
                                          zip(new_variables, expr.variables)):
            return Derivative(new_expr, *new_variables)

        return expr

    else:
        new_args = tuple(my_syntactic_subs(arg, subst_dict) for arg in expr.args)

        if any(new_arg != arg for arg, new_arg in zip(expr.args, new_args)):
            return expr.func(*new_args)

        return expr


def is_obj_array_like(ary):
    return (
            isinstance(ary, (tuple, list))
            or (isinstance(ary, np.ndarray) and ary.dtype.char == "O"))


def reduced_row_echelon_form(m):
    """Calculates a reduced row echelon form of a
    matrix `m`.

    :arg m: a 2D :class:`numpy.ndarray` or a list of lists or a sympy Matrix
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
            if mat[k, i] != 0 and pivot == nrows:
                pivot = k
            if abs(mat[k, i]) == 1:
                pivot = k
                break
        if pivot == nrows:
            continue
        if pivot != index:
            mat[[pivot, index], :] = mat[[index, pivot], :]

        pivot_cols.append(i)
        scale = mat[index, i]
        if isinstance(scale, (int, sym.Integer)):
            scale = int(scale)

        for j in range(mat.shape[1]):
            elem = mat[index, j]
            if isinstance(scale, int) and isinstance(elem, (int, sym.Integer)):
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


def nullspace(m):
    """Calculates the nullspace of a matrix `m`.

    :arg m: a 2D :class:`numpy.ndarray` or a list of lists or a sympy Matrix
    :return: nullspace of `m` as a 2D :class:`numpy.ndarray`
    """
    mat, pivot_cols = reduced_row_echelon_form(m)
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


def find_linear_relationship(matrix):
    """
    This method does elementary row operations to figure out the first row
    which is linearly dependent on the previous rows. Partial pivoting is not done
    to find the row with the lowest degree.
    """
    ncols = matrix.shape[1]
    nrows = min(matrix.shape[0], ncols+1)
    augment = np.eye(nrows, nrows, dtype=matrix.dtype)
    mat = np.hstack((matrix[:nrows, :], augment))
    for i in range(nrows):
        for j in range(ncols):
            if mat[i, j] != 0:
                col = j
                break
        else:
            pde_dict = {}
            for col in range(ncols, ncols+nrows):
                if mat[i, col] != 0:
                    pde_dict[col-ncols] = mat[i, col]
            return pde_dict
        for j in range(i+1, nrows):
            mat[j, :] = mat[j, :]*mat[i, col] - mat[i, :]*mat[j, col]
    return {}

# vim: fdm=marker
