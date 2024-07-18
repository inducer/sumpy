from __future__ import annotations


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

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Hashable, Sequence

import loopy as lp
import pymbolic.primitives as prim
from pytools import memoize_method

import sumpy.symbolic as sym
from sumpy.kernel import Kernel
from sumpy.tools import add_mi


logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: ExpansionBase

Expansion Wranglers
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ExpansionTermsWrangler
.. autoclass:: FullExpansionTermsWrangler
.. autoclass:: LinearPDEBasedExpansionTermsWrangler

Expansion Factories
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ExpansionFactoryBase
.. autoclass:: DefaultExpansionFactory
.. autoclass:: VolumeTaylorExpansionFactory
"""


# {{{ expansion base

class ExpansionBase(ABC):
    """
    .. attribute:: kernel
    .. attribute:: order
    .. attribute:: use_rscale

    .. automethod:: get_coefficient_identifiers
    .. automethod:: coefficients_from_source
    .. automethod:: coefficients_from_source_vec
    .. automethod:: loopy_expansion_formation
    .. automethod:: evaluate
    .. automethod:: loopy_evaluator

    .. automethod:: with_kernel
    .. automethod:: copy

    .. automethod:: __len__
    .. automethod:: __eq__
    .. automethod:: __ne__
    """
    init_arg_names: tuple[str, ...] = ("kernel", "order", "use_rscale")

    def __init__(self,
            kernel: Kernel,
            order: int,
            use_rscale: bool | None = None) -> None:
        # Don't be tempted to remove target derivatives here.
        # Line Taylor QBX can't do without them, because it can't
        # apply those derivatives to the expanded quantity.

        self.kernel = kernel
        self.order = order

        if use_rscale is None:
            use_rscale = True

        self.use_rscale = use_rscale

    # {{{ propagate kernel interface

    # This is here to conform this to enough of the kernel interface
    # to make it fit into sumpy.qbx.LayerPotential.

    @property
    def dim(self) -> int:
        return self.kernel.dim

    @property
    def is_complex_valued(self) -> bool:
        return self.kernel.is_complex_valued

    def get_code_transformer(self):
        return self.kernel.get_code_transformer()

    def get_global_scaling_const(self):
        return self.kernel.get_global_scaling_const()

    def get_args(self):
        return self.kernel.get_args()

    def get_source_args(self):
        return self.kernel.get_source_args()

    # }}}

    # {{{ abstract interface

    @abstractmethod
    def get_coefficient_identifiers(self) -> list[Hashable]:
        """
        :returns: the identifiers of the coefficients that actually get stored.
        """

    @abstractmethod
    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        """Form an expansion from a source point.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """

    def coefficients_from_source_vec(self,
            kernels, avec, bvec, rscale, weights, sac=None):
        """Form an expansion with a linear combination of kernels and weights.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        result = [0]*len(self)
        for knl, weight in zip(kernels, weights):
            coeffs = self.coefficients_from_source(knl, avec, bvec, rscale, sac=sac)
            for i in range(len(result)):
                result[i] += weight * coeffs[i]
        return result

    def loopy_expansion_formation(
            self, kernels: Sequence[Kernel],
            strength_usage: Sequence[int], nstrengths: int) -> lp.TranslationUnit:
        """
        :returns: a :mod:`loopy` kernel that returns the coefficients
            for the expansion given by *kernels* with each kernel using
            the strength given by *strength_usage*.
        """
        from sumpy.expansion.loopy import make_p2e_loopy_kernel
        return make_p2e_loopy_kernel(self, kernels, strength_usage, nstrengths)

    @abstractmethod
    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        """
        :returns: a :mod:`sympy` expression corresponding
            to the evaluated expansion with the coefficients
            in *coeffs*.
        """

    def loopy_evaluator(self, kernels: Sequence[Kernel]) -> lp.TranslationUnit:
        """
        :returns: a :mod:`loopy` kernel that returns the evaluated
            target transforms of the potential given by *kernels*.
        """
        from sumpy.expansion.loopy import make_e2p_loopy_kernel
        return make_e2p_loopy_kernel(self, kernels)

    # }}}

    # {{{ copy

    def with_kernel(self, kernel: Kernel) -> ExpansionBase:
        return type(self)(kernel, self.order, self.use_rscale)

    def copy(self, **kwargs) -> ExpansionBase:
        new_kwargs = {
                name: getattr(self, name)
                for name in self.init_arg_names}

        for name in self.init_arg_names:
            new_kwargs[name] = kwargs.pop(name, getattr(self, name))

        if kwargs:
            raise TypeError(
                "unexpected keyword arguments '{}'".format(", ".join(kwargs)))

        return type(self)(**new_kwargs)

    # }}}

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.kernel)
        key_builder.rec(key_hash, self.order)
        key_builder.rec(key_hash, self.use_rscale)

    def __len__(self):
        return len(self.get_coefficient_identifiers())

    def __eq__(self, other):
        return (
                type(self) is type(other)
                and self.kernel == other.kernel
                and self.order == other.order
                and self.use_rscale == other.use_rscale)

    def __ne__(self, other):
        return not self.__eq__(other)

# }}}


# {{{ expansion terms wrangler

class ExpansionTermsWrangler(ABC):
    """
    .. attribute:: order
    .. attribute:: dim
    .. attribute:: max_mi

    .. automethod:: get_coefficient_identifiers
    .. automethod:: get_full_kernel_derivatives_from_stored
    .. automethod:: get_stored_mpole_coefficients_from_full

    .. automethod:: get_full_coefficient_identifiers
    """
    init_arg_names: tuple[str, ...] = ("order", "dim", "max_mi")

    def __init__(self,
            order: int,
            dim: int,
            max_mi: tuple[int, ...] | None = None) -> None:
        self.order = order
        self.dim = dim
        self.max_mi = max_mi

    # {{{ abstract interface

    @abstractmethod
    def get_coefficient_identifiers(self) -> list[tuple[int, ...]]:
        pass

    @abstractmethod
    def get_full_kernel_derivatives_from_stored(self,
            stored_kernel_derivatives, rscale, sac=None):
        pass

    @abstractmethod
    def get_stored_mpole_coefficients_from_full(self,
            full_mpole_coefficients, rscale, sac=None):
        pass

    # }}}

    @memoize_method
    def get_full_coefficient_identifiers(self) -> list[Hashable]:
        """
        Returns identifiers for every coefficient in the complete expansion.
        """
        from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
        )

        res = sorted(gnitstam(self.order, self.dim), key=sum)

        if self.max_mi is None:
            return res

        return [mi for mi in res if
            all(mi[i] <= self.max_mi[i] for i in range(self.dim))]

    def copy(self, **kwargs):
        new_kwargs = {
                name: getattr(self, name)
                for name in self.init_arg_names}

        for name in self.init_arg_names:
            new_kwargs[name] = kwargs.pop(name, getattr(self, name))

        if kwargs:
            raise TypeError(
                "unexpected keyword arguments '{}'".format(", ".join(kwargs)))

        return type(self)(**new_kwargs)

    # {{{ hyperplane helpers

    def _get_mi_hyperpplanes(self) -> list[tuple[int, int]]:
        r"""
        Coefficient storage is organized into "hyperplanes" in multi-index
        space. Potentially only a subset of these hyperplanes contain
        coefficients that need to be stored. This routine returns that
        subset, which is then used in :meth:`_split_coeffs_into_hyperplanes`
        to appropriately partition the coefficients into segments corresponding
        to the hyperplanes, settling any overlap in the process.

        Returns a list of hyperplane where each hyperplane is represented by
        a tuple of integers. The first integer `d` is the axis number that the
        hyperplane is orthogonal to and the second integer is the `d`-th
        component of the lattice point where the hyperplane intersects the
        axis `d`.
        """
        d = self.dim - 1
        hyperplanes = [(d, const) for const in range(self.order + 1)]
        return hyperplanes

    @memoize_method
    def _split_coeffs_into_hyperplanes(
            self) -> list[tuple[int, list[tuple[int, ...]]]]:
        r"""
        This splits the coefficients into :math:`O(p)` disjoint sets
        so that for each set, all the identifiers have the form,
        :math:`(m_1, m_2, ..., m_{j-1}, c, m_{j+1}, ... , m_{\text{dim}})`
        where :math:`c` is a constant. Geometrically, each set is an axis-aligned
        hyperplane.

        If this is an instance of :class:`LinearPDEBasedExpansionTermsWrangler`,
        then the number of sets will be :math:`O(1)`.

        In the returned object ``[(axis, [mi_1, mi2, ...]), ...]``,
        each element in the outer list represents a hyperplane. Each element
        is a 2-tuple where the first element in the tuple is the axis number *axis*
        to which the hyperplane is orthogonal. The second element in the tuple
        is a list of multi-indices in the hyperplane corresponding to the stored
        coefficients.

        E.g. for Laplace 3D order 4, the result could be::
        [
          (2, [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0),
               (2, 1, 0), (0, 2, 0), (1, 2, 0), (0, 3, 0)]),
          (2, [(0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]),
        ]
        """
        hyperplanes = self._get_mi_hyperpplanes()
        res = []
        seen_mis = set()
        for d, const in hyperplanes:
            coeffs_in_hyperplane = []
            for mi in self.get_coefficient_identifiers():
                # Check if the multi-index is in this hyperplane and
                # if it is not in any of the hyperplanes we saw before
                # (This rejects coefficients that might be in multiple
                # hyperplanes, such as near the origin.)
                if mi[d] == const and mi not in seen_mis:
                    coeffs_in_hyperplane.append(mi)
                    seen_mis.add(mi)
            res.append((d, coeffs_in_hyperplane))

        return res

    # }}}


class FullExpansionTermsWrangler(ExpansionTermsWrangler):

    def get_storage_index(self, mi, order=None):
        if not order:
            order = sum(mi)
        if self.dim == 3:
            return (order*(order + 1)*(order + 2))//6 + \
                    (order + 2)*mi[2] - (mi[2]*(mi[2] + 1))//2 + mi[1]
        elif self.dim == 2:
            return (order*(order + 1))//2 + mi[1]
        else:
            raise NotImplementedError

    def get_coefficient_identifiers(self):
        return super().get_full_coefficient_identifiers()

    def get_full_kernel_derivatives_from_stored(self,
            stored_kernel_derivatives, rscale, sac=None):
        return stored_kernel_derivatives

    def get_stored_mpole_coefficients_from_full(self,
            full_mpole_coefficients, rscale, sac=None):
        return self.get_full_kernel_derivatives_from_stored(
            full_mpole_coefficients, rscale, sac=sac)

    @memoize_method
    def _get_mi_ordering_key_and_axis_permutation(self):
        """
        Returns a degree lexicographic order as a callable that can be used as a
        ``sort`` key on multi-indices and a permutation of the axis ordered
        from the slowest varying axis to the fastest varying axis of the
        multi-indices when sorted.
        """
        from sumpy.expansion.diff_op import DerivativeIdentifier

        axis_permutation = list(reversed(list(range(self.dim))))

        def mi_key(ident):
            if isinstance(ident, DerivativeIdentifier):
                mi = ident.mi
            else:
                mi = ident
            return (sum(mi), *list(reversed(mi)))

        return mi_key, axis_permutation
# }}}


# {{{ sparse matrix-vector multiplication

class CSEMatVecOperator:
    """
    A class to facilitate a fast matrix vector multiplication with
    common subexpression eliminated. In compressed Taylor
    series, the compression matrix's operation count can be
    reduced to `O(p**d)` from `O(p**{2d-1})` using CSE.
    Each row in the matrix is represented by a linear combination
    of values from input vector and a linear combination of values
    from the output vector.

    .. attribute:: from_input_coeffs_by_row

        An object of type ``List[List[Tuple[int, Any]]]``. Each element
        in the list represents a row of the matrix using a linear combination
        of values from the input vector. Each element has the form
        ``(index of input vector, coeff)``.

        Number of rows in the matrix represented is equal to the
        length of the `from_input_coeffs_by_row` list.

    .. attribute:: from_output_coeffs_by_row

        An object of type ``List[List[Tuple[int, Any]]]``. Each element
        in the list represents a row of the matrix using a linear combination
        of values from the output vector. Each element has the form
        ``(index of output vector, coeff)``.

    .. attribute:: shape

        Shape of the matrix as a tuple.
    """

    def __init__(self, from_input_coeffs_by_row, from_output_coeffs_by_row, shape):
        self.from_input_coeffs_by_row = from_input_coeffs_by_row
        self.from_output_coeffs_by_row = from_output_coeffs_by_row
        self.shape = shape
        assert len(self.from_input_coeffs_by_row) == shape[0]
        assert len(self.from_output_coeffs_by_row) == shape[0]

    def matvec(self, inp, wrap_intermediate=lambda x: x):
        """
        :arg inp: vector for the matrix vector multiplication

        :arg wrap_intermediate: a function to wrap intermediate expressions
             If not given, the number of operations might grow in the
             final expressions in the vector resulting in an expensive matvec.
        """
        assert len(inp) == self.shape[1]
        out = []
        for i in range(self.shape[0]):
            value = 0
            for input_index, coeff in self.from_input_coeffs_by_row[i]:
                value += inp[input_index] * coeff
            for output_index, coeff in self.from_output_coeffs_by_row[i]:
                value += out[output_index] * coeff
            out.append(wrap_intermediate(value))
        return out

    def transpose_matvec(self, inp, wrap_intermediate=lambda x: x):
        assert len(inp) == self.shape[0]
        res = [0]*self.shape[1]
        expr_all = list(inp)
        for i in reversed(range(self.shape[0])):
            for output_index, coeff in self.from_output_coeffs_by_row[i]:
                expr_all[output_index] += expr_all[i] * coeff
                expr_all[output_index] = wrap_intermediate(expr_all[output_index])
            for input_index, coeff in self.from_input_coeffs_by_row[i]:
                res[input_index] += expr_all[i] * coeff
        return res

# }}}


# {{{ LinearPDEBasedExpansionTermsWrangler

class LinearPDEBasedExpansionTermsWrangler(ExpansionTermsWrangler):
    """
    .. automethod:: __init__
    """

    init_arg_names = ("order", "dim", "knl", "max_mi")

    def __init__(self,
            order: int, dim: int, knl: Kernel,
            max_mi: tuple[int, ...] | None = None) -> None:
        r"""
        :param order: order of the expansion
        :param dim: number of dimensions
        :param knl: kernel for the PDE
        """
        super().__init__(order, dim, max_mi)
        self.knl = knl

    def get_coefficient_identifiers(self):
        return self.stored_identifiers

    def get_full_kernel_derivatives_from_stored(self,
            stored_kernel_derivatives, rscale, sac=None):
        from sumpy.tools import add_to_sac

        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.matvec(stored_kernel_derivatives,
                lambda x: add_to_sac(sac, x))

    def get_stored_mpole_coefficients_from_full(self,
            full_mpole_coefficients, rscale, sac=None):
        from sumpy.tools import add_to_sac

        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.transpose_matvec(full_mpole_coefficients,
                lambda x: add_to_sac(sac, x))

    @property
    def stored_identifiers(self):
        stored_identifiers, _ = self.get_stored_ids_and_unscaled_projection_matrix()
        return stored_identifiers

    # If there exists an axis-normal hyperplane without a PDE (derivative)
    # multi-index on it, then the coefficients on that hyperplane *must* be
    # stored (because it cannot be reached by any PDE identities). To find
    # storage hyperplanes that reach a maximal (-ish?) number of coefficients,
    # look for on-axis PDE coefficient multi-indices, and start enumerating
    # hyperplanes normal to that axis.  Practically, this is done by reordering
    # the axes so that the axis with the on-axis coefficient comes first in the
    # multi-index tuple.
    @memoize_method
    def _get_mi_ordering_key_and_axis_permutation(self):
        """
        A degree lexicographic order with the slowest varying index depending on
        the PDE is used, returned as a callable that can be used as a
        ``sort`` key on multi-indices.
        A degree lexicographic ordering is needed for
        multipole-to-multipole translation to get lower error bounds.
        The slowest varying index is chosen such that the multipole-to-local
        translation cost is optimized.

        Also returns a permutation of the axis ordered from the slowest varying
        axis to the fastest varying axis of the multi-indices when sorted.
        """
        dim = self.dim
        deriv_id_to_coeff, = self.knl.get_pde_as_diff_op().eqs
        slowest_varying_index = dim - 1
        for ident in deriv_id_to_coeff:
            if ident.mi.count(0) == dim - 1:
                non_zero_index = next(i for i in range(self.dim) if ident.mi[i] != 0)
                slowest_varying_index = min(slowest_varying_index, non_zero_index)

        axis_permutation = list(range(self.dim))
        axis_permutation[0], axis_permutation[slowest_varying_index] = \
                slowest_varying_index, 0

        from sumpy.expansion.diff_op import DerivativeIdentifier

        def mi_key(ident):
            if isinstance(ident, DerivativeIdentifier):
                mi = ident.mi
            else:
                mi = ident
            key = [sum(mi)]
            for i in range(dim):
                key.append(mi[axis_permutation[i]])
            return tuple(key)

        return mi_key, axis_permutation

    def _get_mi_hyperpplanes(self) -> list[tuple[int, int]]:
        mis = self.get_full_coefficient_identifiers()
        mi_to_index = {mi: i for i, mi in enumerate(mis)}

        hyperplanes = []
        deriv_id_to_coeff, = self.knl.get_pde_as_diff_op().eqs

        if not all(ident.mi in mi_to_index for ident in deriv_id_to_coeff):
            # The order of the expansion is less than the order of the PDE.
            # Treat as if full expansion.
            hyperplanes = super()._get_mi_hyperpplanes()
        else:
            # Calculate the multi-index that appears last in in the PDE in
            # the degree lexicographic order given by
            # _get_mi_ordering_key_and_axis_permutation.
            ordering_key, _ = self._get_mi_ordering_key_and_axis_permutation()
            max_mi = max(deriv_id_to_coeff, key=ordering_key).mi
            hyperplanes = [(d, const)
                for d in range(self.dim)
                for const in range(max_mi[d])]

        return hyperplanes

    def get_full_coefficient_identifiers(self):
        identifiers = super().get_full_coefficient_identifiers()
        key, _ = self._get_mi_ordering_key_and_axis_permutation()
        return sorted(identifiers, key=key)

    def get_storage_index(self, mi, order=None):
        if not order:
            order = sum(mi)

        ordering_key, axis_permutation = \
                self._get_mi_ordering_key_and_axis_permutation()
        deriv_id_to_coeff, = self.knl.get_pde_as_diff_op().eqs
        max_mi = max(deriv_id_to_coeff, key=ordering_key).mi

        if all(m != 0 for m in max_mi):
            raise NotImplementedError("non-elliptic PDEs")

        c = max_mi[axis_permutation[0]]

        mi = list(mi)
        mi[axis_permutation[0]], mi[0] = mi[0], mi[axis_permutation[0]]

        if self.dim == 3:
            if all(isinstance(axis, int) for axis in mi):
                if order < c - 1:
                    return (order*(order + 1)*(order + 2))//6 + \
                        (order + 2)*mi[0] - (mi[0]*(mi[0] + 1))//2 + mi[1]
                else:
                    return (c*(c-1)*(c-2))//6 + (c * order * (2 + order - c)
                        + mi[0]*(3 - mi[0]+2*order))//2 + mi[1]
            else:
                return prim.If(prim.Comparison(order, "<", c - 1),
                    (order*(order + 1)*(order + 2))//6
                        + (order + 2)*mi[0] - (mi[0]*(mi[0] + 1))//2 + mi[1],
                    (c*(c-1)*(c-2))//6 + (c * order * (2 + order - c)
                        + mi[0]*(3 - mi[0]+2*order))//2 + mi[1]
                )
        elif self.dim == 2:
            if all(isinstance(axis, int) for axis in mi):
                if order < c - 1:
                    return (order*(order + 1))//2 + mi[0]
                else:
                    return (c*(c-1))//2 + c*(order - c + 1) + mi[0]
            else:
                return prim.If(prim.Comparison(order, "<", c - 1),
                    (order*(order + 1))//2 + mi[0],
                    (c*(c-1))//2 + c*(order - c + 1) + mi[0])
        else:
            raise NotImplementedError

    @memoize_method
    def get_stored_ids_and_unscaled_projection_matrix(self):
        from pytools import ProcessLogger
        plog = ProcessLogger(logger, "compute PDE for Taylor coefficients")

        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = {tuple(mi): i for
                                            (i, mi) in enumerate(mis)}

        diff_op = self.knl.get_pde_as_diff_op()
        assert len(diff_op.eqs) == 1
        mi_to_coeff = {k.mi: v for k, v in diff_op.eqs[0].items()}
        for ident in mi_to_coeff:
            if ident not in coeff_ident_enumerate_dict:
                # Order of the expansion is less than the order of the PDE.
                # In that case, the compression matrix is the identity matrix
                # and there's nothing to project
                from_input_coeffs_by_row = [[(i, 1)] for i in range(len(mis))]
                from_output_coeffs_by_row = [[] for _ in range(len(mis))]
                shape = (len(mis), len(mis))
                op = CSEMatVecOperator(from_input_coeffs_by_row,
                                       from_output_coeffs_by_row, shape)

                plog.done()

                return mis, op

        ordering_key, _ = self._get_mi_ordering_key_and_axis_permutation()
        max_mi = max((ident for ident in mi_to_coeff.keys()), key=ordering_key)
        max_mi_coeff = mi_to_coeff[max_mi]
        max_mi_mult = -1/sym.sympify(max_mi_coeff)

        def is_stored(mi):
            """
            A multi_index mi is not stored if mi >= max_mi
            """
            return any(mi[d] < max_mi[d] for d in range(self.dim))

        stored_identifiers = []

        from_input_coeffs_by_row = []
        from_output_coeffs_by_row = []
        for i, mi in enumerate(mis):
            # If the multi-index is to be stored, keep the projection matrix
            # entry empty
            if is_stored(mi):
                idx = len(stored_identifiers)
                stored_identifiers.append(mi)
                from_input_coeffs_by_row.append([(idx, 1)])
                from_output_coeffs_by_row.append([])
                continue
            diff = [mi[d] - max_mi[d] for d in range(self.dim)]

            # eg: u_xx + u_yy + u_zz is represented as
            # [((2, 0, 0), 1), ((0, 2, 0), 1), ((0, 0, 2), 1)]
            assignment = []
            for other_mi, coeff in mi_to_coeff.items():
                j = coeff_ident_enumerate_dict[add_mi(other_mi, diff)]
                if i == j:
                    # Skip the u_zz part here.
                    continue
                # PDE might not have max_mi_coeff = -1, divide by -max_mi_coeff
                # to get a relation of the form, u_zz = - u_xx - u_yy for Laplace 3D.
                assignment.append((j, coeff*max_mi_mult))
            from_input_coeffs_by_row.append([])
            from_output_coeffs_by_row.append(assignment)

        plog.done()

        logger.debug("number of Taylor coefficients was reduced from %d to %d",
                     len(self.get_full_coefficient_identifiers()),
                     len(stored_identifiers))

        shape = (len(mis), len(stored_identifiers))
        op = CSEMatVecOperator(from_input_coeffs_by_row,
                               from_output_coeffs_by_row, shape)
        return stored_identifiers, op

    @memoize_method
    def get_projection_matrix(self, rscale):
        r"""
        Return a :class:`CSEMatVecOperator` object which exposes a matrix vector
        multiplication operator for the projection matrix that expresses
        every derivative in terms of a set of "stored" derivatives.

        For example, for the PDE::

            u_xx + u_yy + u_zz = 0

        the coefficient matrix features the following entries::

                ... u_xx u_yy ... <= cols = only stored derivatives
                ==================
             ...| ...  ...  ... ...
                |
            u_zz| ...  -1   -1  ...

            ^ rows = one for every derivative

        the projection matrix `M` is the transpose of the coefficient matrix,
        so that
        .. math::

            c^{\text{local}}_{\text{full}} = M^T c^{\text{local}}_{\text{stored}}.\\
            c^{\text{mpole}}_{\text{stored}} = M c^{\text{mpole}}_{\text{full}}.
        """
        _, projection_matrix = \
            self.get_stored_ids_and_unscaled_projection_matrix()

        full_coeffs = self.get_full_coefficient_identifiers()

        projection_with_rscale = []
        for row, assignment in \
                enumerate(projection_matrix.from_output_coeffs_by_row):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[row])
            from_output_coeffs_with_rscale = []
            for k, coeff in assignment:
                diff = row_rscale - sum(full_coeffs[k])
                mult = rscale**diff
                from_output_coeffs_with_rscale.append((k, coeff * mult))
            projection_with_rscale.append(from_output_coeffs_with_rscale)

        shape = projection_matrix.shape
        return CSEMatVecOperator(projection_matrix.from_input_coeffs_by_row,
                                 projection_with_rscale, shape)


# }}}


# {{{ volume taylor expansion

# FIXME: This is called an expansion but doesn't inherit from ExpansionBase?

class VolumeTaylorExpansionMixin:
    expansion_terms_wrangler_class: ClassVar[type[ExpansionTermsWrangler]]
    expansion_terms_wrangler_cache: ClassVar[dict[Hashable, Any]] = {}

    @classmethod
    def get_or_make_expansion_terms_wrangler(cls, *key):
        """
        This stores the expansion terms wrangler at the class attribute level
        because recreating the expansion terms wrangler implicitly empties its
        caches.
        """
        try:
            wrangler = cls.expansion_terms_wrangler_cache[key]
        except KeyError:
            wrangler = cls.expansion_terms_wrangler_class(*key)
            cls.expansion_terms_wrangler_cache[key] = wrangler

        return wrangler

    @property
    def expansion_terms_wrangler(self):
        return self.get_or_make_expansion_terms_wrangler(
                *self.expansion_terms_wrangler_key)

    def get_coefficient_identifiers(self):
        """
        Returns the identifiers of the coefficients that actually get stored.
        """
        return self.expansion_terms_wrangler.get_coefficient_identifiers()

    def get_full_coefficient_identifiers(self):
        return self.expansion_terms_wrangler.get_full_coefficient_identifiers()

    @property
    @memoize_method
    def _storage_loc_dict(self):
        return {i: idx for idx, i in
                    enumerate(self.get_coefficient_identifiers())}

    def get_storage_index(self, i):
        return self._storage_loc_dict[i]


class VolumeTaylorExpansion(VolumeTaylorExpansionMixin):
    expansion_terms_wrangler_class = FullExpansionTermsWrangler

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel: Kernel, order: int, use_rscale: bool) -> None:
        self.expansion_terms_wrangler_key = (order, kernel.dim)


class LinearPDEConformingVolumeTaylorExpansion(VolumeTaylorExpansionMixin):
    expansion_terms_wrangler_class = LinearPDEBasedExpansionTermsWrangler

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel: Kernel, order: int, use_rscale: bool) -> None:
        self.expansion_terms_wrangler_key = (order, kernel.dim, kernel)


class LaplaceConformingVolumeTaylorExpansion(
        LinearPDEConformingVolumeTaylorExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("LaplaceConformingVolumeTaylorExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class HelmholtzConformingVolumeTaylorExpansion(
        LinearPDEConformingVolumeTaylorExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("HelmholtzConformingVolumeTaylorExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class BiharmonicConformingVolumeTaylorExpansion(
        LinearPDEConformingVolumeTaylorExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("BiharmonicConformingVolumeTaylorExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

# }}}


# {{{ expansion factory

class ExpansionFactoryBase(ABC):
    """
    .. automethod:: get_local_expansion_class
    .. automethod:: get_multipole_expansion_class
    """

    @abstractmethod
    def get_local_expansion_class(self, base_kernel):
        """
        :returns: a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """

    @abstractmethod
    def get_multipole_expansion_class(self, base_kernel):
        """
        :returns: a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """


class VolumeTaylorExpansionFactory(ExpansionFactoryBase):
    """An implementation of :class:`ExpansionFactoryBase` that uses Volume Taylor
    expansions for each kernel.
    """

    def get_local_expansion_class(self, base_kernel):
        """
        :returns: a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.local import VolumeTaylorLocalExpansion
        return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """
        :returns: a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
        return VolumeTaylorMultipoleExpansion


class DefaultExpansionFactory(ExpansionFactoryBase):
    """An implementation of :class:`ExpansionFactoryBase` that gives the 'best known'
    expansion for each kernel.
    """

    def get_local_expansion_class(self, base_kernel):
        """
        :returns: a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.local import (
            LinearPDEConformingVolumeTaylorLocalExpansion,
            VolumeTaylorLocalExpansion,
        )
        try:
            base_kernel.get_base_kernel().get_pde_as_diff_op()
            return LinearPDEConformingVolumeTaylorLocalExpansion
        except NotImplementedError:
            return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """
        :returns: a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.multipole import (
            LinearPDEConformingVolumeTaylorMultipoleExpansion,
            VolumeTaylorMultipoleExpansion,
        )
        try:
            base_kernel.get_base_kernel().get_pde_as_diff_op()
            return LinearPDEConformingVolumeTaylorMultipoleExpansion
        except NotImplementedError:
            return VolumeTaylorMultipoleExpansion

# }}}


# vim: fdm=marker
