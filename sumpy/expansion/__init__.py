from __future__ import division
from __future__ import absolute_import
from six.moves import range

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

import numpy as np
import logging
from pytools import memoize_method
import sumpy.symbolic as sym
from sumpy.tools import MiDerivativeTaker

__doc__ = """
.. autoclass:: ExpansionBase
"""

logger = logging.getLogger(__name__)


# {{{ base class

class ExpansionBase(object):

    def __init__(self, kernel, order):
        # Don't be tempted to remove target derivatives here.
        # Line Taylor QBX can't do without them, because it can't
        # apply those derivatives to the expanded quantity.

        self.kernel = kernel
        self.order = order

    # {{{ propagate kernel interface

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def is_complex_valued(self):
        return self.kernel.is_complex_valued

    def prepare_loopy_kernel(self, loopy_knl):
        return self.kernel.prepare_loopy_kernel(loopy_knl)

    def get_code_transformer(self):
        return self.kernel.get_code_transformer()

    def get_scaling(self):
        return self.kernel.get_scaling()

    def get_args(self):
        return self.kernel.get_args()

    def get_source_args(self):
        return self.kernel.get_source_args()

    # }}}

    def __len__(self):
        return len(self.get_coefficient_identifiers())

    def coefficients_from_source(self, avec, bvec):
        """Form an expansion from a source point.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        raise NotImplementedError

    def evaluate(self, coeffs, bvec):
        """
        :return: a :mod:`sympy` expression corresponding
            to the evaluated expansion with the coefficients
            in *coeffs*.
        """

        raise NotImplementedError

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.kernel)
        key_builder.rec(key_hash, self.order)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.kernel == other.kernel
                and self.order == other.order)

    def __ne__(self, other):
        return not self.__eq__(other)

# }}}


# {{{ derivative wrangler

class DerivativeWrangler(object):

    def __init__(self, order, dim):
        self.order = order
        self.dim = dim

    def get_coefficient_identifiers(self):
        raise NotImplementedError

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives):
        raise NotImplementedError

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients):
        raise NotImplementedError

    def get_derivative_taker(self, expr, var_list):
        raise NotImplementedError

    @memoize_method
    def get_full_coefficient_identifiers(self):
        """
        Returns identifiers for every coefficient in the complete expansion.
        """
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        res = sorted(gnitstam(self.order, self.dim), key=sum)
        return res


class FullDerivativeWrangler(DerivativeWrangler):

    def get_derivative_taker(self, expr, dvec):
        return MiDerivativeTaker(expr, dvec)

    get_coefficient_identifiers = (
            DerivativeWrangler.get_full_coefficient_identifiers)

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives):
        return stored_kernel_derivatives

    get_stored_mpole_coefficients_from_full = (
            get_full_kernel_derivatives_from_stored)


# {{{ sparse matrix-vector multiplication

def _spmv(spmat, x, sparse_vectors):
    """
    :param spmat: maps row indices to list of (col idx, value)
    :param x: maps vector indices to values
    :param sparse_vectors: If True, treat vectors as dict-like, otherwise list-like
    """
    if sparse_vectors:
        result = {}
    else:
        result = []

    for row in range(len(spmat)):
        acc = 0

        for j, coeff in spmat[row]:
            if sparse_vectors:
                # Check if the index exists in the vector.
                if x.get(j, 0) == 0:
                    continue

            acc += coeff * x[j]

        if sparse_vectors:
            if acc != 0:
                result[row] = acc
        else:
            result.append(acc)

    return result

# }}}


class LinearRecurrenceBasedDerivativeWrangler(DerivativeWrangler):

    def __init__(self, order, dim):
        DerivativeWrangler.__init__(self, order, dim)

    def get_coefficient_identifiers(self):
        return self.stored_identifiers

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives):
        return _spmv(self.coeff_matrix, stored_kernel_derivatives,
                     sparse_vectors=False)

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients):
        # = M^T x, where M = coeff matrix
        result = [0] * len(self.stored_identifiers)
        for row, coeff in enumerate(full_mpole_coefficients):
            for col, val in self.coeff_matrix[row]:
                result[col] += coeff * val
        return result

    def precompute_recurrences(self):
        """
        Build up a matrix that expresses every derivative in terms of a
        set of "stored" derivatives.

        For example, for the recurrence

            u_xx + u_yy + u_zz = 0

        the coefficient matrix features the following entries:

              ... u_xx u_yy ... <= cols = only stored derivatives
             ==================
         ...| ...  ...  ... ...
            |
        u_zz| ...  -1   -1  ...

           ^ rows = one for every derivative
        """
        stored_identifiers = []
        identifiers_so_far = {}

        import time
        start_time = time.time()
        logger.debug("computing recurrence for Taylor coefficients: start")

        # Sparse matrix, indexed by row
        from collections import defaultdict
        coeff_matrix_transpose = defaultdict(lambda: [])

        # Build up the matrix transpose by row.
        from six import iteritems
        for i, identifier in enumerate(self.get_full_coefficient_identifiers()):
            expr = self.try_get_recurrence_for_derivative(
                    identifier, identifiers_so_far)

            if expr is None:
                # Identifier should be stored
                coeff_matrix_transpose[len(stored_identifiers)] = [(i, 1)]
                stored_identifiers.append(identifier)
            else:
                expr = dict((identifiers_so_far[ident], val) for ident, val in
                            iteritems(expr))
                result = _spmv(coeff_matrix_transpose, expr, sparse_vectors=True)
                for j, item in iteritems(result):
                    coeff_matrix_transpose[j].append((i, item))

            identifiers_so_far[identifier] = i

        self.stored_identifiers = stored_identifiers

        coeff_matrix = defaultdict(lambda: [])
        for i, row in iteritems(coeff_matrix_transpose):
            for j, val in row:
                coeff_matrix[j].append((i, val))

        logger.debug("computing recurrence for Taylor coefficients: "
                     "done after {dur:.2f} seconds"
                     .format(dur=time.time() - start_time))

        logger.debug("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(self.stored_identifiers)))

        self.coeff_matrix = coeff_matrix

    def try_get_recurrence_for_derivative(self, deriv, in_terms_of):
        raise NotImplementedError

    def get_derivative_taker(self, expr, var_list):
        from sumpy.tools import LinearRecurrenceBasedMiDerivativeTaker
        return LinearRecurrenceBasedMiDerivativeTaker(expr, var_list, self)


class LaplaceDerivativeWrangler(LinearRecurrenceBasedDerivativeWrangler):

    def __init__(self, order, dim):
        LinearRecurrenceBasedDerivativeWrangler.__init__(self, order, dim)
        self.precompute_recurrences()

    def try_get_recurrence_for_derivative(self, deriv, in_terms_of):
        deriv = np.array(deriv, dtype=int)

        for dim in np.nonzero(2 <= deriv)[0]:
            # Check if we can reduce this dimension in terms of the other
            # dimensions.

            reduced_deriv = deriv.copy()
            reduced_deriv[dim] -= 2
            coeffs = {}

            for other_dim in range(self.dim):
                if other_dim == dim:
                    continue
                needed_deriv = reduced_deriv.copy()
                needed_deriv[other_dim] += 2
                needed_deriv = tuple(needed_deriv)

                if needed_deriv not in in_terms_of:
                    break

                coeffs[needed_deriv] = -1
            else:
                return coeffs


class HelmholtzDerivativeWrangler(LinearRecurrenceBasedDerivativeWrangler):

    def __init__(self, order, dim, helmholtz_k_name):
        LinearRecurrenceBasedDerivativeWrangler.__init__(self, order, dim)
        self.helmholtz_k_name = helmholtz_k_name
        self.precompute_recurrences()

    def try_get_recurrence_for_derivative(self, deriv, in_terms_of):
        deriv = np.array(deriv, dtype=int)

        for dim in np.nonzero(2 <= deriv)[0]:
            # Check if we can reduce this dimension in terms of the other
            # dimensions.

            reduced_deriv = deriv.copy()
            reduced_deriv[dim] -= 2
            coeffs = {}

            for other_dim in range(self.dim):
                if other_dim == dim:
                    continue
                needed_deriv = reduced_deriv.copy()
                needed_deriv[other_dim] += 2
                needed_deriv = tuple(needed_deriv)

                if needed_deriv not in in_terms_of:
                    break

                coeffs[needed_deriv] = -1
            else:
                k = sym.Symbol(self.helmholtz_k_name)
                coeffs[tuple(reduced_deriv)] = -k*k
                return coeffs

# }}}


# {{{ volume taylor

class VolumeTaylorExpansionBase(object):

    @classmethod
    def get_or_make_derivative_wrangler(cls, *key):
        """
        This stores the derivative wrangler at the class attribute level because
        precomputing the derivative wrangler may become expensive.
        """
        try:
            wrangler = cls.derivative_wrangler_cache[key]
        except KeyError:
            wrangler = cls.derivative_wrangler_class(*key)
            cls.derivative_wrangler_cache[key] = wrangler

        return wrangler

    @property
    def derivative_wrangler(self):
        return self.get_or_make_derivative_wrangler(*self.derivative_wrangler_key)

    def get_coefficient_identifiers(self):
        """
        Returns the identifiers of the coefficients that actually get stored.
        """
        return self.derivative_wrangler.get_coefficient_identifiers()

    def get_full_coefficient_identifiers(self):
        return self.derivative_wrangler.get_full_coefficient_identifiers()

    @property
    @memoize_method
    def _storage_loc_dict(self):
        return dict((i, idx) for idx, i in
                    enumerate(self.get_coefficient_identifiers()))

    def get_storage_index(self, i):
        return self._storage_loc_dict[i]


class VolumeTaylorExpansion(VolumeTaylorExpansionBase):

    derivative_wrangler_class = FullDerivativeWrangler
    derivative_wrangler_cache = {}

    def __init__(self, kernel, order):
        self.derivative_wrangler_key = (order, kernel.dim)


class LaplaceConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    derivative_wrangler_class = LaplaceDerivativeWrangler
    derivative_wrangler_cache = {}

    def __init__(self, kernel, order):
        self.derivative_wrangler_key = (order, kernel.dim)


class HelmholtzConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    derivative_wrangler_class = HelmholtzDerivativeWrangler
    derivative_wrangler_cache = {}

    def __init__(self, kernel, order):
        helmholtz_k_name = kernel.get_base_kernel().helmholtz_k_name
        self.derivative_wrangler_key = (order, kernel.dim, helmholtz_k_name)

# }}}


# vim: fdm=marker
