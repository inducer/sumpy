from __future__ import division
from __future__ import absolute_import

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
import sympy as sp
from pytools import memoize_method

__doc__ = """
.. autoclass:: ExpansionBase
"""


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


# {{{ volume taylor

class VolumeTaylorExpansionBase(object):

    def get_coefficient_identifiers(self):
        """
        Returns the identifiers of the coefficients that actually get stored.
        """
        raise NotImplementedError

    @property
    @memoize_method
    def _storage_loc_dict(self):
        return dict((i, idx) for idx, i in
                    enumerate(self.get_coefficient_identifiers()))

    def get_storage_index(self, i):
        return self._storage_loc_dict[i]

    @memoize_method
    def get_full_coefficient_identifiers(self):
        """
        Returns identifiers for every coefficient in the complete expansion.
        """
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        return sorted(gnitstam(self.order, self.kernel.dim), key=sum)

    def stored_to_full(self, coeff_idx, stored_coeffs):
        raise NotImplementedError

    def full_to_stored(self, coeff_idx, full_coeffs):
        raise NotImplementedError


class VolumeTaylorExpansion(VolumeTaylorExpansionBase):

    get_coefficient_identifiers = (
        VolumeTaylorExpansionBase.get_full_coefficient_identifiers)

    def stored_to_full(self, stored_coeffs):
        return stored_coeffs

    full_to_stored = stored_to_full


class DerivativeWrangler(object):

    def __init__(self, kernel, order):
        self.kernel = kernel
        self.order = order

    def get_full_coefficient_identifiers(self):
        raise NotImplementedError

    def get_coefficient_identifiers(self):
        raise NotImplementedError

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives):
        raise NotImplementedError

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients):
        raise NotImplementedError

    def get_kernel_derivative_taker(self, dvec):
        raise NotImplementedError


class LinearRecurrenceBasedDerivativeWrangler(DerivativeWrangler):

    def __init__(self, kernel, order):
        DerivativeWrangler.__init__(self, kernel, order)
        self.precompute_coeff_matrix()

    def get_coefficient_identifiers(self):
        return self.stored_identifiers

    @memoize_method
    def get_full_coefficient_identifiers(self):
        """
        Returns identifiers for every coefficient in the complete expansion.
        """
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        return sorted(gnitstam(self.order, self.kernel.dim), key=sum)

    def stored_to_full(self, stored_coeffs):
        return self.coeff_matrix.dot(stored_coeffs)

    def full_to_stored(self, full_coeffs):
        return self.coeff_matrix.T.dot(full_coeffs)

    def precompute_coeff_matrix(self):
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

        ncoeffs = len(self.get_full_coefficient_identifiers())
        from sympy.matrices.sparse import SparseMatrix
        #coeff_matrix_transpose = SparseMatrix(ncoeffs, ncoeffs, {}).as_mutable()

        # Sparse matrix, indexed by row
        from collections import defaultdict
        coeff_matrix_transpose = defaultdict(lambda row: [])

        # Build up the matrix by row.
        for i, identifier in enumerate(self.get_full_coefficient_identifiers()):
            expr = self.try_get_recurrence_for_derivative(
                    identifier, identifiers_so_far)

            if expr is None:
                # Identifier should be stored
                coeff_matrix_transpose[len(stored_identifiers)] = [(i, 1)]
                stored_identifiers.append(identifier)
            else:
                nstored = len(stored_identifiers)
                ntotal = len(identifiers_so_far)

                result = {}
                for row in range(nstored):
                    acc = 0

                    """
                    # Use the smaller of the two.
                    if len(coeff_matrix_transpose[row]) < len(expr):
                        smaller = coeff_matrix_transpose[row]
                        indices, vals = zip(*sorted(expr.items()))
                    else:
                        smaller = expr.items()
                        indices, vals = zip(*coeff_matrix_transpose[row])
                    indices = list(indices)
                    vals = list(vals)

                    import bisect
                    for idx, coeff in smaller:
                        if coeff == 0:
                            continue
                        pos = bisect.bisect_left(indices, idx)
                        if pos == len(indices) or indices[pos] != idx or vals[pos] == 0:
                            continue
                        acc += coeff * vals[pos]
                    """

                    for j, val in coeff_matrix_transpose[row]:
                        if expr.get(j, 0) == 0:
                            continue
                        acc += expr[j] * val

                    if acc != 0:
                        result[row] = acc

                for j, item in result.items():
                    coeff_matrix_transpose[j].append((i, item))

            identifiers_so_far[identifier] = i

        self.stored_identifiers = stored_identifiers
        #self.coeff_matrix = coeff_matrix_transpose.T[:,:len(stored_identifiers)]
        self.coeff_matrix = coeff_matrix_transpose
        print(self.coeff_matrix)
        print(ncoeffs, len(self.stored_identifiers))

    def try_get_recurrence_for_derivative(self, deriv, in_terms_of):
        raise NotImplementedError


    def get_derivative_taker(self, expr, dvec):
        return LinearRecurrenceBasedDerivativeWrangler(
                expr, dvec, self)


from sumpy.tools import MiDerviativeTaker

class LinearRecurrenceBasedMiDerivativeTaker(MiDerviativeTaker):

    def __init__(self, expr, dvec, wrangler):
        MiDerviativeTaker.__init__(expr, dvec)
        self.wrangler = wrangler

    def diff(self, mi):
        try:
            return self.cache_by_mi[mi]
        except KeyError:
            closest_mi = self.get_closest_mi(mi)
            expr = self.cache_by_mi[closest_mi]

            for needed_mi in self.get_needed_derivatives(closest_mi, dest_mi):
                recurrence = self.wranger.try_get_recurrence_for_derivative(
                        needed_mi, self.cache_by_mi)
                if recurrence is not None:
                    pass
                else:
                    
                # For each derivative that we need
                # 1. Use try_get_recurrence_for_derivative
                # 2. Otherwise, do it the old fashioned way.


class LaplaceDerivativeWrangler(LinearRecurrenceBasedDerivativeWrangler):

    def try_get_recurrence_for_derivative(self, deriv, in_terms_of):
        deriv = np.array(deriv, dtype=int)

        for dim in np.nonzero(2 <= deriv)[0]:
            # Check if we can reduce this dimension in terms of the other
            # dimensions.

            reduced_deriv = deriv.copy()
            reduced_deriv[dim] -= 2
            needed_derivs = []

            for other_dim in range(self.kernel.dim):
                if other_dim == dim:
                    continue
                needed_deriv = reduced_deriv.copy()
                needed_deriv[other_dim] += 2
                needed_deriv = tuple(needed_deriv)

                if needed_deriv not in in_terms_of:
                    break

                needed_derivs.append(needed_deriv)
            else:
                expr = {}
                for needed_deriv in needed_derivs:
                    expr[in_terms_of[needed_deriv]] = -1

                return expr


class HelmholtzDerivativeWrangler(LinearRecurrenceBasedDerivativeWrangler):

    def __init__(self, kernel, order):
        k = sp.Symbol(kernel.get_base_kernel().helmholtz_k_name)
        self.k = -k*k
        LinearRecurrenceBasedDerivativeWrangler.__init__(self, kernel, order)
        print("HI HI")
        print("self.k", k)


    def try_get_recurrence_for_derivative(self, deriv, in_terms_of):
        deriv = np.array(deriv, dtype=int)

        for dim in np.nonzero(2 <= deriv)[0]:
            # Check if we can reduce this dimension in terms of the other
            # dimensions.

            reduced_deriv = deriv.copy()
            reduced_deriv[dim] -= 2
            needed_derivs = []

            for other_dim in range(self.kernel.dim):
                if other_dim == dim:
                    continue
                needed_deriv = reduced_deriv.copy()
                needed_deriv[other_dim] += 2
                needed_deriv = tuple(needed_deriv)

                if needed_deriv not in in_terms_of:
                    break

                needed_derivs.append((-1, needed_deriv))
            else:
                needed_derivs.append((self.k, tuple(reduced_deriv)))

                expr = {}
                for coeff, needed_deriv in needed_derivs:
                    expr[in_terms_of[needed_deriv]] = coeff

                return expr

"""
class HelmholtzConformingVolumeTaylorExpansion(
        LinearRecurrenceBasedVolumeTaylorExpansion):

    def try_get_recurrence_for_derivative(self, deriv, in_terms_of, ncoeffs):
        deriv = np.array(deriv)

        for dim in np.nonzero(2 <= deriv)[0]:
            # Check if we can reduce this dimension in terms of the other
            # dimensions.

            reduced_deriv = deriv.copy()
            reduced_deriv[dim] -= 2

            needed_derivs = []
            for other_dim in range(self.kernel.dim):
                if other_dim == dim:
                    continue
                needed_deriv = reduced_deriv.copy()
                needed_deriv[other_dim] += 2

                needed_derivs.append((-1, tuple(needed_deriv)))

            k = sp.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)

            needed_derivs.append((-k*k, tuple(reduced_deriv)))

            expr = np.zeros(ncoeffs, dtype=object)
            try:
                for coeff, needed_deriv in needed_derivs:
                    deriv_idx = in_terms_of.index(needed_deriv)
                    expr[deriv_idx] = coeff
            except ValueError:
                continue

            return expr
"""

# }}}


# vim: fdm=marker
