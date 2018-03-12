from __future__ import division, absolute_import

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

from six.moves import range
import six
import numpy as np
import logging
from pytools import memoize_method
import sumpy.symbolic as sym
from sumpy.tools import MiDerivativeTaker
from collections import defaultdict


__doc__ = """
.. autoclass:: ExpansionBase

Expansion Factories
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ExpansionFactoryBase
.. autoclass:: DefaultExpansionFactory
.. autoclass:: VolumeTaylorExpansionFactory
"""

logger = logging.getLogger(__name__)


# {{{ base class

class ExpansionBase(object):
    """
    .. automethod:: with_kernel
    .. automethod:: __len__
    .. automethod:: get_coefficient_identifiers
    .. automethod:: coefficients_from_source
    .. automethod:: translate_from
    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __init__(self, kernel, order, use_rscale=None):
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
    def dim(self):
        return self.kernel.dim

    @property
    def is_complex_valued(self):
        return self.kernel.is_complex_valued

    def prepare_loopy_kernel(self, loopy_knl):
        return self.kernel.prepare_loopy_kernel(loopy_knl)

    def get_code_transformer(self):
        return self.kernel.get_code_transformer()

    def get_global_scaling_const(self):
        return self.kernel.get_global_scaling_const()

    def get_args(self):
        return self.kernel.get_args()

    def get_source_args(self):
        return self.kernel.get_source_args()

    # }}}

    def with_kernel(self, kernel):
        return type(self)(kernel, self.order, self.use_rscale)

    def __len__(self):
        return len(self.get_coefficient_identifiers())

    def get_coefficient_identifiers(self):
        """
        Returns the identifiers of the coefficients that actually get stored.
        """
        raise NotImplementedError

    def coefficients_from_source(self, avec, bvec, rscale):
        """Form an expansion from a source point.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        raise NotImplementedError

    def evaluate(self, coeffs, bvec, rscale):
        """
        :return: a :mod:`sympy` expression corresponding
            to the evaluated expansion with the coefficients
            in *coeffs*.
        """

        raise NotImplementedError

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        raise NotImplementedError

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.kernel)
        key_builder.rec(key_hash, self.order)
        key_builder.rec(key_hash, self.use_rscale)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.kernel == other.kernel
                and self.order == other.order
                and self.use_rscale == other.use_rscale)

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

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale):
        raise NotImplementedError

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale):
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

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale):
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

    def __init__(self, order, dim, deriv_multiplier):
        super(LinearRecurrenceBasedDerivativeWrangler, self).__init__(order, dim)
        self.deriv_multiplier = deriv_multiplier

    def get_coefficient_identifiers(self):
        return self.stored_identifiers

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale):
        coeff_matrix = self.get_coefficient_matrix(rscale)
        return _spmv(coeff_matrix, stored_kernel_derivatives,
                     sparse_vectors=False)

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale):
        # = M^T x, where M = coeff matrix

        coeff_matrix = self.get_coefficient_matrix(rscale)
        result = [0] * len(self.stored_identifiers)
        for row, coeff in enumerate(full_mpole_coefficients):
            for col, val in coeff_matrix[row]:
                result[col] += coeff * val
        return result

    @property
    def stored_identifiers(self):
        stored_identifiers, coeff_matrix = self._get_stored_ids_and_coeff_mat()
        return stored_identifiers

    @memoize_method
    def get_coefficient_matrix(self, rscale):
        """
        Return a matrix that expresses every derivative in terms of a
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
        stored_identifiers, coeff_matrix = self._get_stored_ids_and_coeff_mat()

        full_coeffs = self.get_full_coefficient_identifiers()
        matrix_rows = []
        for irow, row in six.iteritems(coeff_matrix):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[irow])
            matrix_row = []
            for icol, coeff in row:
                diff = row_rscale - sum(stored_identifiers[icol])
                mult = (rscale*self.deriv_multiplier)**diff
                matrix_row.append((icol, coeff * mult))
            matrix_rows.append((irow, matrix_row))

        return defaultdict(list, matrix_rows)

    @memoize_method
    def _get_stored_ids_and_coeff_mat(self):
        from six import iteritems
        from sumpy.tools import nullspace

        tol = 1e-13
        stored_identifiers = []

        import time
        start_time = time.time()
        logger.debug("computing recurrence for Taylor coefficients: start")

        pde_dict = self.get_pde_dict()
        pde_mat = []

        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = dict((tuple(mi), i) for
                                            (i, mi) in enumerate(mis))

        for mi in mis:
            for pde_mi, coeff in iteritems(pde_dict):
                diff = np.array(mi, dtype=int) - pde_mi
                if (diff >= 0).all():
                    eq = [0]*len(mis)
                    for pde_mi2, coeff2 in iteritems(pde_dict):
                        c = tuple(pde_mi2 + diff)
                        if c not in coeff_ident_enumerate_dict:
                            break
                        eq[coeff_ident_enumerate_dict[c]] = 1
                    else:
                        pde_mat.append(eq)

        if len(pde_mat) > 0:
            pde_mat = np.array(pde_mat, dtype=np.float64)
            n = nullspace(pde_mat, atol=tol)
            idx = self.get_reduced_coeffs()
            assert len(idx) >= n.shape[1]
            s = np.linalg.solve(n.T[:, idx], n.T)
            stored_identifiers = [mis[i] for i in idx]
        else:
            s = np.eye(len(mis))
            stored_identifiers = mis

        coeff_matrix = defaultdict(lambda: [])
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if np.abs(s[i][j]) > tol:
                    coeff_matrix[j].append((i, s[i][j]))

        logger.debug("computing recurrence for Taylor coefficients: "
                     "done after {dur:.2f} seconds"
                     .format(dur=time.time() - start_time))

        logger.debug("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(stored_identifiers)))

        return stored_identifiers, coeff_matrix

    def get_pde_dict(self):
        """
        Return the PDE as a dictionary of derivative_identifier: coeff such that
        sum(derivative_identifer * coeff) = 0 is the PDE.
        """

        raise NotImplementedError

    def get_reduced_coeffs(self):
        """
        If the coefficients of the derivatives can be reduced and the reduced
        coefficients are known, returns a list of indices. Returning None
        indicates the reduced coefficients are unknown and will be calculated
        using an Interpolative Decomposition of the PDE matrix
        """
        raise NotImplementedError

    def get_derivative_taker(self, expr, var_list):
        from sumpy.tools import NewLinearRecurrenceBasedMiDerivativeTaker
        return NewLinearRecurrenceBasedMiDerivativeTaker(expr, var_list, self)


class LaplaceDerivativeWrangler(LinearRecurrenceBasedDerivativeWrangler):

    def __init__(self, order, dim):
        super(LaplaceDerivativeWrangler, self).__init__(order, dim, 1)

    def get_pde_dict(self):
        pde_dict = {}
        for d in range(self.dim):
            mi = [0]*self.dim
            mi[d] = 2
            pde_dict[tuple(mi)] = 1
        return pde_dict

    def get_reduced_coeffs(self):
        idx = []
        for i, mi in enumerate(self.get_full_coefficient_identifiers()):
            if mi[-1] < 2:
                idx.append(i)
        return idx


class HelmholtzDerivativeWrangler(LinearRecurrenceBasedDerivativeWrangler):

    def __init__(self, order, dim, helmholtz_k_name):
        multiplier = sym.Symbol(helmholtz_k_name)
        super(HelmholtzDerivativeWrangler, self).__init__(order, dim, multiplier)

    def get_pde_dict(self, **kwargs):
        pde_dict = {}
        for d in range(self.dim):
            mi = [0]*self.dim
            mi[d] = 2
            pde_dict[tuple(mi)] = 1
        pde_dict[tuple([0]*self.dim)] = -1
        return pde_dict

    def get_reduced_coeffs(self):
        idx = []
        for i, mi in enumerate(self.get_full_coefficient_identifiers()):
            if mi[-1] < 2:
                idx.append(i)
        return idx

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

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.derivative_wrangler_key = (order, kernel.dim)


class LaplaceConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    derivative_wrangler_class = LaplaceDerivativeWrangler
    derivative_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.derivative_wrangler_key = (order, kernel.dim)


class HelmholtzConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    derivative_wrangler_class = HelmholtzDerivativeWrangler
    derivative_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        helmholtz_k_name = kernel.get_base_kernel().helmholtz_k_name
        self.derivative_wrangler_key = (order, kernel.dim, helmholtz_k_name)

# }}}


# {{{ expansion factory

class ExpansionFactoryBase(object):
    """An interface
    .. automethod:: get_local_expansion_class
    .. automethod:: get_multipole_expansion_class
    """

    def get_local_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        raise NotImplementedError()

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        raise NotImplementedError()


class VolumeTaylorExpansionFactory(ExpansionFactoryBase):
    """An implementation of :class:`ExpansionFactoryBase` that uses Volume Taylor
    expansions for each kernel.
    """

    def get_local_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.local import VolumeTaylorLocalExpansion
        return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
        return VolumeTaylorMultipoleExpansion


class DefaultExpansionFactory(ExpansionFactoryBase):
    """An implementation of :class:`ExpansionFactoryBase` that gives the 'best known'
    expansion for each kernel.
    """

    def get_local_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.kernel import HelmholtzKernel, LaplaceKernel, YukawaKernel
        if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
                and base_kernel.dim == 2):
            from sumpy.expansion.local import H2DLocalExpansion
            return H2DLocalExpansion
        elif (isinstance(base_kernel.get_base_kernel(), YukawaKernel)
                and base_kernel.dim == 2):
            from sumpy.expansion.local import Y2DLocalExpansion
            return Y2DLocalExpansion
        elif isinstance(base_kernel.get_base_kernel(), HelmholtzKernel):
            from sumpy.expansion.local import \
                    HelmholtzConformingVolumeTaylorLocalExpansion
            return HelmholtzConformingVolumeTaylorLocalExpansion
        elif isinstance(base_kernel.get_base_kernel(), LaplaceKernel):
            from sumpy.expansion.local import \
                    LaplaceConformingVolumeTaylorLocalExpansion
            return LaplaceConformingVolumeTaylorLocalExpansion
        else:
            from sumpy.expansion.local import VolumeTaylorLocalExpansion
            return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.kernel import HelmholtzKernel, LaplaceKernel, YukawaKernel
        if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
                and base_kernel.dim == 2):
            from sumpy.expansion.multipole import H2DMultipoleExpansion
            return H2DMultipoleExpansion
        elif (isinstance(base_kernel.get_base_kernel(), YukawaKernel)
                and base_kernel.dim == 2):
            from sumpy.expansion.multipole import Y2DMultipoleExpansion
            return Y2DMultipoleExpansion
        elif isinstance(base_kernel.get_base_kernel(), LaplaceKernel):
            from sumpy.expansion.multipole import (
                    LaplaceConformingVolumeTaylorMultipoleExpansion)
            return LaplaceConformingVolumeTaylorMultipoleExpansion
        elif isinstance(base_kernel.get_base_kernel(), HelmholtzKernel):
            from sumpy.expansion.multipole import (
                    HelmholtzConformingVolumeTaylorMultipoleExpansion)
            return HelmholtzConformingVolumeTaylorMultipoleExpansion
        else:
            from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
            return VolumeTaylorMultipoleExpansion

# }}}


# vim: fdm=marker
