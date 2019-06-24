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
from collections import defaultdict
from sumpy.tools import add_mi, find_linear_independent_row, CoeffIdentifier
from .pde_utils import (make_pde_syms, process_pde, laplacian, div, grad,
    PDE)

__doc__ = """
.. autoclass:: ExpansionBase
.. autoclass:: LinearPDEBasedExpansionTermsWrangler

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


# {{{ expansion terms wrangler

class ExpansionTermsWrangler(object):

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        self.order = order
        self.dim = dim
        self.max_mi = max_mi

    def get_coefficient_identifiers(self):
        raise NotImplementedError

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale):
        raise NotImplementedError

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale):
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

        def filter_tuple(tup):
            if self.max_mi is None:
                return True
            for a, b in zip(tup, self.max_mi):
                if a > b:
                    return False
            return True

        res = list(filter(filter_tuple, res))
        return res

    def copy(self, **kwargs):
        new_kwargs = dict(
                (name, getattr(self, name))
                for name in self.init_arg_names)

        for name in self.init_arg_names:
            new_kwargs[name] = kwargs.pop(name, getattr(self, name))

        if kwargs:
            raise TypeError("unexpected keyword arguments '%s'"
                % ", ".join(kwargs))

        return type(self)(**new_kwargs)


class FullExpansionTermsWrangler(ExpansionTermsWrangler):

    get_coefficient_identifiers = (
            ExpansionTermsWrangler.get_full_coefficient_identifiers)

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


def _reconstruct(reconstruct_matrix, stored):
    res = [0] * len(reconstruct_matrix)
    stored_idx = 0
    for row, deps in reconstruct_matrix:
        if len(deps) == 0:
            res[row] = stored[stored_idx]
            stored_idx += 1
            continue
        for k, v in deps.items():
            res[row] += res[k] * v
    return res


class LinearPDEBasedExpansionTermsWrangler(ExpansionTermsWrangler):
    """
    .. automethod:: __init__
    .. automethod:: get_pdes
    .. automethod:: get_reduced_coeffs
    """

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        r"""
        :param order: order of the expansion
        :param dim: number of dimensions
        """
        super(LinearPDEBasedExpansionTermsWrangler, self).__init__(order, dim,
                max_mi)

    def get_coefficient_identifiers(self):
        return self.stored_identifiers

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale):
        coeff_matrix, reconstruct_matrix, use_reconstruct = \
            self.get_coefficient_matrix(rscale)
        if not use_reconstruct:
            return _spmv(coeff_matrix, stored_kernel_derivatives,
                     sparse_vectors=False)
        return _reconstruct(reconstruct_matrix, stored_kernel_derivatives)

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale):
        # = M^T x, where M = coeff matrix

        coeff_matrix, _, _ = self.get_coefficient_matrix(rscale)
        result = [0] * len(self.stored_identifiers)
        for row, coeff in enumerate(full_mpole_coefficients):
            for col, val in coeff_matrix[row]:
                result[col] += coeff * val
        return result

    @property
    def stored_identifiers(self):
        stored_identifiers, coeff_matrix, _ = self.get_stored_ids_and_coeff_mat()
        return stored_identifiers

    @memoize_method
    def get_coefficient_matrix(self, rscale):
        """
        Return a matrix that expresses every derivative in terms of a
        set of "stored" derivatives.

        For example, for the PDE::

            u_xx + u_yy + u_zz = 0

        the coefficient matrix features the following entries::

                ... u_xx u_yy ... <= cols = only stored derivatives
                ==================
             ...| ...  ...  ... ...
                |
            u_zz| ...  -1   -1  ...

            ^ rows = one for every derivative
        """
        stored_identifiers, coeff_matrix, reconstruct_matrix = \
            self.get_stored_ids_and_coeff_mat()

        full_coeffs = self.get_full_coefficient_identifiers()
        matrix_rows = []
        count_nonzero_coeff = 0
        for irow, row in six.iteritems(coeff_matrix):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[irow])
            matrix_row = []
            for icol, coeff in row:
                diff = row_rscale - sum(stored_identifiers[icol])
                mult = rscale**diff
                matrix_row.append((icol, coeff * mult))
            count_nonzero_coeff += len(row)
            matrix_rows.append((irow, matrix_row))

        if reconstruct_matrix is None:
            return defaultdict(list, matrix_rows), None, False

        reconstruct_matrix_with_rscale = []
        count_nonzero_reconstruct = 0
        for row, deps in reconstruct_matrix:
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[row])
            matrix_row = []
            deps_with_rscale = {}
            for k, coeff in deps.items():
                diff = row_rscale - sum(full_coeffs[k])
                mult = rscale**diff
                deps_with_rscale[k] = coeff * mult
            count_nonzero_reconstruct += len(deps)
            reconstruct_matrix_with_rscale.append((row, deps_with_rscale))

        use_reconstruct = count_nonzero_reconstruct < count_nonzero_coeff

        return defaultdict(list, matrix_rows), reconstruct_matrix_with_rscale, \
            use_reconstruct

    @memoize_method
    def get_stored_ids_and_coeff_mat(self):
        from six import iteritems
        from sumpy.tools import nullspace, solve_symbolic

        from pytools import ProcessLogger
        plog = ProcessLogger(logger, "compute PDE for Taylor coefficients")

        pdes, iexpr, nexpr = self.get_pdes()
        pde_mat = []
        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = dict((tuple(mi), i) for
                                            (i, mi) in enumerate(mis))
        offset = len(mis)

        for mi in mis:
            for pde_dict in pdes.eqs:
                eq = [0]*(len(mis)*nexpr)
                for ident, coeff in iteritems(pde_dict):
                    c = tuple(add_mi(ident.mi, mi))
                    if c not in coeff_ident_enumerate_dict:
                        break
                    eq[offset*ident.iexpr + coeff_ident_enumerate_dict[c]] = coeff
                else:
                    pde_mat.append(eq)

        if len(pde_mat) > 0:
            r"""
            Find a matrix :math:`s` such that :math:`K = S^T K_{[r]}`
            where :math:`r` is the indices of the  reduced coefficients and
            :math:`K` is a column vector of coefficients. Let :math:`P` be the
            PDE matrix, i.e. the matrix obtained by applying the PDE
            as an identity to each of the Taylor coefficients.
            (Realize that those, as functions of space, each still satisfy the PDE.)
            As a result, if :math:`\mathbf\alpha` is a vector of Taylor coefficients,
            one would expect :math:`P\mathbf\alpha = \mathbf 0`.
            Further, let :math:`N` be the nullspace of :math:`P`.
            Then, :math:`S^T = N * N_{[r, :]}^{-1}` which implies,
            :math:`S = N_{[r, :]}^{-T} N^T = N_{[r, :]}^{-T} N^T`.
            """
            n = nullspace(pde_mat)

            # Move the rows corresponding to this expression to the front
            rearrange = list(range(offset*iexpr, offset*(iexpr+1)))
            for i in range(nexpr*offset):
                if i < offset*iexpr or i >= offset*(iexpr+1):
                    rearrange.append(i)
            n = n[rearrange, :]

            # Get the indices of the reduced coefficients
            idx_all_exprs = self.get_reduced_coeffs(n)

            s = solve_symbolic(n.T[:, idx_all_exprs], n.T)
            # Filter out coefficients belonging to this expression
            indices = list(filter(lambda x: x < offset, idx_all_exprs))
            s = s[:len(indices), :offset]

            stored_identifiers = [mis[i] for i in indices]
        else:
            s = np.eye(len(mis))
            stored_identifiers = mis
            idx_all_exprs = list(range(len(mis)))

        # coeff_matrix is a dictionary of lists. Each key in the dictionary
        # is a row number and the values are pairs of column number and value.
        coeff_matrix = defaultdict(list)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if s[i, j] != 0:
                    coeff_matrix[j].append((i, s[i, j]))

        plog.done()

        print("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(stored_identifiers)))

        return stored_identifiers, coeff_matrix, pde_mat

    def get_pdes(self):
        r"""
        Returns a list of PDEs, expression number, number of expressions.
        A PDE is a dictionary of (ident, coeff) such that ident is a
        namedtuple of (mi, iexpr) where mi is the multi-index of the
        derivative, iexpr is the expression number
        """

        raise NotImplementedError

    @memoize_method
    def get_single_pde(self):
        from six import iteritems
        from sumpy.tools import nullspace

        pdes, iexpr, nexpr = self.get_pdes()
        pdes, multiplier = process_pde(pdes)
        if nexpr == 1:
            return pdes

        from pytools import ProcessLogger
        plog = ProcessLogger(logger, "computing single PDE for multiple PDEs")

        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        for order in range(1, 100):
            mis = sorted(gnitstam(order, self.dim), key=sum)

            pde_mat = []
            coeff_ident_enumerate_dict = dict((tuple(mi), i) for
                                                (i, mi) in enumerate(mis))
            offset = len(mis)

            for mi in mis:
                for pde_dict in pdes.eqs:
                    eq = [0]*(len(mis)*nexpr)
                    for ident, coeff in iteritems(pde_dict):
                        c = tuple(add_mi(ident.mi, mi))
                        if c not in coeff_ident_enumerate_dict:
                            break
                        idx = offset*ident.iexpr + coeff_ident_enumerate_dict[c]
                        eq[idx] = coeff
                    else:
                        pde_mat.append(eq)

            if len(pde_mat) == 0:
                continue

            n = nullspace(pde_mat)[offset*iexpr:offset*(iexpr+1), :]
            indep_row = find_linear_independent_row(n)
            if len(indep_row) > 0:
                pde_dict = {}
                min_order = min(sum(mis[k]) for k in indep_row.keys())
                for k, v in indep_row.items():
                    mi = mis[k]
                    mult = multiplier**(sum(mi)-min_order)
                    pde_dict[CoeffIdentifier(mi, 0)] = v * mult
                plog.done()
                return PDE(self.dim, pde_dict)

        plog.done()
        assert False


class LaplaceExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        super(LaplaceExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pdes(self):
        w = make_pde_syms(self.dim, 1)
        return laplacian(w), 0, 1


class HelmholtzExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "helmholtz_k_name", "max_mi")

    def __init__(self, order, dim, helmholtz_k_name, max_mi=None):
        self.helmholtz_k_name = helmholtz_k_name
        super(HelmholtzExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pdes(self, **kwargs):
        w = make_pde_syms(self.dim, 1)
        k = sym.Symbol(self.helmholtz_k_name)
        return (laplacian(w) + k**2 * w), 0, 1


class StokesExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "icomp", "viscosity_mu_name", "max_mi")

    def __init__(self, order, dim, icomp, viscosity_mu_name, max_mi=None):
        self.viscosity_mu_name = viscosity_mu_name
        self.icomp = icomp
        super(StokesExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pdes(self, **kwargs):
        w = make_pde_syms(self.dim, self.dim+1)
        mu = sym.Symbol(self.viscosity_mu_name)
        u = w[:self.dim]
        p = w[-1]
        pdes = PDE(self.dim, mu * laplacian(u) - grad(p), div(u))
        return pdes, self.icomp, self.dim+1
# }}}


# {{{ volume taylor

class VolumeTaylorExpansionBase(object):

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
        return dict((i, idx) for idx, i in
                    enumerate(self.get_coefficient_identifiers()))

    def get_storage_index(self, i):
        return self._storage_loc_dict[i]


class VolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = FullExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.expansion_terms_wrangler_key = (order, kernel.dim)


class LaplaceConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = LaplaceExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.expansion_terms_wrangler_key = (order, kernel.dim)


class HelmholtzConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = HelmholtzExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        helmholtz_k_name = kernel.get_base_kernel().helmholtz_k_name
        self.expansion_terms_wrangler_key = (order, kernel.dim, helmholtz_k_name)


class StokesConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = StokesExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        icomp = kernel.get_base_kernel().icomp
        viscosity_mu_name = kernel.get_base_kernel().viscosity_mu_name
        self.expansion_terms_wrangler_key = (order, kernel.dim,
            icomp, viscosity_mu_name)

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
