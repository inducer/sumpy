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
import logging
from pytools import memoize_method
import sumpy.symbolic as sym
from collections import defaultdict
from sumpy.tools import add_mi
from .pde_utils import make_pde_sym, laplacian

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

    @memoize_method
    def _get_coeff_identifier_split(self):
        """
        This splits the coefficients into a O(p) number of disjoint sets
        so that for each set, all the identifiers have the form,
        (m_1, m_2, ..., m_{j-1}, c, m_{j+1}, ... , m_d)
        where c is a constant.

        If this is an instance of LinearPDEBasedExpansionTermsWrangler,
        then the number of sets will be O(1).

        Returns a List[Tuple[j, List[identifiers]]]
        """
        res = []
        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = dict((tuple(mi), i) for
            (i, mi) in enumerate(mis))

        max_mi = None
        if isinstance(self, LinearPDEBasedExpansionTermsWrangler):
            pde_dict = self.get_pde().eq
            for ident in pde_dict.keys():
                if ident not in coeff_ident_enumerate_dict:
                    break
            else:
                max_mi_idx = max(coeff_ident_enumerate_dict[ident] for
                                 ident in pde_dict.keys())
                max_mi = mis[max_mi_idx]

        if max_mi is None:
            max_mi = [max(mi[d] for mi in mis) for d in range(self.dim)]

        seen_mis = set()
        for d in range(self.dim):
            filtered_mis = [mi for mi in mis if mi[d] < max_mi[d]]
            for i in range(max_mi[d]):
                new_mis = [mi for mi in filtered_mis if mi[d] == i
                             and mi not in seen_mis]
                seen_mis.update(new_mis)
                res.append((d, new_mis))
        return res


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


def _fast_spmv(reconstruct_matrix, vec, transpose=False):
    if not transpose:
        res = [0] * len(reconstruct_matrix)
        stored_idx = 0
        for row, deps in enumerate(reconstruct_matrix):
            if len(deps) == 0:
                res[row] = vec[stored_idx]
                stored_idx += 1
            else:
                for k, v in deps:
                    res[row] += res[k] * v
            res[row] = sym.UnevaluatedExpr(res[row])
        return res
    else:
        res = []
        expr_all = list(vec)
        for row, deps in reversed(list(enumerate(reconstruct_matrix))):
            if len(deps) == 0:
                res.append(expr_all[row])
                continue
            new_sym = sym.UnevaluatedExpr(expr_all[row])
            for k, v in deps:
                expr_all[k] += new_sym * v
        res.reverse()
        return res


class LinearPDEBasedExpansionTermsWrangler(ExpansionTermsWrangler):
    """
    .. automethod:: __init__
    .. automethod:: get_pde
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
        return _fast_spmv(reconstruct_matrix, stored_kernel_derivatives)

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale):
        # = M^T x, where M = coeff matrix
        coeff_matrix, reconstruct_matrix, use_reconstruct = \
            self.get_coefficient_matrix(rscale)
        if not use_reconstruct:
            result = [0] * len(self.stored_identifiers)
            for row, coeff in enumerate(full_mpole_coefficients):
                for col, val in coeff_matrix[row]:
                    result[col] += coeff * val
            return result
        return _fast_spmv(reconstruct_matrix, full_mpole_coefficients,
                transpose=True)

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
        count_nonzero_coeff = -len(stored_identifiers)
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
        for row, deps in enumerate(reconstruct_matrix):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[row])
            matrix_row = []
            deps_with_rscale = []
            for k, coeff in deps:
                diff = row_rscale - sum(full_coeffs[k])
                mult = rscale**diff
                deps_with_rscale.append((k, coeff * mult))
            count_nonzero_reconstruct += len(deps)
            reconstruct_matrix_with_rscale.append(deps_with_rscale)

        use_reconstruct = count_nonzero_reconstruct < count_nonzero_coeff

        return defaultdict(list, matrix_rows), reconstruct_matrix_with_rscale, \
            use_reconstruct

    @memoize_method
    def get_stored_ids_and_coeff_mat(self):
        from six import iteritems

        from pytools import ProcessLogger
        plog = ProcessLogger(logger, "compute PDE for Taylor coefficients")

        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = dict((tuple(mi), i) for
                                            (i, mi) in enumerate(mis))

        pde_dict = self.get_pde().eq
        for ident in pde_dict.keys():
            if ident not in coeff_ident_enumerate_dict:
                coeff_matrix = defaultdict(list)
                reconstruct_matrix = []
                for i in range(len(mis)):
                    coeff_matrix[i] = [(i, 1)]
                    reconstruct_matrix.append([])
                return mis, coeff_matrix, reconstruct_matrix

        max_mi_idx = max(coeff_ident_enumerate_dict[ident] for
                         ident in pde_dict.keys())
        max_mi = mis[max_mi_idx]
        max_mi_coeff = pde_dict[max_mi]
        max_mi_mult = -1/sym.sympify(max_mi_coeff)

        def is_stored(mi):
            """
            A multi_index mi is not stored if mi >= max_mi
            """
            return any(mi[d] < max_mi[d] for d in range(self.dim))

        stored_identifiers = [mi for mi in mis if is_stored(mi)]
        stored_ident_enumerate_dict = dict((tuple(mi), i) for
                                            (i, mi) in enumerate(stored_identifiers))

        coeff_matrix_dict = defaultdict(lambda: defaultdict(lambda: 0))
        reconstruct_matrix = []
        for i, mi in enumerate(mis):
            reconstruct_matrix.append([])
            if is_stored(mi):
                coeff_matrix_dict[i][stored_ident_enumerate_dict[mi]] = 1
                continue
            diff = [mi[d] - max_mi[d] for d in range(self.dim)]
            for other_mi, coeff in iteritems(pde_dict):
                j = coeff_ident_enumerate_dict[add_mi(other_mi, diff)]
                if i == j:
                    continue
                # PDE might not have max_mi_coeff = -1, divide by -max_mi_coeff
                # to get a relation of the form, u_zz = - u_xx - u_yy for Laplace 3D.
                reconstruct_matrix[i].append((j, coeff*max_mi_mult))
                # j can be a stored or a non-stored multi-index
                # Look at the coeff_matrix of j to get the j as a linear combination
                # of stored coefficients.
                for dep, other_coeff in iteritems(coeff_matrix_dict[j]):
                    coeff_matrix_dict[i][dep] += other_coeff*coeff*max_mi_mult

        # coeff_matrix is a dictionary of lists. Each key in the dictionary
        # is a row number and the values are pairs of column number and value.
        coeff_matrix = defaultdict(list)
        for row, deps in iteritems(coeff_matrix_dict):
            for col, val in iteritems(deps):
                if val != 0:
                    coeff_matrix[row].append((col, val))

        plog.done()

        print("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(stored_identifiers)))

        return stored_identifiers, coeff_matrix, reconstruct_matrix

    def get_pde(self):
        r"""
        Returns a PDE. A PDE stores a dictionary of (mi, coeff)
        where mi is the multi-index of the  derivative and coeff is the
        coefficient
        """

        raise NotImplementedError


class LaplaceExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        super(LaplaceExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pde(self):
        w = make_pde_sym(self.dim)
        return laplacian(w)


class HelmholtzExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "helmholtz_k_name", "max_mi")

    def __init__(self, order, dim, helmholtz_k_name, max_mi=None):
        self.helmholtz_k_name = helmholtz_k_name
        super(HelmholtzExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pde(self, **kwargs):
        w = make_pde_sym(self.dim)
        k = sym.Symbol(self.helmholtz_k_name)
        return (laplacian(w) + k**2 * w)


class BiharmonicExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        super(BiharmonicExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pde(self, **kwargs):
        w = make_pde_sym(self.dim)
        return laplacian(laplacian(w))
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


class BiharmonicConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = BiharmonicExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.expansion_terms_wrangler_key = (order, kernel.dim)

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
        from sumpy.kernel import (HelmholtzKernel, LaplaceKernel, YukawaKernel,
                BiharmonicKernel, StokesletKernel, StressletKernel)

        from sumpy.expansion.local import (H2DLocalExpansion, Y2DLocalExpansion,
                HelmholtzConformingVolumeTaylorLocalExpansion,
                LaplaceConformingVolumeTaylorLocalExpansion,
                BiharmonicConformingVolumeTaylorLocalExpansion,
                VolumeTaylorLocalExpansion)

        if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
                and base_kernel.dim == 2):
            return H2DLocalExpansion
        elif (isinstance(base_kernel.get_base_kernel(), YukawaKernel)
                and base_kernel.dim == 2):
            return Y2DLocalExpansion
        elif isinstance(base_kernel.get_base_kernel(), HelmholtzKernel):
            return HelmholtzConformingVolumeTaylorLocalExpansion
        elif isinstance(base_kernel.get_base_kernel(), LaplaceKernel):
            return LaplaceConformingVolumeTaylorLocalExpansion
        elif isinstance(base_kernel.get_base_kernel(),
                (BiharmonicKernel, StokesletKernel, StressletKernel)):
            return BiharmonicConformingVolumeTaylorLocalExpansion
        else:
            return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.kernel import (HelmholtzKernel, LaplaceKernel, YukawaKernel,
                BiharmonicKernel, StokesletKernel, StressletKernel)

        from sumpy.expansion.multipole import (H2DMultipoleExpansion,
                Y2DMultipoleExpansion,
                LaplaceConformingVolumeTaylorMultipoleExpansion,
                HelmholtzConformingVolumeTaylorMultipoleExpansion,
                BiharmonicConformingVolumeTaylorMultipoleExpansion,
                VolumeTaylorMultipoleExpansion)

        if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
                and base_kernel.dim == 2):
            return H2DMultipoleExpansion
        elif (isinstance(base_kernel.get_base_kernel(), YukawaKernel)
                and base_kernel.dim == 2):
            return Y2DMultipoleExpansion
        elif isinstance(base_kernel.get_base_kernel(), LaplaceKernel):
            return LaplaceConformingVolumeTaylorMultipoleExpansion
        elif isinstance(base_kernel.get_base_kernel(), HelmholtzKernel):
            return HelmholtzConformingVolumeTaylorMultipoleExpansion
        elif isinstance(base_kernel.get_base_kernel(),
                (BiharmonicKernel, StokesletKernel, StressletKernel)):
            return BiharmonicConformingVolumeTaylorMultipoleExpansion
        else:
            return VolumeTaylorMultipoleExpansion

# }}}


# vim: fdm=marker
