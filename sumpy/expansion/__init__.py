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
import logging
from pytools import memoize_method
import sumpy.symbolic as sym
from sumpy.tools import add_mi
from .pde import make_pde_sym, laplacian

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

    def coefficients_from_source(self, avec, bvec, rscale, sac=None):
        """Form an expansion from a source point.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        raise NotImplementedError

    def evaluate(self, coeffs, bvec, rscale, sac=None):
        """
        :return: a :mod:`sympy` expression corresponding
            to the evaluated expansion with the coefficients
            in *coeffs*.
        """

        raise NotImplementedError

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None):
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
            rscale, sac=None):
        raise NotImplementedError

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale, sac=None):
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

        if self.max_mi is None:
            return res

        return [mi for mi in res if
            all(mi[i] <= self.max_mi[i] for i in range(self.dim))]

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
            rscale, sac=None):
        return stored_kernel_derivatives

    get_stored_mpole_coefficients_from_full = (
            get_full_kernel_derivatives_from_stored)


# {{{ sparse matrix-vector multiplication

class CSEMatVec(object):
    """
    A class to facilitate a fast matrix vector multiplication with
    common subexpression eliminated. In compressed Taylor
    series, the compression matrix's operation count can be
    reduced to `O(p**d)` from `O(p**{2d-1})` using CSE.

    .. attribute:: assignments

        A list of tuples where the first argument of the tuple
        is a row and the second argument is a list. If the
        second argument is empty, the row is equal to the first
        unused element from the vector. If not, the second
        argument gives a list of tuples indicating a linear
        combination of previously calculated row.

        The matrix represented is a square matrix with the number
        of rows equal to the length of the `assignments` list.
    """

    def __init__(self, assignments):
        self.assignments = assignments

    def matvec(self, vec, sac):
        res = [0] * len(self.assignments)
        stored_idx = 0
        for row, deps in enumerate(self.assignments):
            if len(deps) == 0:
                res[row] = vec[stored_idx]
                stored_idx += 1
            else:
                for k, v in deps:
                    res[row] += res[k] * v
            if sac is not None:
                new_sym = sym.Symbol(sac.assign_unique("projection_temp", res[row]))
                res[row] = new_sym
        return res

    def transpose_matvec(self, vec, sac):
        res = []
        expr_all = list(vec)
        for row, deps in reversed(list(enumerate(self.assignments))):
            if len(deps) == 0:
                res.append(expr_all[row])
                continue
            if sac is not None:
                new_sym = sym.Symbol(sac.assign_unique("compress_temp", expr_all[row]))
                for k, v in deps:
                    expr_all[k] += new_sym * v
            else:
                for k, v in deps:
                    expr_all[k] += sym.sympify(expr_all[row] * v).expand(deep=False)
        res.reverse()
        return res

# }}}


class LinearPDEBasedExpansionTermsWrangler(ExpansionTermsWrangler):
    """
    .. automethod:: __init__
    .. automethod:: get_pde
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
            rscale, sac=None):
        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.matvec(stored_kernel_derivatives, sac)

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale, sac=None):
        # = M^T x, where M = projection matrix
        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.transpose_matvec(full_mpole_coefficients, sac)

    @property
    def stored_identifiers(self):
        stored_identifiers, _ = self.get_stored_ids_and_unscaled_projection_matrix()
        return stored_identifiers

    def get_pde(self):
        r"""
        Returns a PDE. A PDE stores a dictionary of (mi, coeff)
        where mi is the multi-index of the  derivative and coeff is the
        coefficient
        """

        raise NotImplementedError

    @memoize_method
    def get_stored_ids_and_unscaled_projection_matrix(self):
        from six import iteritems

        from pytools import ProcessLogger
        plog = ProcessLogger(logger, "compute PDE for Taylor coefficients")

        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = {tuple(mi): i for
                                            (i, mi) in enumerate(mis)}

        pde_dict = self.get_pde().eq
        for ident in pde_dict.keys():
            if ident not in coeff_ident_enumerate_dict:
                # Order of the expansion is less than the order of the PDE.
                # In that case, the compression matrix is the identity matrix
                # and there's nothing to project
                projection_matrix_assignments = [list() for i in range(len(mis))]
                return mis, CSEMatVec(projection_matrix_assignments)

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

        projection_matrix_assignments = []
        for i, mi in enumerate(mis):
            projection_matrix_assignments.append([])
            if is_stored(mi):
                continue
            diff = [mi[d] - max_mi[d] for d in range(self.dim)]
            for other_mi, coeff in iteritems(pde_dict):
                j = coeff_ident_enumerate_dict[add_mi(other_mi, diff)]
                if i == j:
                    continue
                # PDE might not have max_mi_coeff = -1, divide by -max_mi_coeff
                # to get a relation of the form, u_zz = - u_xx - u_yy for Laplace 3D.
                projection_matrix_assignments[i].append((j, coeff*max_mi_mult))

        plog.done()

        logger.debug("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(stored_identifiers)))

        return stored_identifiers, CSEMatVec(projection_matrix_assignments)

    @memoize_method
    def get_projection_matrix(self, rscale):
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

        the projection matrix `M` is the transpose of the coefficient matrix
        """
        _, projection_matrix = \
            self.get_stored_ids_and_unscaled_projection_matrix()

        full_coeffs = self.get_full_coefficient_identifiers()

        projection_with_rscale = []
        for row, deps in enumerate(projection_matrix.assignments):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[row])
            deps_with_rscale = []
            for k, coeff in deps:
                diff = row_rscale - sum(full_coeffs[k])
                mult = rscale**diff
                deps_with_rscale.append((k, coeff * mult))
            projection_with_rscale.append(deps_with_rscale)

        return CSEMatVec(projection_with_rscale)


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
