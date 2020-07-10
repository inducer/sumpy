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

import logging
from pytools import memoize_method
import sumpy.symbolic as sym
from sumpy.tools import add_mi
from .diff_op import make_identity_diff_op, laplacian

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

class CSEMatVecOperator(object):
    """
    A class to facilitate a fast matrix vector multiplication with
    common subexpression eliminated. In compressed Taylor
    series, the compression matrix's operation count can be
    reduced to `O(p**d)` from `O(p**{2d-1})` using CSE.

    .. attribute:: assignments

        An object of type
        ``List[Tuple[List[Tuple[int, Any]], List[Tuple[int, Any]]]]``.
        Each element in the list represents a row of the Matrix using a tuple
        of linear combinations. In the tuple, the first argument is a list of tuples
        representing ``(index of input vector, coeff)`` and the second argument of a
        list of tuples representing ``(index of output vector, coeff)``.

        Number of rows in the matrix represented is equal to the
        length of the `assignments` list.

    .. attribute:: shape

        Shape of the matrix as a tuple.
    """

    def __init__(self, assignments, shape):
        self.assignments = assignments
        self.shape = shape
        assert len(self.assignments) == shape[0]

    def matvec(self, vec, wrap_intermediate=lambda x: x):
        """
        :arg vec: vector for the matrix vector multiplication

        :arg wrap_intermediate: a function to wrap intermediate expressions
             If not given, the number of operations might grow in the
             final expressions in the vector resulting in an expensive matvec.
        """
        assert len(vec) == self.shape[1]
        res = []
        for input_vec_list, output_vec_list in self.assignments:
            value = 0
            for input_index, coeff in input_vec_list:
                value += vec[input_index] * coeff
            for output_index, coeff in output_vec_list:
                value += res[output_index] * coeff
            res.append(wrap_intermediate(value))
        return res

    def transpose_matvec(self, vec, wrap_intermediate=lambda x: x):
        assert len(vec) == self.shape[0]
        res = [0]*self.shape[1]
        expr_all = list(vec)
        for i, (input_vec_list, output_vec_list) in \
                reversed(list(enumerate(self.assignments))):
            for output_index, coeff in output_vec_list:
                expr_all[output_index] += expr_all[i] * coeff
                expr_all[output_index] = wrap_intermediate(expr_all[output_index])
            for input_index, coeff in input_vec_list:
                res[input_index] += expr_all[i] * coeff
        return res

# }}}


class LinearPDEBasedExpansionTermsWrangler(ExpansionTermsWrangler):
    """
    .. automethod:: __init__
    .. automethod:: get_pde_as_diff_op
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

        def save_intermediate(expr):
            if sac is None:
                return expr
            return sym.Symbol(sac.assign_unique("projection_temp", expr))

        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.matvec(stored_kernel_derivatives,
                                        save_intermediate)

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale, sac=None):

        def save_intermediate(expr):
            if sac is None:
                return expr
            return sym.Symbol(sac.assign_unique("compress_temp", expr))

        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.transpose_matvec(full_mpole_coefficients,
                                                  save_intermediate)

    @property
    def stored_identifiers(self):
        stored_identifiers, _ = self.get_stored_ids_and_unscaled_projection_matrix()
        return stored_identifiers

    def get_pde_as_diff_op(self):
        r"""
        Returns the PDE as a :class:`sumpy.expansion.diff_op.DifferentialOperator`
        object `L` where `L(u) = 0` is the PDE.
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

        pde_dict = self.get_pde_as_diff_op().mi_to_coeff
        for ident in pde_dict.keys():
            if ident not in coeff_ident_enumerate_dict:
                # Order of the expansion is less than the order of the PDE.
                # In that case, the compression matrix is the identity matrix
                # and there's nothing to project
                projection_matrix_assignments = \
                    [([(i, 1)], []) for i in range(len(mis))]
                shape = (len(mis), len(mis))
                return mis, CSEMatVecOperator(projection_matrix_assignments, shape)

        # Calculate the multi-index that appears last in in the PDE in
        # reverse degree lexicographic order.
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

        stored_identifiers = []

        projection_matrix_assignments = []
        for i, mi in enumerate(mis):
            # If the multi-index is to be stored, keep the projection matrix
            # entry empty
            if is_stored(mi):
                idx = len(stored_identifiers)
                stored_identifiers.append(mi)
                projection_matrix_assignments.append(([(idx, 1)], []))
                continue
            diff = [mi[d] - max_mi[d] for d in range(self.dim)]

            # eg: u_xx + u_yy + u_zz is represented as
            # [((2, 0, 0), 1), ((0, 2, 0), 1), ((0, 0, 2), 1)]
            assignment = []
            for other_mi, coeff in iteritems(pde_dict):
                j = coeff_ident_enumerate_dict[add_mi(other_mi, diff)]
                if i == j:
                    # Skip the u_zz part here.
                    continue
                # PDE might not have max_mi_coeff = -1, divide by -max_mi_coeff
                # to get a relation of the form, u_zz = - u_xx - u_yy for Laplace 3D.
                assignment.append((j, coeff*max_mi_mult))
            projection_matrix_assignments.append(([], assignment))

        plog.done()

        logger.debug("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(stored_identifiers)))

        shape = (len(mis), len(stored_identifiers))
        op = CSEMatVecOperator(projection_matrix_assignments, shape)
        return stored_identifiers, op

    @memoize_method
    def get_projection_matrix(self, rscale):
        """
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

        the projection matrix `M` is the transpose of the coefficient matrix
        """
        _, projection_matrix = \
            self.get_stored_ids_and_unscaled_projection_matrix()

        full_coeffs = self.get_full_coefficient_identifiers()

        projection_with_rscale = []
        for row, assignment in enumerate(projection_matrix.assignments):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[row])
            assignment_with_rscale = [assignment[0].copy(), []]
            for k, coeff in assignment[1]:
                diff = row_rscale - sum(full_coeffs[k])
                mult = rscale**diff
                assignment_with_rscale[1].append((k, coeff * mult))
            projection_with_rscale.append(tuple(assignment_with_rscale))

        shape = projection_matrix.shape
        return CSEMatVecOperator(projection_with_rscale, shape)


class LaplaceExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        super(LaplaceExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pde_as_diff_op(self):
        w = make_identity_diff_op(self.dim)
        return laplacian(w)


class HelmholtzExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "helmholtz_k_name", "max_mi")

    def __init__(self, order, dim, helmholtz_k_name, max_mi=None):
        self.helmholtz_k_name = helmholtz_k_name
        super(HelmholtzExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pde_as_diff_op(self, **kwargs):
        w = make_identity_diff_op(self.dim)
        k = sym.Symbol(self.helmholtz_k_name)
        return (laplacian(w) + k**2 * w)


class BiharmonicExpansionTermsWrangler(LinearPDEBasedExpansionTermsWrangler):

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        super(BiharmonicExpansionTermsWrangler, self).__init__(order=order, dim=dim,
            max_mi=max_mi)

    def get_pde_as_diff_op(self, **kwargs):
        w = make_identity_diff_op(self.dim)
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
